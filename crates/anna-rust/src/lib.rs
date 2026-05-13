use half::{bf16, f16};
use pyo3::exceptions::{PyFileNotFoundError, PyKeyError, PyRuntimeError, PyValueError};
use pyo3::types::PyByteArray;
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: std::collections::HashMap<String, String>,
}

#[derive(Clone)]
struct TensorEntry {
    name: String,
    dtype: String,
    shape: Vec<u64>,
    data_offsets: (u64, u64),
}

#[derive(Clone)]
struct ShardPlan {
    path: PathBuf,
    tensors: Vec<TensorEntry>,
    size_bytes: u64,
    header_len: u64,
}

fn path_to_string(path: &Path) -> PyResult<String> {
    path.to_str()
        .map(|value| value.to_owned())
        .ok_or_else(|| PyValueError::new_err(format!("Path is not valid UTF-8: {}", path.display())))
}

fn file_size(path: &Path) -> PyResult<u64> {
    fs::metadata(path)
        .map(|metadata| metadata.len())
        .map_err(|exc| PyRuntimeError::new_err(format!("Failed to stat {}: {exc}", path.display())))
}

fn read_safetensors_header(path: &Path) -> PyResult<(u64, BTreeMap<String, TensorEntry>)> {
    let mut file = File::open(path)
        .map_err(|exc| PyRuntimeError::new_err(format!("Failed to open {}: {exc}", path.display())))?;
    let mut header_len_bytes = [0_u8; 8];
    file.read_exact(&mut header_len_bytes).map_err(|exc| {
        PyRuntimeError::new_err(format!(
            "Failed to read safetensors header length from {}: {exc}",
            path.display()
        ))
    })?;
    let header_len = u64::from_le_bytes(header_len_bytes);
    if header_len == 0 || header_len > 512 * 1024 * 1024 {
        return Err(PyValueError::new_err(format!(
            "Invalid safetensors header length {} in {}",
            header_len,
            path.display()
        )));
    }
    file.seek(SeekFrom::Start(8)).map_err(|exc| {
        PyRuntimeError::new_err(format!("Failed to seek safetensors header in {}: {exc}", path.display()))
    })?;
    let mut header = vec![0_u8; header_len as usize];
    file.read_exact(&mut header).map_err(|exc| {
        PyRuntimeError::new_err(format!("Failed to read safetensors header from {}: {exc}", path.display()))
    })?;
    let header: Value = serde_json::from_slice(&header).map_err(|exc| {
        PyValueError::new_err(format!("Invalid safetensors header JSON in {}: {exc}", path.display()))
    })?;
    let object = header.as_object().ok_or_else(|| {
        PyValueError::new_err(format!("Safetensors header is not an object in {}", path.display()))
    })?;
    let mut entries = BTreeMap::new();
    for (name, value) in object {
        if name == "__metadata__" {
            continue;
        }
        let tensor = value.as_object().ok_or_else(|| {
            PyValueError::new_err(format!("Invalid safetensors tensor entry {name:?} in {}", path.display()))
        })?;
        let dtype = tensor
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| PyValueError::new_err(format!("Missing dtype for tensor {name:?} in {}", path.display())))?
            .to_owned();
        let shape = tensor
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| PyValueError::new_err(format!("Missing shape for tensor {name:?} in {}", path.display())))?
            .iter()
            .map(|dim| {
                dim.as_u64().ok_or_else(|| {
                    PyValueError::new_err(format!("Invalid shape dim for tensor {name:?} in {}", path.display()))
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let offsets = tensor
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                PyValueError::new_err(format!("Missing data_offsets for tensor {name:?} in {}", path.display()))
            })?;
        if offsets.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "Invalid data_offsets length for tensor {name:?} in {}",
                path.display()
            )));
        }
        let start = offsets[0].as_u64().ok_or_else(|| {
            PyValueError::new_err(format!("Invalid data_offsets start for tensor {name:?} in {}", path.display()))
        })?;
        let end = offsets[1].as_u64().ok_or_else(|| {
            PyValueError::new_err(format!("Invalid data_offsets end for tensor {name:?} in {}", path.display()))
        })?;
        entries.insert(
            name.clone(),
            TensorEntry {
                name: name.clone(),
                dtype,
                shape,
                data_offsets: (start, end),
            },
        );
    }
    Ok((header_len, entries))
}

fn build_safetensors_plan(model_dir: &str) -> PyResult<(Vec<ShardPlan>, u64)> {
    let model_path = PathBuf::from(model_dir);
    let index_path = model_path.join("model.safetensors.index.json");

    if index_path.exists() {
        let raw = fs::read_to_string(&index_path).map_err(|exc| {
            PyRuntimeError::new_err(format!("Failed to read {}: {exc}", index_path.display()))
        })?;
        let index: SafetensorsIndex = serde_json::from_str(&raw).map_err(|exc| {
            PyValueError::new_err(format!("Invalid safetensors index {}: {exc}", index_path.display()))
        })?;
        if index.weight_map.is_empty() {
            return Err(PyKeyError::new_err("safetensors index weight_map is empty"));
        }

        let mut keys_by_shard = BTreeMap::<String, Vec<String>>::new();
        for (key, shard) in index.weight_map.iter() {
            keys_by_shard.entry(shard.clone()).or_default().push(key.clone());
        }
        let mut files = Vec::with_capacity(keys_by_shard.len());
        for (shard, mut keys) in keys_by_shard {
            keys.sort();
            let path = model_path.join(shard);
            let size_bytes = file_size(&path)?;
            let (header_len, header_entries) = read_safetensors_header(&path)?;
            let tensors = keys
                .iter()
                .map(|key| {
                    header_entries.get(key).cloned().ok_or_else(|| {
                        PyKeyError::new_err(format!("Tensor {key:?} is in index but missing from {}", path.display()))
                    })
                })
                .collect::<PyResult<Vec<_>>>()?;
            files.push(ShardPlan {
                path,
                tensors,
                size_bytes,
                header_len,
            });
        }
        let total_bytes = files.iter().map(|plan| plan.size_bytes).sum();
        return Ok((files, total_bytes));
    }

    let direct_file = model_path.join("model.safetensors");
    if direct_file.exists() {
        let size_bytes = file_size(&direct_file)?;
        let (header_len, header_entries) = read_safetensors_header(&direct_file)?;
        return Ok((
            vec![ShardPlan {
                path: direct_file,
                tensors: header_entries.into_values().collect(),
                size_bytes,
                header_len,
            }],
            size_bytes,
        ));
    }

    let mut files = Vec::new();
    let entries = fs::read_dir(&model_path).map_err(|exc| {
        PyFileNotFoundError::new_err(format!("Failed to read model directory {}: {exc}", model_path.display()))
    })?;
    for entry in entries {
        let entry = entry.map_err(|exc| PyRuntimeError::new_err(format!("Failed to read directory entry: {exc}")))?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "safetensors") {
            files.push(path);
        }
    }
    files.sort();
    if files.is_empty() {
        return Err(PyFileNotFoundError::new_err(format!(
            "No safetensors weights found in {}",
            model_path.display()
        )));
    }
    let mut plans = Vec::with_capacity(files.len());
    for path in files {
        let size_bytes = file_size(&path)?;
        let (header_len, header_entries) = read_safetensors_header(&path)?;
        plans.push(ShardPlan {
            path,
            tensors: header_entries.into_values().collect(),
            size_bytes,
            header_len,
        });
    }
    let total_bytes = plans.iter().map(|plan| plan.size_bytes).sum();
    Ok((plans, total_bytes))
}

/// Return (weight_files, total_bytes) for a Hugging Face safetensors model directory.
#[pyfunction]
fn inspect_safetensors_manifest(model_dir: &str) -> PyResult<(Vec<String>, u64)> {
    let (plans, total_bytes) = build_safetensors_plan(model_dir)?;
    let file_strings = plans
        .iter()
        .map(|plan| path_to_string(&plan.path))
        .collect::<PyResult<Vec<_>>>()?;
    Ok((file_strings, total_bytes))
}

/// Return (shard_plans, total_bytes) where each shard plan is
/// (path, size_bytes, header_len, tensor_entries). Tensor entries are
/// (name, dtype, shape, data_start, data_end); offsets are safetensors
/// data-section offsets, so absolute file offsets start at 8 + header_len.
#[pyfunction]
fn inspect_safetensors_load_plan(
    model_dir: &str,
) -> PyResult<(Vec<(String, u64, u64, Vec<(String, String, Vec<u64>, u64, u64)>)>, u64)> {
    let (plans, total_bytes) = build_safetensors_plan(model_dir)?;
    let py_plans = plans
        .iter()
        .map(|plan| {
            let entries = plan
                .tensors
                .iter()
                .map(|entry| {
                    Ok((
                        entry.name.clone(),
                        entry.dtype.clone(),
                        entry.shape.clone(),
                        entry.data_offsets.0,
                        entry.data_offsets.1,
                    ))
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok((path_to_string(&plan.path)?, plan.size_bytes, plan.header_len, entries))
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok((py_plans, total_bytes))
}

fn read_f32(buffer: &[u8], index: usize, dtype: &str) -> PyResult<f32> {
    match dtype {
        "F32" => {
            let start = index * 4;
            Ok(f32::from_le_bytes(buffer[start..start + 4].try_into().unwrap()))
        }
        "F16" => {
            let start = index * 2;
            let bits = u16::from_le_bytes(buffer[start..start + 2].try_into().unwrap());
            Ok(f16::from_bits(bits).to_f32())
        }
        "BF16" => {
            let start = index * 2;
            let bits = u16::from_le_bytes(buffer[start..start + 2].try_into().unwrap());
            Ok(bf16::from_bits(bits).to_f32())
        }
        _ => Err(PyValueError::new_err(format!(
            "Rust int4 quantization supports F32/F16/BF16 safetensors, got {dtype}"
        ))),
    }
}

struct QuantizeLinearSpec {
    name: String,
    data_start: u64,
    data_end: u64,
    dtype: String,
    out_features: usize,
    in_features: usize,
    group_size: usize,
    padded_in_features: usize,
}

struct QuantizedLinear {
    name: String,
    qweight: Vec<u8>,
    qscale: Vec<u8>,
    qzeros: Vec<u8>,
    out_features: usize,
    padded_in_features: usize,
    group_size: usize,
}

fn elem_size(dtype: &str) -> PyResult<usize> {
    match dtype {
        "F32" => Ok(4),
        "F16" | "BF16" => Ok(2),
        _ => Err(PyValueError::new_err(format!(
            "Rust int4 quantization supports F32/F16/BF16 safetensors, got {dtype}"
        ))),
    }
}

fn validate_quantize_spec(spec: &QuantizeLinearSpec) -> PyResult<u64> {
    if spec.group_size == 0 || spec.padded_in_features % spec.group_size != 0 || spec.padded_in_features % 8 != 0 {
        return Err(PyValueError::new_err("Invalid group_size or padded_in_features for int4 quantization"));
    }
    if spec.padded_in_features < spec.in_features {
        return Err(PyValueError::new_err("padded_in_features cannot be smaller than in_features"));
    }
    let expected_bytes = spec
        .out_features
        .checked_mul(spec.in_features)
        .and_then(|count| count.checked_mul(elem_size(&spec.dtype).ok()?))
        .ok_or_else(|| PyValueError::new_err("Tensor byte size overflow"))? as u64;
    if spec.data_end < spec.data_start || spec.data_end - spec.data_start != expected_bytes {
        return Err(PyValueError::new_err(format!(
            "Tensor {} byte range does not match shape: range={} expected={expected_bytes}",
            spec.name,
            spec.data_end.saturating_sub(spec.data_start)
        )));
    }
    Ok(expected_bytes)
}

fn quantize_linear_raw(spec: &QuantizeLinearSpec, raw: &[u8]) -> PyResult<QuantizedLinear> {
    let group_count = spec.padded_in_features / spec.group_size;
    let packed_cols = spec.padded_in_features / 8;
    let row_results = (0..spec.out_features)
        .into_par_iter()
        .map(|out_idx| {
            let mut row_qweight = vec![0_u8; packed_cols * 4];
            let mut row_qscale = vec![0_u8; group_count * 4];
            for group_idx in 0..group_count {
                let group_base = group_idx * spec.group_size;
                let valid = spec.in_features.saturating_sub(group_base).min(spec.group_size);
                let mut max_abs = 0.0_f32;
                for offset in 0..valid {
                    let value = read_f32(raw, out_idx * spec.in_features + group_base + offset, &spec.dtype)?;
                    max_abs = max_abs.max(value.abs());
                }
                let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
                row_qscale[group_idx * 4..group_idx * 4 + 4].copy_from_slice(&scale.to_le_bytes());

                for offset in 0..spec.group_size {
                    let in_idx = group_base + offset;
                    let value = if in_idx < spec.in_features {
                        read_f32(raw, out_idx * spec.in_features + in_idx, &spec.dtype)?
                    } else {
                        0.0
                    };
                    let quantized = ((value / scale) + 8.0).round().clamp(0.0, 15.0) as u32;
                    let packed_col = in_idx / 8;
                    let lane = in_idx % 8;
                    let packed_index = packed_col * 4;
                    let current = u32::from_le_bytes(row_qweight[packed_index..packed_index + 4].try_into().unwrap());
                    let updated = current | (quantized << (lane * 4));
                    row_qweight[packed_index..packed_index + 4].copy_from_slice(&updated.to_le_bytes());
                }
            }
            Ok::<_, PyErr>((out_idx, row_qweight, row_qscale))
        })
        .collect::<PyResult<Vec<_>>>()?;

    let mut qweight = vec![0_u8; spec.out_features * packed_cols * 4];
    let mut qscale = vec![0_u8; group_count * spec.out_features * 4];
    for (out_idx, row_qweight, row_qscale) in row_results {
        let qweight_start = out_idx * packed_cols * 4;
        qweight[qweight_start..qweight_start + packed_cols * 4].copy_from_slice(&row_qweight);
        for group_idx in 0..group_count {
            let dst = (group_idx * spec.out_features + out_idx) * 4;
            let src = group_idx * 4;
            qscale[dst..dst + 4].copy_from_slice(&row_qscale[src..src + 4]);
        }
    }
    let qzeros = vec![8_u8; group_count * spec.out_features];
    Ok(QuantizedLinear {
        name: spec.name.clone(),
        qweight,
        qscale,
        qzeros,
        out_features: spec.out_features,
        padded_in_features: spec.padded_in_features,
        group_size: spec.group_size,
    })
}

#[pyfunction]
fn quantize_safetensors_linear_int4<'py>(
    py: Python<'py>,
    shard_path: &str,
    header_len: u64,
    data_start: u64,
    data_end: u64,
    dtype: &str,
    out_features: usize,
    in_features: usize,
    group_size: usize,
    padded_in_features: usize,
) -> PyResult<(Bound<'py, PyByteArray>, Bound<'py, PyByteArray>, Bound<'py, PyByteArray>)> {
    let spec = QuantizeLinearSpec {
        name: "<single>".to_owned(),
        data_start,
        data_end,
        dtype: dtype.to_owned(),
        out_features,
        in_features,
        group_size,
        padded_in_features,
    };
    let expected_bytes = validate_quantize_spec(&spec)?;

    let absolute_start = 8 + header_len + data_start;
    let mut file = File::open(shard_path)
        .map_err(|exc| PyRuntimeError::new_err(format!("Failed to open {shard_path}: {exc}")))?;
    file.seek(SeekFrom::Start(absolute_start))
        .map_err(|exc| PyRuntimeError::new_err(format!("Failed to seek {shard_path}: {exc}")))?;
    let mut raw = vec![0_u8; expected_bytes as usize];
    file.read_exact(&mut raw)
        .map_err(|exc| PyRuntimeError::new_err(format!("Failed to read tensor bytes from {shard_path}: {exc}")))?;

    let quantized = py.allow_threads(|| quantize_linear_raw(&spec, &raw))?;

    Ok((
        PyByteArray::new_bound(py, &quantized.qweight),
        PyByteArray::new_bound(py, &quantized.qscale),
        PyByteArray::new_bound(py, &quantized.qzeros),
    ))
}

#[pyfunction]
fn quantize_safetensors_linear_int4_batch<'py>(
    py: Python<'py>,
    shard_path: &str,
    header_len: u64,
    specs: Vec<(String, u64, u64, String, usize, usize, usize, usize)>,
) -> PyResult<Vec<(String, Bound<'py, PyByteArray>, Bound<'py, PyByteArray>, Bound<'py, PyByteArray>, usize, usize, usize)>> {
    let specs = specs
        .into_iter()
        .map(
            |(name, data_start, data_end, dtype, out_features, in_features, group_size, padded_in_features)| {
                QuantizeLinearSpec {
                    name,
                    data_start,
                    data_end,
                    dtype,
                    out_features,
                    in_features,
                    group_size,
                    padded_in_features,
                }
            },
        )
        .collect::<Vec<_>>();
    let expected_sizes = specs
        .iter()
        .map(validate_quantize_spec)
        .collect::<PyResult<Vec<_>>>()?;
    let mut file = File::open(shard_path)
        .map_err(|exc| PyRuntimeError::new_err(format!("Failed to open {shard_path}: {exc}")))?;
    let mut raw_tensors = Vec::with_capacity(specs.len());
    for (spec, expected_bytes) in specs.iter().zip(expected_sizes.iter()) {
        let absolute_start = 8 + header_len + spec.data_start;
        file.seek(SeekFrom::Start(absolute_start))
            .map_err(|exc| PyRuntimeError::new_err(format!("Failed to seek {shard_path}: {exc}")))?;
        let mut raw = vec![0_u8; *expected_bytes as usize];
        file.read_exact(&mut raw)
            .map_err(|exc| PyRuntimeError::new_err(format!("Failed to read tensor {} from {shard_path}: {exc}", spec.name)))?;
        raw_tensors.push(raw);
    }
    let quantized = py.allow_threads(|| {
        specs
            .par_iter()
            .zip(raw_tensors.par_iter())
            .map(|(spec, raw)| quantize_linear_raw(spec, raw))
            .collect::<PyResult<Vec<_>>>()
    })?;
    Ok(quantized
        .into_iter()
        .map(|item| {
            (
                item.name,
                PyByteArray::new_bound(py, &item.qweight),
                PyByteArray::new_bound(py, &item.qscale),
                PyByteArray::new_bound(py, &item.qzeros),
                item.out_features,
                item.padded_in_features,
                item.group_size,
            )
        })
        .collect())
}

#[pymodule]
fn _rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(inspect_safetensors_manifest, module)?)?;
    module.add_function(wrap_pyfunction!(inspect_safetensors_load_plan, module)?)?;
    module.add_function(wrap_pyfunction!(quantize_safetensors_linear_int4, module)?)?;
    module.add_function(wrap_pyfunction!(quantize_safetensors_linear_int4_batch, module)?)?;
    Ok(())
}
