from __future__ import annotations

import argparse
import traceback

import torch

from anna.runtime.device import RuntimeSafetyPolicy
from anna.runtime.model_runtime_loader import load_model_runtime_from_model_dir
from anna.runtime.qwen3_5_text_engine import GenerationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default=r"D:\Projects\Anna\models\Intel\Qwen3___5-35B-A3B-int4-AutoRound",
    )
    parser.add_argument("--resident-expert-layers", type=int, default=None)
    parser.add_argument("--cached-experts-per-layer", type=int, default=None)
    parser.add_argument("--offload-mode", default="auto")
    parser.add_argument("--expert-quant", default="auto")
    parser.add_argument("--offload-token-io", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    engine = load_model_runtime_from_model_dir(
        model_dir=args.model_dir,
        model_id="qwen3.5",
        device="xpu",
        dtype="bf16",
        weight_quant="none",
        offload_vision=True,
        default_enable_thinking=False,
        reasoning_format="deepseek",
        offload_mode=args.offload_mode,
        expert_quant=args.expert_quant,
        resident_expert_layers=args.resident_expert_layers,
        cached_experts_per_layer=args.cached_experts_per_layer,
        safety_policy=RuntimeSafetyPolicy(
            min_free_bytes=256 << 20,
            reserve_margin_bytes=128 << 20,
            max_estimated_usage_ratio=0.95,
            generation_memory_safety_factor=1.25,
        ),
    )
    if args.offload_token_io:
        resident_indices = tuple(engine._resident_expert_layer_indices())
        engine.model.configure_runtime(
            engine.device_context.device,
            offload_experts=engine.offload_mode == "experts",
            offload_vision=engine.offload_vision,
            offload_token_io=True,
            resident_expert_layers=0,
            resident_expert_layer_indices=resident_indices,
            expert_quant=engine.expert_quant,
            cached_experts_per_layer=engine.cached_experts_per_layer,
        )
        engine.model.tie_weights()
    print("ENGINE_LOADED")
    print(engine.health()["memory"])
    print(
        {
            "offload_mode": engine.offload_mode,
            "expert_quant": engine.expert_quant,
            "resident_expert_layers": engine.resident_expert_layers,
            "resident_expert_layer_indices": engine._resident_expert_layer_indices(),
            "cached_experts_per_layer": engine.cached_experts_per_layer,
        }
    )
    for name, probe in (
        ("float_probe", torch.arange(8, dtype=torch.float32)),
        ("long_probe", torch.arange(8, dtype=torch.long)),
    ):
        try:
            moved = probe.to("xpu")
            print(name, "OK", moved.dtype, moved.device)
        except Exception as exc:  # pragma: no cover - smoke runner
            print(name, "FAIL", type(exc).__name__, exc)
    prompt_messages = [{"role": "user", "content": "hi"}]
    prepared = engine._prepare_messages(prompt_messages, enable_thinking=False)
    prepared_for_generation = engine._move_prepared_for_generation(
        prepared,
        config=GenerationConfig(
            max_new_tokens=8,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
        ),
    )
    print(
        "prepared",
        {
            "input_ids_shape": None if prepared_for_generation.input_ids is None else tuple(prepared_for_generation.input_ids.shape),
            "attention_mask_shape": None
            if prepared_for_generation.attention_mask is None
            else tuple(prepared_for_generation.attention_mask.shape),
            "input_ids_device": None
            if prepared_for_generation.input_ids is None
            else str(prepared_for_generation.input_ids.device),
        },
    )
    try:
        language_model = engine.model.model.language_model
        batch_size, seq_len = prepared_for_generation.input_ids.shape
        embed_device = language_model.embed_tokens.weight.device
        input_ids_cpu = prepared_for_generation.input_ids.to(embed_device)
        inputs_embeds_cpu = language_model.embed_tokens(input_ids_cpu)
        print(
            "embedding_probe",
            {
                "embed_device": str(embed_device),
                "input_ids_cpu_device": str(input_ids_cpu.device),
                "inputs_embeds_shape": tuple(inputs_embeds_cpu.shape),
                "inputs_embeds_dtype": str(inputs_embeds_cpu.dtype),
                "inputs_embeds_contiguous": inputs_embeds_cpu.is_contiguous(),
                "inputs_embeds_stride": tuple(inputs_embeds_cpu.stride()),
            },
        )
        inputs_embeds_xpu = inputs_embeds_cpu.contiguous().to("xpu")
        print(
            "embedding_probe_xpu OK",
            tuple(inputs_embeds_xpu.shape),
            inputs_embeds_xpu.dtype,
            inputs_embeds_xpu.device,
        )
        torch.xpu.synchronize()
        print("embedding_probe_sync OK")
        position_ids = torch.arange(seq_len, dtype=torch.long).view(1, -1).expand(batch_size, -1)
        rotary = language_model.rotary_emb
        expanded_position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq = rotary.cpu_inv_freq
        inv_freq_expanded = inv_freq[None, None, :, None].float().expand(3, expanded_position_ids.shape[1], -1, 1)
        position_ids_expanded = expanded_position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        freqs = rotary.apply_interleaved_mrope(freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cpu = emb.cos() * rotary.attention_scaling
        sin_cpu = emb.sin() * rotary.attention_scaling
        print(
            "rotary_cpu_shapes",
            {
                "cos_shape": tuple(cos_cpu.shape),
                "sin_shape": tuple(sin_cpu.shape),
                "cos_contiguous": cos_cpu.is_contiguous(),
                "sin_contiguous": sin_cpu.is_contiguous(),
                "cos_stride": tuple(cos_cpu.stride()),
                "sin_stride": tuple(sin_cpu.stride()),
            },
        )
        generic = torch.zeros_like(cos_cpu)
        generic_xpu = generic.to("xpu")
        print("rotary_generic_probe OK", tuple(generic_xpu.shape), generic_xpu.dtype, generic_xpu.device)
        cos_xpu = cos_cpu.contiguous().to("xpu")
        sin_xpu = sin_cpu.contiguous().to("xpu")
        print("rotary_probe OK", tuple(cos_xpu.shape), cos_xpu.dtype, cos_xpu.device, tuple(sin_xpu.shape), sin_xpu.dtype, sin_xpu.device)
        torch.xpu.synchronize()
        print("rotary_probe_sync OK")
        layernorm = language_model.layers[0].input_layernorm
        for name, tensor in (
            ("layernorm_probe_actual", inputs_embeds_xpu),
            ("layernorm_probe_cloned", inputs_embeds_xpu.clone()),
            ("layernorm_probe_random", torch.randn_like(inputs_embeds_xpu)),
        ):
            try:
                layernorm_output = layernorm(tensor)
                torch.xpu.synchronize()
                print(
                    name,
                    "OK",
                    tuple(layernorm_output.shape),
                    layernorm_output.dtype,
                    layernorm_output.device,
                )
            except Exception as probe_exc:  # pragma: no cover - smoke runner
                print(name, "FAIL", type(probe_exc).__name__, probe_exc)
                break
    except Exception as exc:  # pragma: no cover - smoke runner
        print("rotary_probe FAIL", type(exc).__name__, exc)
    try:
        output = engine.generate_chat(
            messages=prompt_messages,
            config=GenerationConfig(
                max_new_tokens=8,
                temperature=0.0,
                top_p=1.0,
                top_k=1,
            ),
            enable_thinking=False,
            reasoning_format="deepseek",
        )
        print("GEN_OK")
        print(output)
    except Exception as exc:  # pragma: no cover - smoke runner
        print("GEN_FAIL", type(exc).__name__, exc)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
