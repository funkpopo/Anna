#include "sycl_bridge.h"

#include <stdint.h>

#include <algorithm>
#include <cctype>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

namespace {

thread_local std::string g_last_error;

struct device_not_found_error : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

void set_last_error(std::string message) {
    g_last_error = std::move(message);
}

std::string lower_copy(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

template <typename Fn>
int guard(Fn&& fn) {
    try {
        fn();
        return ANNA_SYCL_SUCCESS;
    } catch (const sycl::exception& ex) {
        set_last_error(ex.what());
        return ANNA_SYCL_RUNTIME_ERROR;
    } catch (const device_not_found_error& ex) {
        set_last_error(ex.what());
        return ANNA_SYCL_DEVICE_NOT_FOUND;
    } catch (const std::invalid_argument& ex) {
        set_last_error(ex.what());
        return ANNA_SYCL_INVALID_ARGUMENT;
    } catch (const std::bad_alloc&) {
        set_last_error("SYCL allocation failed");
        return ANNA_SYCL_ALLOCATION_FAILED;
    } catch (const std::exception& ex) {
        set_last_error(ex.what());
        return ANNA_SYCL_RUNTIME_ERROR;
    } catch (...) {
        set_last_error("Unknown SYCL backend error");
        return ANNA_SYCL_RUNTIME_ERROR;
    }
}

struct runtime_impl {
    sycl::queue queue;
    std::string platform_name;
    std::string device_name;
    std::string vendor_name;

    explicit runtime_impl(sycl::device device)
        : queue(device),
          platform_name(device.get_platform().get_info<sycl::info::platform::name>()),
          device_name(device.get_info<sycl::info::device::name>()),
          vendor_name(device.get_info<sycl::info::device::vendor>()) {}
};

struct buffer_f32_impl {
    runtime_impl* runtime;
    float* data;
    std::size_t len;
};

struct dense_weights_impl {
    runtime_impl* runtime;
    float* weight;
    float* bias;
    std::size_t in_features;
    std::size_t out_features;
    bool has_bias;
};

struct autoround_weights_impl {
    runtime_impl* runtime;
    int* qweight;
    int* qzeros;
    float* scales;
    float* bias;
    std::size_t in_features;
    std::size_t out_features;
    std::size_t group_size;
    bool has_bias;
};

sycl::device pick_device() {
    const auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devices.empty()) {
        throw device_not_found_error("No SYCL GPU device available");
    }

    for (const auto& device : devices) {
        const std::string vendor = lower_copy(device.get_info<sycl::info::device::vendor>());
        const std::string name = lower_copy(device.get_info<sycl::info::device::name>());
        if (vendor.find("intel") != std::string::npos || name.find("arc") != std::string::npos) {
            return device;
        }
    }

    return devices.front();
}

template <typename T>
T* malloc_device_or_throw(runtime_impl* runtime, std::size_t len) {
    T* ptr = sycl::malloc_device<T>(len, runtime->queue);
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
    return ptr;
}

template <typename T>
T* copy_to_device(runtime_impl* runtime, const T* values, std::size_t len) {
    T* ptr = malloc_device_or_throw<T>(runtime, len);
    runtime->queue.memcpy(ptr, values, sizeof(T) * len).wait_and_throw();
    return ptr;
}

void validate_buffer(const buffer_f32_impl* buffer, std::size_t expected_len, const char* name) {
    if (buffer == nullptr) {
        throw std::invalid_argument(std::string(name) + " buffer is null");
    }
    if (buffer->len != expected_len) {
        throw std::invalid_argument(std::string(name) + " buffer length mismatch");
    }
}

} // namespace

extern "C" const char* anna_sycl_last_error_message(void) {
    return g_last_error.c_str();
}

extern "C" int anna_sycl_runtime_create(struct anna_sycl_runtime** out_runtime) {
    return guard([&] {
        if (out_runtime == nullptr) {
            throw std::invalid_argument("runtime output pointer is null");
        }
        sycl::device device = pick_device();
        auto runtime = std::make_unique<runtime_impl>(device);
        *out_runtime = reinterpret_cast<struct anna_sycl_runtime*>(runtime.release());
    });
}

extern "C" void anna_sycl_runtime_destroy(struct anna_sycl_runtime* runtime) {
    delete reinterpret_cast<runtime_impl*>(runtime);
}

extern "C" const char* anna_sycl_runtime_platform_name(const struct anna_sycl_runtime* runtime) {
    if (runtime == nullptr) return "";
    return reinterpret_cast<const runtime_impl*>(runtime)->platform_name.c_str();
}

extern "C" const char* anna_sycl_runtime_device_name(const struct anna_sycl_runtime* runtime) {
    if (runtime == nullptr) return "";
    return reinterpret_cast<const runtime_impl*>(runtime)->device_name.c_str();
}

extern "C" const char* anna_sycl_runtime_vendor_name(const struct anna_sycl_runtime* runtime) {
    if (runtime == nullptr) return "";
    return reinterpret_cast<const runtime_impl*>(runtime)->vendor_name.c_str();
}

extern "C" int anna_sycl_dense_create(
    struct anna_sycl_runtime* runtime,
    const float* weight,
    std::size_t weight_len,
    const float* bias,
    std::size_t bias_len,
    std::size_t out_features,
    std::size_t in_features,
    struct anna_sycl_dense_weights** out_weights) {
    return guard([&] {
        if (runtime == nullptr || out_weights == nullptr) {
            throw std::invalid_argument("dense create received null runtime or output");
        }
        if (weight == nullptr || weight_len != out_features * in_features) {
            throw std::invalid_argument("dense weight shape mismatch");
        }
        if (bias != nullptr && bias_len != out_features) {
            throw std::invalid_argument("dense bias shape mismatch");
        }

        auto* runtime_impl_ptr = reinterpret_cast<runtime_impl*>(runtime);
        auto handle = std::make_unique<dense_weights_impl>();
        handle->runtime = runtime_impl_ptr;
        handle->weight = copy_to_device(runtime_impl_ptr, weight, weight_len);
        handle->bias = bias != nullptr ? copy_to_device(runtime_impl_ptr, bias, bias_len) : nullptr;
        handle->in_features = in_features;
        handle->out_features = out_features;
        handle->has_bias = bias != nullptr;
        *out_weights = reinterpret_cast<struct anna_sycl_dense_weights*>(handle.release());
    });
}

extern "C" void anna_sycl_dense_destroy(struct anna_sycl_dense_weights* weights) {
    auto* handle = reinterpret_cast<dense_weights_impl*>(weights);
    if (handle == nullptr) return;
    sycl::free(handle->weight, handle->runtime->queue);
    if (handle->bias != nullptr) {
        sycl::free(handle->bias, handle->runtime->queue);
    }
    delete handle;
}

extern "C" int anna_sycl_autoround_create(
    struct anna_sycl_runtime* runtime,
    const int* qweight,
    std::size_t qweight_len,
    const int* qzeros,
    std::size_t qzeros_len,
    const float* scales,
    std::size_t scales_len,
    const float* bias,
    std::size_t bias_len,
    std::size_t out_features,
    std::size_t in_features,
    std::size_t group_size,
    struct anna_sycl_autoround_weights** out_weights) {
    return guard([&] {
        if (runtime == nullptr || out_weights == nullptr) {
            throw std::invalid_argument("autoround create received null runtime or output");
        }
        if (qweight == nullptr || qzeros == nullptr || scales == nullptr) {
            throw std::invalid_argument("autoround tensors must not be null");
        }
        if (bias != nullptr && bias_len != out_features) {
            throw std::invalid_argument("autoround bias shape mismatch");
        }

        auto* runtime_impl_ptr = reinterpret_cast<runtime_impl*>(runtime);
        auto handle = std::make_unique<autoround_weights_impl>();
        handle->runtime = runtime_impl_ptr;
        handle->qweight = copy_to_device(runtime_impl_ptr, qweight, qweight_len);
        handle->qzeros = copy_to_device(runtime_impl_ptr, qzeros, qzeros_len);
        handle->scales = copy_to_device(runtime_impl_ptr, scales, scales_len);
        handle->bias = bias != nullptr ? copy_to_device(runtime_impl_ptr, bias, bias_len) : nullptr;
        handle->in_features = in_features;
        handle->out_features = out_features;
        handle->group_size = group_size;
        handle->has_bias = bias != nullptr;
        *out_weights = reinterpret_cast<struct anna_sycl_autoround_weights*>(handle.release());
    });
}

extern "C" void anna_sycl_autoround_destroy(struct anna_sycl_autoround_weights* weights) {
    auto* handle = reinterpret_cast<autoround_weights_impl*>(weights);
    if (handle == nullptr) return;
    sycl::free(handle->qweight, handle->runtime->queue);
    sycl::free(handle->qzeros, handle->runtime->queue);
    sycl::free(handle->scales, handle->runtime->queue);
    if (handle->bias != nullptr) {
        sycl::free(handle->bias, handle->runtime->queue);
    }
    delete handle;
}

extern "C" int anna_sycl_buffer_upload_f32(
    struct anna_sycl_runtime* runtime,
    const float* values,
    std::size_t len,
    struct anna_sycl_buffer_f32** out_buffer) {
    return guard([&] {
        if (runtime == nullptr || out_buffer == nullptr) {
            throw std::invalid_argument("buffer upload received null runtime or output");
        }
        if (values == nullptr && len != 0) {
            throw std::invalid_argument("buffer upload values are null");
        }

        auto* runtime_impl_ptr = reinterpret_cast<runtime_impl*>(runtime);
        auto handle = std::make_unique<buffer_f32_impl>();
        handle->runtime = runtime_impl_ptr;
        handle->data = len == 0 ? nullptr : copy_to_device(runtime_impl_ptr, values, len);
        handle->len = len;
        *out_buffer = reinterpret_cast<struct anna_sycl_buffer_f32*>(handle.release());
    });
}

extern "C" int anna_sycl_buffer_alloc_f32(
    struct anna_sycl_runtime* runtime,
    std::size_t len,
    struct anna_sycl_buffer_f32** out_buffer) {
    return guard([&] {
        if (runtime == nullptr || out_buffer == nullptr) {
            throw std::invalid_argument("buffer alloc received null runtime or output");
        }

        auto* runtime_impl_ptr = reinterpret_cast<runtime_impl*>(runtime);
        auto handle = std::make_unique<buffer_f32_impl>();
        handle->runtime = runtime_impl_ptr;
        handle->data = len == 0 ? nullptr : malloc_device_or_throw<float>(runtime_impl_ptr, len);
        handle->len = len;
        *out_buffer = reinterpret_cast<struct anna_sycl_buffer_f32*>(handle.release());
    });
}

extern "C" void anna_sycl_buffer_destroy(struct anna_sycl_buffer_f32* buffer) {
    auto* handle = reinterpret_cast<buffer_f32_impl*>(buffer);
    if (handle == nullptr) return;
    if (handle->data != nullptr) {
        sycl::free(handle->data, handle->runtime->queue);
    }
    delete handle;
}

extern "C" int anna_sycl_buffer_read_f32(
    struct anna_sycl_runtime* runtime,
    const struct anna_sycl_buffer_f32* buffer,
    float* out_values,
    std::size_t len) {
    return guard([&] {
        auto* runtime_impl_ptr = reinterpret_cast<runtime_impl*>(runtime);
        auto* buffer_impl_ptr = reinterpret_cast<const buffer_f32_impl*>(buffer);
        if (runtime_impl_ptr == nullptr || buffer_impl_ptr == nullptr || out_values == nullptr) {
            throw std::invalid_argument("buffer read received null pointer");
        }
        validate_buffer(buffer_impl_ptr, len, "read");
        runtime_impl_ptr->queue.memcpy(out_values, buffer_impl_ptr->data, sizeof(float) * len).wait_and_throw();
    });
}

extern "C" int anna_sycl_buffer_write_f32(
    struct anna_sycl_runtime* runtime,
    const struct anna_sycl_buffer_f32* buffer,
    const float* values,
    std::size_t len) {
    return guard([&] {
        auto* runtime_impl_ptr = reinterpret_cast<runtime_impl*>(runtime);
        auto* buffer_impl_ptr = reinterpret_cast<const buffer_f32_impl*>(buffer);
        if (runtime_impl_ptr == nullptr || buffer_impl_ptr == nullptr || values == nullptr) {
            throw std::invalid_argument("buffer write received null pointer");
        }
        validate_buffer(buffer_impl_ptr, len, "write");
        runtime_impl_ptr->queue.memcpy(buffer_impl_ptr->data, values, sizeof(float) * len).wait_and_throw();
    });
}

extern "C" int anna_sycl_dense_run(
    struct anna_sycl_runtime* runtime,
    const struct anna_sycl_dense_weights* weights,
    const struct anna_sycl_buffer_f32* input,
    const struct anna_sycl_buffer_f32* output) {
    return guard([&] {
        auto* runtime_impl_ptr = reinterpret_cast<runtime_impl*>(runtime);
        auto* weights_impl_ptr = reinterpret_cast<const dense_weights_impl*>(weights);
        auto* input_impl_ptr = reinterpret_cast<const buffer_f32_impl*>(input);
        auto* output_impl_ptr = reinterpret_cast<const buffer_f32_impl*>(output);
        if (runtime_impl_ptr == nullptr || weights_impl_ptr == nullptr || input_impl_ptr == nullptr || output_impl_ptr == nullptr) {
            throw std::invalid_argument("dense run received null pointer");
        }
        validate_buffer(input_impl_ptr, weights_impl_ptr->in_features, "dense input");
        validate_buffer(output_impl_ptr, weights_impl_ptr->out_features, "dense output");

        const float* weight = weights_impl_ptr->weight;
        const float* bias = weights_impl_ptr->bias;
        const bool has_bias = weights_impl_ptr->has_bias;
        const float* input_data = input_impl_ptr->data;
        float* output_data = output_impl_ptr->data;
        const std::size_t in_features = weights_impl_ptr->in_features;
        const std::size_t out_features = weights_impl_ptr->out_features;

        runtime_impl_ptr->queue.parallel_for(sycl::range<1>(out_features), [=](sycl::id<1> index) {
            const std::size_t row = index[0];
            float sum = has_bias ? bias[row] : 0.0f;
            const std::size_t row_offset = row * in_features;
            for (std::size_t col = 0; col < in_features; ++col) {
                sum += weight[row_offset + col] * input_data[col];
            }
            output_data[row] = sum;
        }).wait_and_throw();
    });
}

extern "C" int anna_sycl_autoround_run(
    struct anna_sycl_runtime* runtime,
    const struct anna_sycl_autoround_weights* weights,
    const struct anna_sycl_buffer_f32* input,
    const struct anna_sycl_buffer_f32* output) {
    return guard([&] {
        auto* runtime_impl_ptr = reinterpret_cast<runtime_impl*>(runtime);
        auto* weights_impl_ptr = reinterpret_cast<const autoround_weights_impl*>(weights);
        auto* input_impl_ptr = reinterpret_cast<const buffer_f32_impl*>(input);
        auto* output_impl_ptr = reinterpret_cast<const buffer_f32_impl*>(output);
        if (runtime_impl_ptr == nullptr || weights_impl_ptr == nullptr || input_impl_ptr == nullptr || output_impl_ptr == nullptr) {
            throw std::invalid_argument("autoround run received null pointer");
        }
        validate_buffer(input_impl_ptr, weights_impl_ptr->in_features, "autoround input");
        validate_buffer(output_impl_ptr, weights_impl_ptr->out_features, "autoround output");

        const int* qweight = weights_impl_ptr->qweight;
        const int* qzeros = weights_impl_ptr->qzeros;
        const float* scales = weights_impl_ptr->scales;
        const float* bias = weights_impl_ptr->bias;
        const bool has_bias = weights_impl_ptr->has_bias;
        const float* input_data = input_impl_ptr->data;
        float* output_data = output_impl_ptr->data;
        const std::size_t in_features = weights_impl_ptr->in_features;
        const std::size_t out_features = weights_impl_ptr->out_features;
        const std::size_t group_size = weights_impl_ptr->group_size;
        const std::size_t packed_in = (in_features + 7) / 8;
        const std::size_t packed_out = (out_features + 7) / 8;
        const std::size_t group_count = (in_features + group_size - 1) / group_size;

        runtime_impl_ptr->queue.parallel_for(sycl::range<1>(out_features), [=](sycl::id<1> index) {
            const std::size_t row = index[0];
            float sum = has_bias ? bias[row] : 0.0f;
            const std::size_t zero_pack = row / 8;
            const std::size_t zero_shift = (row % 8) * 4;
            for (std::size_t pack = 0; pack < packed_in; ++pack) {
                const std::uint32_t qword = static_cast<std::uint32_t>(qweight[pack * out_features + row]);
                const std::size_t base_col = pack * 8;
                const std::size_t remaining = in_features - base_col;
                const std::size_t lanes = remaining < 8 ? remaining : 8;
                for (std::size_t lane = 0; lane < lanes; ++lane) {
                    const std::size_t col = base_col + lane;
                    const std::size_t group_raw = col / group_size;
                    const std::size_t group = group_raw < group_count ? group_raw : group_count - 1;
                    const std::uint32_t zword = static_cast<std::uint32_t>(qzeros[group * packed_out + zero_pack]);
                    const int zero = static_cast<int>((zword >> zero_shift) & 15u) + 1;
                    const int qvalue = static_cast<int>((qword >> (lane * 4)) & 15u);
                    const float scale = scales[group * out_features + row];
                    sum += input_data[col] * static_cast<float>(qvalue - zero) * scale;
                }
            }
            output_data[row] = sum;
        }).wait_and_throw();
    });
}

extern "C" int anna_sycl_silu_mul_inplace(
    struct anna_sycl_runtime* runtime,
    const struct anna_sycl_buffer_f32* gate,
    const struct anna_sycl_buffer_f32* up) {
    return guard([&] {
        auto* runtime_impl_ptr = reinterpret_cast<runtime_impl*>(runtime);
        auto* gate_impl_ptr = reinterpret_cast<const buffer_f32_impl*>(gate);
        auto* up_impl_ptr = reinterpret_cast<const buffer_f32_impl*>(up);
        if (runtime_impl_ptr == nullptr || gate_impl_ptr == nullptr || up_impl_ptr == nullptr) {
            throw std::invalid_argument("silu mul received null pointer");
        }
        validate_buffer(gate_impl_ptr, up_impl_ptr->len, "silu mul");

        float* gate_data = gate_impl_ptr->data;
        const float* up_data = up_impl_ptr->data;
        const std::size_t len = gate_impl_ptr->len;
        runtime_impl_ptr->queue.parallel_for(sycl::range<1>(len), [=](sycl::id<1> index) {
            const std::size_t idx = index[0];
            const float gate_value = gate_data[idx];
            gate_data[idx] = (gate_value / (1.0f + sycl::exp(-gate_value))) * up_data[idx];
        }).wait_and_throw();
    });
}

extern "C" int anna_sycl_qwen_rmsnorm_rope_inplace(
    struct anna_sycl_runtime* runtime,
    const struct anna_sycl_buffer_f32* values,
    const struct anna_sycl_buffer_f32* weight,
    std::size_t head_count,
    std::size_t head_dim,
    std::size_t position,
    float theta,
    std::size_t rotary_dim,
    float eps) {
    return guard([&] {
        auto* runtime_impl_ptr = reinterpret_cast<runtime_impl*>(runtime);
        auto* values_impl_ptr = reinterpret_cast<const buffer_f32_impl*>(values);
        auto* weight_impl_ptr = reinterpret_cast<const buffer_f32_impl*>(weight);
        if (runtime_impl_ptr == nullptr || values_impl_ptr == nullptr || weight_impl_ptr == nullptr) {
            throw std::invalid_argument("qwen rmsnorm rope received null pointer");
        }
        if (head_count * head_dim > values_impl_ptr->len) {
            throw std::invalid_argument("qwen rmsnorm rope values buffer too small");
        }
        validate_buffer(weight_impl_ptr, head_dim, "qwen rmsnorm rope weight");
        if (rotary_dim > head_dim || (rotary_dim % 2) != 0) {
            throw std::invalid_argument("invalid rotary dimension");
        }

        float* values_data = values_impl_ptr->data;
        const float* weight_data = weight_impl_ptr->data;
        const float position_f = static_cast<float>(position);
        runtime_impl_ptr->queue.parallel_for(sycl::range<1>(head_count), [=](sycl::id<1> index) {
            const std::size_t head = index[0];
            float* head_values = values_data + head * head_dim;
            float mean_square = 0.0f;
            for (std::size_t dim = 0; dim < head_dim; ++dim) {
                const float value = head_values[dim];
                mean_square += value * value;
            }
            mean_square /= static_cast<float>(head_dim);
            const float inv = 1.0f / sycl::sqrt(mean_square + eps);
            for (std::size_t dim = 0; dim < head_dim; ++dim) {
                head_values[dim] = head_values[dim] * inv * (1.0f + weight_data[dim]);
            }
            const std::size_t half_dim = rotary_dim / 2;
            for (std::size_t dim = 0; dim < half_dim; ++dim) {
                const float exponent = static_cast<float>(dim * 2) / static_cast<float>(rotary_dim);
                const float inv_freq = 1.0f / sycl::pow(theta, exponent);
                const float angle = position_f * inv_freq;
                const float c = sycl::cos(angle);
                const float s = sycl::sin(angle);
                const float x1 = head_values[dim];
                const float x2 = head_values[dim + half_dim];
                head_values[dim] = x1 * c - x2 * s;
                head_values[dim + half_dim] = x2 * c + x1 * s;
            }
        }).wait_and_throw();
    });
}
