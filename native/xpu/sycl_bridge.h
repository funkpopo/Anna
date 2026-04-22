#pragma once

#include <stddef.h>

#ifdef _WIN32
#define ANNA_SYCL_API __declspec(dllexport)
#else
#define ANNA_SYCL_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum anna_sycl_status {
    ANNA_SYCL_SUCCESS = 0,
    ANNA_SYCL_BACKEND_UNAVAILABLE = 1,
    ANNA_SYCL_DEVICE_NOT_FOUND = 2,
    ANNA_SYCL_INVALID_ARGUMENT = 3,
    ANNA_SYCL_ALLOCATION_FAILED = 4,
    ANNA_SYCL_RUNTIME_ERROR = 5,
};

struct anna_sycl_runtime;
struct anna_sycl_buffer_f32;
struct anna_sycl_dense_weights;
struct anna_sycl_autoround_weights;

ANNA_SYCL_API const char* anna_sycl_last_error_message(void);

ANNA_SYCL_API int anna_sycl_runtime_create(struct anna_sycl_runtime** out_runtime);
ANNA_SYCL_API void anna_sycl_runtime_destroy(struct anna_sycl_runtime* runtime);
ANNA_SYCL_API const char* anna_sycl_runtime_platform_name(const struct anna_sycl_runtime* runtime);
ANNA_SYCL_API const char* anna_sycl_runtime_device_name(const struct anna_sycl_runtime* runtime);
ANNA_SYCL_API const char* anna_sycl_runtime_vendor_name(const struct anna_sycl_runtime* runtime);

ANNA_SYCL_API int anna_sycl_dense_create(
    struct anna_sycl_runtime* runtime,
    const float* weight,
    size_t weight_len,
    const float* bias,
    size_t bias_len,
    size_t out_features,
    size_t in_features,
    struct anna_sycl_dense_weights** out_weights);
ANNA_SYCL_API void anna_sycl_dense_destroy(struct anna_sycl_dense_weights* weights);

ANNA_SYCL_API int anna_sycl_autoround_create(
    struct anna_sycl_runtime* runtime,
    const int* qweight,
    size_t qweight_len,
    const int* qzeros,
    size_t qzeros_len,
    const float* scales,
    size_t scales_len,
    const float* bias,
    size_t bias_len,
    size_t out_features,
    size_t in_features,
    size_t group_size,
    struct anna_sycl_autoround_weights** out_weights);
ANNA_SYCL_API void anna_sycl_autoround_destroy(struct anna_sycl_autoround_weights* weights);

ANNA_SYCL_API int anna_sycl_buffer_upload_f32(
    struct anna_sycl_runtime* runtime,
    const float* values,
    size_t len,
    struct anna_sycl_buffer_f32** out_buffer);
ANNA_SYCL_API int anna_sycl_buffer_alloc_f32(
    struct anna_sycl_runtime* runtime,
    size_t len,
    struct anna_sycl_buffer_f32** out_buffer);
ANNA_SYCL_API void anna_sycl_buffer_destroy(struct anna_sycl_buffer_f32* buffer);
ANNA_SYCL_API int anna_sycl_buffer_read_f32(
    struct anna_sycl_runtime* runtime,
    const struct anna_sycl_buffer_f32* buffer,
    float* out_values,
    size_t len);
ANNA_SYCL_API int anna_sycl_buffer_write_f32(
    struct anna_sycl_runtime* runtime,
    const struct anna_sycl_buffer_f32* buffer,
    const float* values,
    size_t len);

ANNA_SYCL_API int anna_sycl_dense_run(
    struct anna_sycl_runtime* runtime,
    const struct anna_sycl_dense_weights* weights,
    const struct anna_sycl_buffer_f32* input,
    const struct anna_sycl_buffer_f32* output);
ANNA_SYCL_API int anna_sycl_autoround_run(
    struct anna_sycl_runtime* runtime,
    const struct anna_sycl_autoround_weights* weights,
    const struct anna_sycl_buffer_f32* input,
    const struct anna_sycl_buffer_f32* output);
ANNA_SYCL_API int anna_sycl_silu_mul_inplace(
    struct anna_sycl_runtime* runtime,
    const struct anna_sycl_buffer_f32* gate,
    const struct anna_sycl_buffer_f32* up);
ANNA_SYCL_API int anna_sycl_qwen_rmsnorm_rope_inplace(
    struct anna_sycl_runtime* runtime,
    const struct anna_sycl_buffer_f32* values,
    const struct anna_sycl_buffer_f32* weight,
    size_t head_count,
    size_t head_dim,
    size_t position,
    float theta,
    size_t rotary_dim,
    float eps);

#ifdef __cplusplus
}
#endif
