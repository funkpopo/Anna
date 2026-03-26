#include <torch/extension.h>

#include <string>

torch::Tensor anna_onednn_linear_pointwise(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    const std::string& activation,
    const std::string& algorithm,
    c10::optional<torch::Tensor> other,
    const std::string& binary);

torch::Tensor anna_onednn_linear_int4_weight_only(
    torch::Tensor x,
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor zero,
    int64_t group_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "linear_pointwise",
        &anna_onednn_linear_pointwise,
        "Anna oneDNN XPU linear pointwise");
    m.def(
        "linear_int4_weight_only",
        &anna_onednn_linear_int4_weight_only,
        "Anna oneDNN XPU int4 weight-only linear");
}
