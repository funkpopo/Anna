#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

namespace {

at::Tensor apply_activation(at::Tensor output, int64_t activation) {
    switch (activation) {
        case 0:
            return output;
        case 1:
            return at::silu(output);
        case 2:
            return at::relu(output);
        case 3:
            return at::gelu(output, "none");
        default:
            TORCH_CHECK(false, "Unsupported anna_xpu::linear_pointwise activation code: ", activation);
    }
}

at::Tensor linear_pointwise_impl(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& residual,
    int64_t activation
) {
    at::Tensor output = at::linear(input, weight, bias);
    output = apply_activation(std::move(output), activation);
    if (residual.has_value()) {
        output = at::add(output, residual.value());
    }
    return output;
}

at::Tensor linear_pointwise_cpu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& residual,
    int64_t activation
) {
    return linear_pointwise_impl(input, weight, bias, residual, activation);
}

}  // namespace

TORCH_LIBRARY(anna_xpu, m) {
    m.def("linear_pointwise(Tensor input, Tensor weight, Tensor? bias=None, Tensor? residual=None, int activation=0) -> Tensor");
}

TORCH_LIBRARY_IMPL(anna_xpu, CPU, m) {
    m.impl("linear_pointwise", linear_pointwise_cpu);
}
