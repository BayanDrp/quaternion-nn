#ifndef QNN_FUNCTIONAL_CONVOLUTION_HPP
#define QNN_FUNCTIONAL_CONVOLUTION_HPP

#include <cstddef>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/tensor.hpp"

namespace qnn {
namespace functional {

// Zero-pad a [1, H, W] input by `padding` on every side.
template <typename T>
tensor<quaternion<T>> add_padding(const tensor<quaternion<T>>& input, int padding) {
    const int H = static_cast<int>(input.dim(1));
    const int W = static_cast<int>(input.dim(2));
    tensor<quaternion<T>> padded(
        shape{input.dim(0), static_cast<std::size_t>(H + 2 * padding),
              static_cast<std::size_t>(W + 2 * padding)});
    for (std::size_t i = 0; i < input.dim(0); ++i) {
        for (std::size_t j = 0; j < input.dim(1); ++j) {
            for (std::size_t k = 0; k < input.dim(2); ++k) {
                padded(i, static_cast<std::size_t>(j + padding),
                       static_cast<std::size_t>(k + padding)) = input(i, j, k);
            }
        }
    }
    return padded;
}

// kernel: [out_channels, 1, kernel_height, kernel_width]
// input:  [1, input_height, input_width]
// output: [kernel_number, output_height, output_width]
template <typename T>
tensor<quaternion<T>> conv2d(const tensor<quaternion<T>>& input,
                             const tensor<quaternion<T>>& kernel, int padding = 0,
                             int stride = 1) {
    const int H = static_cast<int>(input.dim(1));
    const int KH = static_cast<int>(kernel.dim(2));

    const int output_size = (H + 2 * padding - KH) / stride + 1;
    const int valid_output = output_size > 0 ? output_size : 0;

    tensor<quaternion<T>> output(
        shape{kernel.dim(0), static_cast<std::size_t>(valid_output),
              static_cast<std::size_t>(valid_output)});

    const tensor<quaternion<T>> padded =
        padding > 0 ? add_padding(input, padding) : input;

    for (std::size_t k = 0; k < kernel.dim(0); ++k) {
        for (std::size_t i = 0; i < output.dim(1); ++i) {
            for (std::size_t j = 0; j < output.dim(2); ++j) {
                quaternion<T> sum{};
                for (std::size_t m = 0; m < kernel.dim(2); ++m) {
                    for (std::size_t n = 0; n < kernel.dim(3); ++n) {
                        const int pi = static_cast<int>(i) * stride + static_cast<int>(m);
                        const int pj = static_cast<int>(j) * stride + static_cast<int>(n);
                        sum = sum + kernel(k, 0, m, n) * padded(0, static_cast<std::size_t>(pi),
                                                                static_cast<std::size_t>(pj));
                    }
                }
                output(k, i, j) = sum;
            }
        }
    }
    return output;
}

}  // namespace functional
}  // namespace qnn

#endif  // QNN_FUNCTIONAL_CONVOLUTION_HPP