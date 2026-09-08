#ifndef QNN_FUNCTIONAL_CONVOLUTION_HPP
#define QNN_FUNCTIONAL_CONVOLUTION_HPP

#include <cassert>
#include <cstddef>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/tensor.hpp"

namespace qnn {
namespace functional {

// Zero-pad the spatial dims of an image-shaped tensor.
// input: [1, H, W] or [1, C, H, W]  (leading dims are left untouched)
template <typename T>
tensor<quaternion<T>> add_padding(const tensor<quaternion<T>>& input, int padding) {
    const bool has_c = input.rank() == 4;
    const std::size_t h_off = has_c ? 2 : 1;
    const std::size_t H = input.dim(h_off);
    const std::size_t W = input.dim(h_off + 1);

    tensor<quaternion<T>> padded;
    if (has_c) {
        padded = tensor<quaternion<T>>(shape{input.dim(0), input.dim(1),
                                             H + 2 * static_cast<std::size_t>(padding),
                                             W + 2 * static_cast<std::size_t>(padding)});
    } else {
        padded = tensor<quaternion<T>>(shape{input.dim(0),
                                             H + 2 * static_cast<std::size_t>(padding),
                                             W + 2 * static_cast<std::size_t>(padding)});
    }

    const std::size_t C = has_c ? input.dim(1) : 1;
    for (std::size_t c = 0; c < C; ++c) {
        for (std::size_t i = 0; i < input.dim(h_off); ++i) {
            for (std::size_t j = 0; j < input.dim(h_off + 1); ++j) {
                if (has_c) {
                    padded(0, c, i + padding, j + padding) = input(0, c, i, j);
                } else {
                    padded(0, i + padding, j + padding) = input(0, i, j);
                }
            }
        }
    }
    return padded;
}

// kernel: [out_channels, in_channels, kernel_height, kernel_width]
// input:  [1, in_channels, input_height, input_width]  ([1, H, W] == 1 channel)
// output: [out_channels, output_height, output_width]
template <typename T>
tensor<quaternion<T>> conv2d(const tensor<quaternion<T>>& input,
                             const tensor<quaternion<T>>& kernel, int padding = 0,
                             int stride = 1) {
    assert(input.rank() == 3 || input.rank() == 4);
    assert(kernel.rank() == 4);

    const bool has_c = input.rank() == 4;
    const std::size_t h_off = has_c ? 2 : 1;
    const std::size_t C_in = has_c ? input.dim(1) : 1;
    const std::size_t H = input.dim(h_off);
    const std::size_t W = input.dim(h_off + 1);
    assert(kernel.dim(1) == C_in);

    const int KH = static_cast<int>(kernel.dim(2));
    const int KW = static_cast<int>(kernel.dim(3));

    const int output_h = (static_cast<int>(H) + 2 * padding - KH) / stride + 1;
    const int output_w = (static_cast<int>(W) + 2 * padding - KW) / stride + 1;
    const std::size_t oh = output_h > 0 ? static_cast<std::size_t>(output_h) : 0;
    const std::size_t ow = output_w > 0 ? static_cast<std::size_t>(output_w) : 0;

    tensor<quaternion<T>> output(shape{kernel.dim(0), oh, ow});

    const tensor<quaternion<T>> padded =
        padding > 0 ? add_padding(input, padding) : input;

    auto px = [&](std::size_t c, std::size_t i, std::size_t j) -> const quaternion<T>& {
        return has_c ? padded(0, c, i, j) : padded(0, i, j);
    };

    for (std::size_t co = 0; co < kernel.dim(0); ++co) {
        for (std::size_t i = 0; i < oh; ++i) {
            for (std::size_t j = 0; j < ow; ++j) {
                quaternion<T> sum{};
                for (std::size_t c = 0; c < C_in; ++c) {
                    for (std::size_t m = 0; m < kernel.dim(2); ++m) {
                        for (std::size_t n = 0; n < kernel.dim(3); ++n) {
                            const std::size_t pi =
                                static_cast<std::size_t>(static_cast<int>(i) * stride + static_cast<int>(m));
                            const std::size_t pj =
                                static_cast<std::size_t>(static_cast<int>(j) * stride + static_cast<int>(n));
                            sum = sum + kernel(co, c, m, n) * px(c, pi, pj);
                        }
                    }
                }
                output(co, i, j) = sum;
            }
        }
    }
    return output;
}

}  // namespace functional
}  // namespace qnn

#endif  // QNN_FUNCTIONAL_CONVOLUTION_HPP