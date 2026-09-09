#ifndef QNN_FUNCTIONAL_POOL_HPP
#define QNN_FUNCTIONAL_POOL_HPP

#include <cassert>
#include <cstddef>
#include <limits>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/tensor.hpp"

namespace qnn {
namespace functional {

enum class pool_type { max, average };

// input:  [channels, height, width]  (conv output feeds this directly)
// output: [channels, output_height, output_width]
template <typename T>
tensor<quaternion<T>> pool2d(const tensor<quaternion<T>>& input, pool_type type,
                             int pool_h = 2, int pool_w = 2, int stride = 1) {
    assert(input.rank() == 3);

    const std::size_t C = input.dim(0);
    const int H = static_cast<int>(input.dim(1));
    const int W = static_cast<int>(input.dim(2));

    const int output_h = (H - pool_h) / stride + 1;
    const int output_w = (W - pool_w) / stride + 1;
    const std::size_t oh = output_h > 0 ? static_cast<std::size_t>(output_h) : 0;
    const std::size_t ow = output_w > 0 ? static_cast<std::size_t>(output_w) : 0;

    tensor<quaternion<T>> output(shape{C, oh, ow});

    if (type == pool_type::max) {
        for (std::size_t c = 0; c < C; ++c) {
            for (std::size_t i = 0; i < oh; ++i) {
                for (std::size_t j = 0; j < ow; ++j) {
                    quaternion<T> acc{};
                    bool first = true;
                    for (int m = 0; m < pool_h; ++m) {
                        for (int n = 0; n < pool_w; ++n) {
                            const int h_idx = static_cast<int>(i) * stride + m;
                            const int w_idx = static_cast<int>(j) * stride + n;
                            if (h_idx < H && w_idx < W) {
                                const quaternion<T>& q =
                                    input(c, static_cast<std::size_t>(h_idx),
                                          static_cast<std::size_t>(w_idx));
                                if (first) {
                                    acc = q;
                                    first = false;
                                } else {
                                    acc = quaternion<T>(acc.w > q.w ? acc.w : q.w,
                                                        acc.x > q.x ? acc.x : q.x,
                                                        acc.y > q.y ? acc.y : q.y,
                                                        acc.z > q.z ? acc.z : q.z);
                                }
                            }
                        }
                    }
                    output(c, i, j) = acc;
                }
            }
        }
    } else {
        for (std::size_t c = 0; c < C; ++c) {
            for (std::size_t i = 0; i < oh; ++i) {
                for (std::size_t j = 0; j < ow; ++j) {
                    quaternion<T> sum{};
                    int count = 0;
                    for (int m = 0; m < pool_h; ++m) {
                        for (int n = 0; n < pool_w; ++n) {
                            const int h_idx = static_cast<int>(i) * stride + m;
                            const int w_idx = static_cast<int>(j) * stride + n;
                            if (h_idx < H && w_idx < W) {
                                sum = sum + input(c, static_cast<std::size_t>(h_idx),
                                                  static_cast<std::size_t>(w_idx));
                                ++count;
                            }
                        }
                    }
                    output(c, i, j) = sum * (T(1) / static_cast<T>(count));
                }
            }
        }
    }
    return output;
}

// grad_output: [channels, output_height, output_width]
// input: [channels, height, width]  (same shape as the pool2d input)
// returns grad_input [channels, height, width]
// split-max: each component's dy flows to the pixel that maximized that component
template <typename T>
tensor<quaternion<T>> backward_pool2d(const tensor<quaternion<T>>& input,
                                      const tensor<quaternion<T>>& grad_output,
                                      pool_type type, int pool_h = 2,
                                      int pool_w = 2, int stride = 1) {
    assert(input.rank() == 3);
    assert(grad_output.rank() == 3);

    const std::size_t C = input.dim(0);
    const int H = static_cast<int>(input.dim(1));
    const int W = static_cast<int>(input.dim(2));

    const int output_h = (H - pool_h) / stride + 1;
    const int output_w = (W - pool_w) / stride + 1;
    const std::size_t oh = output_h > 0 ? static_cast<std::size_t>(output_h) : 0;
    const std::size_t ow = output_w > 0 ? static_cast<std::size_t>(output_w) : 0;
    assert(grad_output.dim(0) == C);
    assert(grad_output.dim(1) == oh && grad_output.dim(2) == ow);

    tensor<quaternion<T>> grad_input(shape{C, static_cast<std::size_t>(H),
                                            static_cast<std::size_t>(W)});

    if (type == pool_type::max) {
        for (std::size_t c = 0; c < C; ++c) {
            for (std::size_t i = 0; i < oh; ++i) {
                for (std::size_t j = 0; j < ow; ++j) {
                    const quaternion<T>& dy = grad_output(c, i, j);
                    for (int comp = 0; comp < 4; ++comp) {
                        int bm = -1, bn = -1;
                        T best = std::numeric_limits<T>::lowest();
                        for (int m = 0; m < pool_h; ++m) {
                            for (int n = 0; n < pool_w; ++n) {
                                const int h_idx =
                                    static_cast<int>(i) * stride + m;
                                const int w_idx =
                                    static_cast<int>(j) * stride + n;
                                if (h_idx < H && w_idx < W) {
                                    const quaternion<T>& q =
                                        input(c, static_cast<std::size_t>(h_idx),
                                              static_cast<std::size_t>(w_idx));
                                    const T v =
                                        comp == 0 ? q.w
                                                  : comp == 1 ? q.x
                                                              : comp == 2 ? q.y
                                                                           : q.z;
                                    if (v > best) {  // first-seen wins ties (matches forward)
                                        best = v;
                                        bm = m;
                                        bn = n;
                                    }
                                }
                            }
                        }
                        if (bm >= 0 && bn >= 0) {
                            const std::size_t hi = static_cast<std::size_t>(
                                static_cast<int>(i) * stride + bm);
                            const std::size_t wi = static_cast<std::size_t>(
                                static_cast<int>(j) * stride + bn);
                            quaternion<T>& g = grad_input(c, hi, wi);
                            if (comp == 0) g.w = g.w + dy.w;
                            else if (comp == 1) g.x = g.x + dy.x;
                            else if (comp == 2) g.y = g.y + dy.y;
                            else g.z = g.z + dy.z;
                        }
                    }
                }
            }
        }
    } else {
        for (std::size_t c = 0; c < C; ++c) {
            for (std::size_t i = 0; i < oh; ++i) {
                for (std::size_t j = 0; j < ow; ++j) {
                    const quaternion<T>& dy = grad_output(c, i, j);
                    int count = 0;
                    for (int m = 0; m < pool_h; ++m) {
                        for (int n = 0; n < pool_w; ++n) {
                            const int h_idx =
                                static_cast<int>(i) * stride + m;
                            const int w_idx =
                                static_cast<int>(j) * stride + n;
                            if (h_idx < H && w_idx < W) ++count;
                        }
                    }
                    if (count == 0) continue;
                    const quaternion<T> g = dy * (T(1) / static_cast<T>(count));
                    for (int m = 0; m < pool_h; ++m) {
                        for (int n = 0; n < pool_w; ++n) {
                            const int h_idx =
                                static_cast<int>(i) * stride + m;
                            const int w_idx =
                                static_cast<int>(j) * stride + n;
                            if (h_idx < H && w_idx < W) {
                                grad_input(c, static_cast<std::size_t>(h_idx),
                                           static_cast<std::size_t>(w_idx)) =
                                    grad_input(c, static_cast<std::size_t>(h_idx),
                                               static_cast<std::size_t>(w_idx)) +
                                    g;
                            }
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}

}  // namespace functional
}  // namespace qnn

#endif  // QNN_FUNCTIONAL_POOL_HPP