#ifndef QNN_NN_LAYERS_POOL2D_HPP
#define QNN_NN_LAYERS_POOL2D_HPP

#include <cassert>
#include <cstddef>
#include <vector>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/functional/pool.hpp"
#include "qnn/nn/module.hpp"

namespace qnn {
namespace nn {
namespace layers {

// Batched pooling layer. Wraps functional::pool2d / backward_pool2d.
// input:  [N, C, H, W]
// output: [N, C, oh, ow]
template <typename T = float>
class pool2d : public qnn::nn::Module<T> {
public:
    pool2d(std::size_t channels, qnn::shape pool_size,
           int stride = 0,
           qnn::functional::pool_type type = qnn::functional::pool_type::max)
        : c_(channels), ph_(pool_size.dim(0)), pw_(pool_size.dim(1)),
          stride_(stride > 0 ? stride : static_cast<int>(pool_size.dim(0))),
          type_(type) {
        assert(pool_size.rank() == 2);
        assert(c_ > 0 && ph_ > 0 && pw_ > 0 && stride_ > 0);
    }

    std::size_t channels() const { return c_; }
    std::size_t pool_h() const { return ph_; }
    std::size_t pool_w() const { return pw_; }
    int stride() const { return stride_; }
    qnn::functional::pool_type type() const { return type_; }

    std::vector<tensor<quaternion<T>>*> parameters() override { return {}; }
    std::vector<tensor<quaternion<T>>*> gradients() override { return {}; }

    tensor<quaternion<T>> forward(const tensor<quaternion<T>>& x) override {
        assert(x.rank() == 4);
        assert(x.dim(1) == c_);
        const std::size_t batch = x.dim(0);
        const int H = static_cast<int>(x.dim(2));
        const int W = static_cast<int>(x.dim(3));
        const int oh_i = (H - static_cast<int>(ph_)) / stride_ + 1;
        const int ow_i = (W - static_cast<int>(pw_)) / stride_ + 1;
        assert(oh_i > 0 && ow_i > 0);
        const std::size_t oh = static_cast<std::size_t>(oh_i);
        const std::size_t ow = static_cast<std::size_t>(ow_i);

        tensor<quaternion<T>> out(::qnn::shape{batch, c_, oh, ow});
#pragma omp parallel for schedule(static)
        for (std::size_t b = 0; b < batch; ++b) {
            const tensor<quaternion<T>> img = slice_image(x, b);
            const tensor<quaternion<T>> p =
                qnn::functional::pool2d(img, type_, static_cast<int>(ph_),
                                        static_cast<int>(pw_), stride_);
            for (std::size_t c = 0; c < c_; ++c) {
                for (std::size_t i = 0; i < oh; ++i) {
                    for (std::size_t j = 0; j < ow; ++j) {
                        out(b, c, i, j) = p(c, i, j);
                    }
                }
            }
        }
        x_ = x;
        return out;
    }

    tensor<quaternion<T>> backward(const tensor<quaternion<T>>& dy) override {
        assert(x_.rank() == 4);
        assert(dy.rank() == 4);
        assert(dy.dim(0) == x_.dim(0));
        assert(dy.dim(1) == c_);
        const std::size_t batch = x_.dim(0);
        const std::size_t H = x_.dim(2), W = x_.dim(3);

        tensor<quaternion<T>> dx(x_.shape());
#pragma omp parallel for schedule(static)
        for (std::size_t b = 0; b < batch; ++b) {
            const tensor<quaternion<T>> img = slice_image(x_, b);
            const tensor<quaternion<T>> g = slice_grad(dy, b);
            const tensor<quaternion<T>> dxi = qnn::functional::backward_pool2d(
                img, g, type_, static_cast<int>(ph_), static_cast<int>(pw_), stride_);
            for (std::size_t c = 0; c < c_; ++c) {
                for (std::size_t i = 0; i < H; ++i) {
                    for (std::size_t j = 0; j < W; ++j) {
                        dx(b, c, i, j) = dxi(c, i, j);
                    }
                }
            }
        }
        return dx;
    }

private:
    static tensor<quaternion<T>> slice_image(const tensor<quaternion<T>>& x,
                                             std::size_t b) {
        const std::size_t C = x.dim(1), H = x.dim(2), W = x.dim(3);
        tensor<quaternion<T>> img(::qnn::shape{C, H, W});
        for (std::size_t c = 0; c < C; ++c) {
            for (std::size_t i = 0; i < H; ++i) {
                for (std::size_t j = 0; j < W; ++j) {
                    img(c, i, j) = x(b, c, i, j);
                }
            }
        }
        return img;
    }

    static tensor<quaternion<T>> slice_grad(const tensor<quaternion<T>>& dy,
                                            std::size_t b) {
        const std::size_t C = dy.dim(1), oh = dy.dim(2), ow = dy.dim(3);
        tensor<quaternion<T>> g(::qnn::shape{C, oh, ow});
        for (std::size_t c = 0; c < C; ++c) {
            for (std::size_t i = 0; i < oh; ++i) {
                for (std::size_t j = 0; j < ow; ++j) {
                    g(c, i, j) = dy(b, c, i, j);
                }
            }
        }
        return g;
    }

    std::size_t c_;
    std::size_t ph_;
    std::size_t pw_;
    int stride_;
    qnn::functional::pool_type type_;
    tensor<quaternion<T>> x_;
};

}  // namespace layers
}  // namespace nn
}  // namespace qnn

#endif  // QNN_NN_LAYERS_POOL2D_HPP