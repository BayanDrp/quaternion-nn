#ifndef QNN_NN_LAYERS_CONV2D_HPP
#define QNN_NN_LAYERS_CONV2D_HPP

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/functional/convolution.hpp"
#include "qnn/nn/init.hpp"
#include "qnn/nn/module.hpp"
#include "qnn/nn/parameter.hpp"

namespace qnn {
namespace nn {
namespace layers {

// Batched 2D convolution layer. Wraps functional::conv2d / backward_conv2d.
//
// Shape convention (shared by pool2d, flatten, qcnn): image tensors always
// carry an explicit channel dimension, [N, C, H, W]. A single-channel input is
// [N, 1, H, W] — there is no implicit rank-3 [N, H, W] form at layer level.
//
// output: [N, out_channels, oh, ow]
template <typename T = float>
class conv2d : public qnn::nn::Module<T> {
public:
    conv2d(std::size_t out_channels, qnn::shape kernel_size,
           std::size_t in_channels = 1, int padding = 0, int stride = 1,
           bool use_bias = true, std::uint32_t seed = 42)
        : in_(in_channels), out_(out_channels), kh_(kernel_size.dim(0)),
          kw_(kernel_size.dim(1)), padding_(padding), stride_(stride),
          use_bias_(use_bias),
          kernel_(::qnn::shape{out_channels, in_channels, kh_, kw_}),
          bias_(::qnn::shape{out_channels}) {
        assert(kernel_size.rank() == 2);
        assert(kernel_size.dim(0) > 0 && kernel_size.dim(1) > 0);
        std::mt19937 rng(seed);
        qnn::nn::xavier(kernel_.value(), in_channels * kh_ * kw_,
                        out_channels * kh_ * kw_, rng);
        zero_grad();
    }

    std::size_t in_channels() const { return in_; }
    std::size_t out_channels() const { return out_; }
    std::size_t kernel_h() const { return kh_; }
    std::size_t kernel_w() const { return kw_; }
    int padding() const { return padding_; }
    int stride() const { return stride_; }
    bool has_bias() const { return use_bias_; }

    tensor<quaternion<T>>& weight() { return kernel_.value(); }
    const tensor<quaternion<T>>& weight() const { return kernel_.value(); }
    tensor<quaternion<T>>& bias() { return bias_.value(); }
    const tensor<quaternion<T>>& bias() const { return bias_.value(); }
    tensor<quaternion<T>>& dweight() { return kernel_.grad(); }
    const tensor<quaternion<T>>& dweight() const { return kernel_.grad(); }
    tensor<quaternion<T>>& dbias() { return bias_.grad(); }
    const tensor<quaternion<T>>& dbias() const { return bias_.grad(); }

    std::vector<tensor<quaternion<T>>*> parameters() override {
        std::vector<tensor<quaternion<T>>*> ps;
        ps.push_back(&kernel_.value());
        if (use_bias_) ps.push_back(&bias_.value());
        return ps;
    }

    std::vector<tensor<quaternion<T>>*> gradients() override {
        std::vector<tensor<quaternion<T>>*> gs;
        gs.push_back(&kernel_.grad());
        if (use_bias_) gs.push_back(&bias_.grad());
        return gs;
    }

    // input: [N, C_in, H, W]
    tensor<quaternion<T>> forward(const tensor<quaternion<T>>& x) override {
        assert(x.rank() == 4);  // [N, C, H, W] — see class comment
        const std::size_t batch = x.dim(0);
        assert(x.dim(1) == in_);
        const int H = static_cast<int>(x.dim(2));
        const int W = static_cast<int>(x.dim(3));
        const int oh_i = (H + 2 * padding_ - static_cast<int>(kh_)) / stride_ + 1;
        const int ow_i = (W + 2 * padding_ - static_cast<int>(kw_)) / stride_ + 1;
        assert(oh_i > 0 && ow_i > 0);
        const std::size_t oh = static_cast<std::size_t>(oh_i);
        const std::size_t ow = static_cast<std::size_t>(ow_i);

        tensor<quaternion<T>> out(::qnn::shape{batch, out_, oh, ow});
#pragma omp parallel for schedule(static)
        for (std::size_t b = 0; b < batch; ++b) {
            const tensor<quaternion<T>> img = slice_image(x, b);
            const tensor<quaternion<T>> y =
                qnn::functional::conv2d(img, kernel_.value(), padding_, stride_);
            for (std::size_t co = 0; co < out_; ++co) {
                for (std::size_t i = 0; i < oh; ++i) {
                    for (std::size_t j = 0; j < ow; ++j) {
                        quaternion<T> v = y(co, i, j);
                        if (use_bias_) v = v + bias_.value()[co];
                        out(b, co, i, j) = v;
                    }
                }
            }
        }
        x_ = x;
        output_ = out;
        return output_;
    }

    // dy: [N, C_out, oh, ow]; returns dx [N, C_in, H, W]
    tensor<quaternion<T>> backward(const tensor<quaternion<T>>& dy) override {
        assert(dy.rank() == 4);  // [N, C_out, oh, ow]
        assert(dy.dim(0) == x_.dim(0));
        assert(dy.dim(1) == out_);
        const std::size_t batch = x_.dim(0);
        const std::size_t oh = dy.dim(2), ow = dy.dim(3);

        tensor<quaternion<T>> dx(x_.shape());
#pragma omp parallel
        {
            tensor<quaternion<T>> kacc(kernel_.grad().shape());
            tensor<quaternion<T>> bacc(bias_.grad().shape());
#pragma omp for schedule(static)
            for (std::size_t b = 0; b < batch; ++b) {
                const tensor<quaternion<T>> img = slice_image(x_, b);
                const tensor<quaternion<T>> dy_img = slice_grad(dy, b);
                const auto [dx_img, dk] = qnn::functional::backward_conv2d(
                    img, kernel_.value(), dy_img, padding_, stride_);
                for (std::size_t i = 0; i < kernel_.grad().size(); ++i) {
                    kacc[i] = kacc[i] + dk[i];
                }
                for (std::size_t co = 0; co < out_; ++co) {
                    quaternion<T> acc{};
                    for (std::size_t i = 0; i < oh; ++i) {
                        for (std::size_t j = 0; j < ow; ++j) {
                            acc = acc + dy(b, co, i, j);
                        }
                    }
                    bacc[co] = bacc[co] + acc;
                }
                paste_image(dx, b, dx_img);
            }
#pragma omp critical
            {
                for (std::size_t i = 0; i < kernel_.grad().size(); ++i) {
                    kernel_.grad()[i] = kernel_.grad()[i] + kacc[i];
                }
                if (use_bias_) {
                    for (std::size_t i = 0; i < bias_.grad().size(); ++i) {
                        bias_.grad()[i] = bias_.grad()[i] + bacc[i];
                    }
                }
            }
        }
        return dx;
    }

    void apply_gradients(T lr) {
        for (std::size_t i = 0; i < kernel_.value().size(); ++i) {
            kernel_.value()[i] = kernel_.value()[i] - kernel_.grad()[i] * lr;
        }
        if (use_bias_) {
            for (std::size_t i = 0; i < bias_.value().size(); ++i) {
                bias_.value()[i] = bias_.value()[i] - bias_.grad()[i] * lr;
            }
        }
        zero_grad();
    }

    void zero_grad() {
        kernel_.zero_grad();
        if (use_bias_) bias_.zero_grad();
    }

private:
    static tensor<quaternion<T>> slice_image(const tensor<quaternion<T>>& x,
                                             std::size_t b) {
        // functional::conv2d expects a single image with an explicit channel
        // dim: [1, C, H, W]. Rank-3 would be read as a single-channel image.
        const std::size_t C = x.dim(1), H = x.dim(2), W = x.dim(3);
        tensor<quaternion<T>> img(::qnn::shape{1, C, H, W});
        for (std::size_t c = 0; c < C; ++c) {
            for (std::size_t i = 0; i < H; ++i) {
                for (std::size_t j = 0; j < W; ++j) {
                    img(0, c, i, j) = x(b, c, i, j);
                }
            }
        }
        return img;
    }

    static tensor<quaternion<T>> slice_grad(const tensor<quaternion<T>>& dy,
                                            std::size_t b) {
        const std::size_t Cout = dy.dim(1), oh = dy.dim(2), ow = dy.dim(3);
        tensor<quaternion<T>> g(::qnn::shape{Cout, oh, ow});
        for (std::size_t co = 0; co < Cout; ++co) {
            for (std::size_t i = 0; i < oh; ++i) {
                for (std::size_t j = 0; j < ow; ++j) {
                    g(co, i, j) = dy(b, co, i, j);
                }
            }
        }
        return g;
    }

    static void paste_image(tensor<quaternion<T>>& dx, std::size_t b,
                            const tensor<quaternion<T>>& dx_img) {
        const std::size_t C = dx.dim(1), H = dx.dim(2), W = dx.dim(3);
        for (std::size_t c = 0; c < C; ++c) {
            for (std::size_t i = 0; i < H; ++i) {
                for (std::size_t j = 0; j < W; ++j) {
                    dx(b, c, i, j) = dx_img(0, c, i, j);
                }
            }
        }
    }

    std::size_t in_;
    std::size_t out_;
    std::size_t kh_;
    std::size_t kw_;
    int padding_;
    int stride_;
    bool use_bias_;
    Parameter<T> kernel_;
    Parameter<T> bias_;
    tensor<quaternion<T>> x_;
    tensor<quaternion<T>> output_;
};

}  // namespace layers
}  // namespace nn
}  // namespace qnn

#endif  // QNN_NN_LAYERS_CONV2D_HPP