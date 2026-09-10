#ifndef QNN_NN_LAYERS_ACTIVATION_HPP
#define QNN_NN_LAYERS_ACTIVATION_HPP

#include <cassert>
#include <cstddef>
#include <vector>

#include "qnn/core/quaternion.hpp"
#include "qnn/functional/relu.hpp"
#include "qnn/functional/sigmoid.hpp"
#include "qnn/functional/tanh.hpp"
#include "qnn/nn/module.hpp"

namespace qnn {
namespace nn {
namespace layers {

// The per-quaternion split gates live in qnn::functional; re-export them here
// so existing code can keep using `qnn::nn::layers::split_tanh` etc. without
// a source-level change. (See qnn/functional/{relu,sigmoid,tanh}.hpp.)
using qnn::functional::split_relu;
using qnn::functional::split_relu_prime;
using qnn::functional::split_sigmoid;
using qnn::functional::split_sigmoid_prime;
using qnn::functional::split_tanh;
using qnn::functional::split_tanh_prime;

// Split activation as a Module, so it composes inside sequential<T>.
// Applies the chosen activation elementwise to every component of every
// quaternion; backward gates dy by the activation derivative (componentwise
// product) at the cached pre-activation input.
//
// CONVENTION: the split gate is a per-component product,
//   dx = (dy.w*g'.w, dy.x*g'.x, dy.y*g'.y, dy.z*g'.z)
// exactly as examples/xor.cpp does. This is NOT a Hamilton product — the
// pre-refactor QCNN examples used "dy * split_*_prime(x)" (Hamilton), which
// cross-mixes components of the backward gradient and is incorrect. Any new
// split gate MUST use the componentwise form above.
template <typename T>
class split_activation : public Module<T> {
public:
    enum class kind { relu, sigmoid, tanh };

    explicit split_activation(kind k = kind::tanh) : kind_(k) {}

    kind activation() const { return kind_; }

    std::vector<tensor<quaternion<T>>*> parameters() override { return {}; }
    std::vector<tensor<quaternion<T>>*> gradients() override { return {}; }

    tensor<quaternion<T>> forward(const tensor<quaternion<T>>& x) override {
        x_ = x;
        tensor<quaternion<T>> y(x.shape());
        for (std::size_t i = 0; i < x.size(); ++i) y[i] = apply1(x[i]);
        return y;
    }

    tensor<quaternion<T>> backward(const tensor<quaternion<T>>& dy) override {
        assert(dy.shape() == x_.shape());
        tensor<quaternion<T>> dx(dy.shape());
        for (std::size_t i = 0; i < dy.size(); ++i) {
            const quaternion<T> g = prime1(x_[i]);
            dx[i] = quaternion<T>(dy[i].w * g.w, dy[i].x * g.x,
                                  dy[i].y * g.y, dy[i].z * g.z);
        }
        return dx;
    }

private:
    quaternion<T> apply1(const quaternion<T>& q) const {
        switch (kind_) {
            case kind::relu: return split_relu(q);
            case kind::sigmoid: return split_sigmoid(q);
            case kind::tanh: return split_tanh(q);
        }
        return q;
    }

    quaternion<T> prime1(const quaternion<T>& q) const {
        switch (kind_) {
            case kind::relu: return split_relu_prime(q);
            case kind::sigmoid: return split_sigmoid_prime(q);
            case kind::tanh: return split_tanh_prime(q);
        }
        return quaternion<T>(1, 1, 1, 1);
    }

    kind kind_;
    tensor<quaternion<T>> x_;
};

}  // namespace layers
}  // namespace nn
}  // namespace qnn

#endif  // QNN_NN_LAYERS_ACTIVATION_HPP