#ifndef QNN_NN_LAYERS_ACTIVATION_HPP
#define QNN_NN_LAYERS_ACTIVATION_HPP

#include <cmath>
#include <cstddef>
#include <vector>

#include "qnn/core/quaternion.hpp"
#include "qnn/nn/module.hpp"

namespace qnn {
namespace nn {
namespace layers {

template <typename T>
quaternion<T> split_relu(const quaternion<T>& q) {
    return quaternion<T>(q.w > 0 ? q.w : T(0), q.x > 0 ? q.x : T(0),
                         q.y > 0 ? q.y : T(0), q.z > 0 ? q.z : T(0));
}

template <typename T>
quaternion<T> split_relu_prime(const quaternion<T>& q) {
    return quaternion<T>(q.w > 0 ? T(1) : T(0), q.x > 0 ? T(1) : T(0),
                         q.y > 0 ? T(1) : T(0), q.z > 0 ? T(1) : T(0));
}

template <typename T>
quaternion<T> split_sigmoid(const quaternion<T>& q) {
    auto s = [](T v) { return T(1) / (T(1) + std::exp(-v)); };
    return quaternion<T>(s(q.w), s(q.x), s(q.y), s(q.z));
}

template <typename T>
quaternion<T> split_sigmoid_prime(const quaternion<T>& q) {
    auto d = [](T v) {
        T s = T(1) / (T(1) + std::exp(-v));
        return s * (T(1) - s);
    };
    return quaternion<T>(d(q.w), d(q.x), d(q.y), d(q.z));
}

template <typename T>
quaternion<T> split_tanh(const quaternion<T>& q) {
    return quaternion<T>(std::tanh(q.w), std::tanh(q.x), std::tanh(q.y), std::tanh(q.z));
}

template <typename T>
quaternion<T> split_tanh_prime(const quaternion<T>& q) {
    auto d = [](T v) { T t = std::tanh(v); return T(1) - t * t; };
    return quaternion<T>(d(q.w), d(q.x), d(q.y), d(q.z));
}

// Split activation as a Module, so it composes inside sequential<T>.
// Applies the chosen activation elementwise to every component of every
// quaternion; backward gates dy by the activation derivative at the cached
// pre-activation input.
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