#ifndef QNN_NN_LAYERS_ACTIVATION_HPP
#define QNN_NN_LAYERS_ACTIVATION_HPP

#include <cmath>

#include "qnn/core/quaternion.hpp"

namespace qnn {
namespace nn {
namespace layers {

template <typename T>
quaternion<T> split_relu(const quaternion<T>& q) {
    return quaternion<T>(q.w > 0 ? q.w : T(0), q.x > 0 ? q.x : T(0),
                         q.y > 0 ? q.y : T(0), q.z > 0 ? q.z : T(0));
}

template <typename T>
quaternion<T> split_sigmoid(const quaternion<T>& q) {
    auto s = [](T v) { return T(1) / (T(1) + std::exp(-v)); };
    return quaternion<T>(s(q.w), s(q.x), s(q.y), s(q.z));
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

}  // namespace layers
}  // namespace nn
}  // namespace qnn

#endif  // QNN_NN_LAYERS_ACTIVATION_HPP