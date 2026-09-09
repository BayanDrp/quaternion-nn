#ifndef QNN_FUNCTIONAL_TAN_HPP
#define QNN_FUNCTIONAL_TAN_HPP

#include <cmath>
#include <cstddef>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/tensor.hpp"

namespace qnn {
namespace functional {

template <typename T>
quaternion<T> split_tanh(const quaternion<T>& q) {
    return quaternion<T>(std::tanh(q.w), std::tanh(q.x), std::tanh(q.y), std::tanh(q.z));
}

template <typename T>
quaternion<T> split_tanh_prime(const quaternion<T>& q) {
    auto d = [](T v) {
        T t = std::tanh(v);
        return T(1) - t * t;
    };
    return quaternion<T>(d(q.w), d(q.x), d(q.y), d(q.z));
}

template <typename T>
tensor<quaternion<T>> tanh(const tensor<quaternion<T>>& x) {
    tensor<quaternion<T>> y(x.shape());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = split_tanh(x[i]);
    return y;
}

template <typename T>
tensor<quaternion<T>> tanh_prime(const tensor<quaternion<T>>& x) {
    tensor<quaternion<T>> y(x.shape());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = split_tanh_prime(x[i]);
    return y;
}

}  // namespace functional
}  // namespace qnn

#endif  // QNN_FUNCTIONAL_TAN_HPP