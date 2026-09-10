#ifndef QNN_FUNCTIONAL_SIGMOID_HPP
#define QNN_FUNCTIONAL_SIGMOID_HPP

#include <cmath>
#include <cstddef>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/tensor.hpp"

namespace qnn {
namespace functional {

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
tensor<quaternion<T>> sigmoid(const tensor<quaternion<T>>& x) {
    tensor<quaternion<T>> y(x.shape());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = split_sigmoid(x[i]);
    return y;
}

template <typename T>
tensor<quaternion<T>> sigmoid_prime(const tensor<quaternion<T>>& x) {
    tensor<quaternion<T>> y(x.shape());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = split_sigmoid_prime(x[i]);
    return y;
}

}  // namespace functional
}  // namespace qnn

#endif  // QNN_FUNCTIONAL_SIGMOID_HPP
