#ifndef QNN_FUNCTIONAL_RELU_HPP
#define QNN_FUNCTIONAL_RELU_HPP

#include <cstddef>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/tensor.hpp"

namespace qnn {
namespace functional {

template <typename T>
quaternion<T> split_relu(const quaternion<T>& q) {
    return quaternion<T>(q.w > T(0) ? q.w : T(0), q.x > T(0) ? q.x : T(0),
                         q.y > T(0) ? q.y : T(0), q.z > T(0) ? q.z : T(0));
}

template <typename T>
quaternion<T> split_relu_prime(const quaternion<T>& q) {
    return quaternion<T>(q.w > T(0) ? T(1) : T(0), q.x > T(0) ? T(1) : T(0),
                         q.y > T(0) ? T(1) : T(0), q.z > T(0) ? T(1) : T(0));
}

template <typename T>
tensor<quaternion<T>> relu(const tensor<quaternion<T>>& x) {
    tensor<quaternion<T>> y(x.shape());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = split_relu(x[i]);
    return y;
}

template <typename T>
tensor<quaternion<T>> relu_prime(const tensor<quaternion<T>>& x) {
    tensor<quaternion<T>> y(x.shape());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = split_relu_prime(x[i]);
    return y;
}

}  // namespace functional
}  // namespace qnn

#endif  // QNN_FUNCTIONAL_RELU_HPP