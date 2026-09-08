#ifndef QNN_FUNCTIONAL_SOFTMAX_HPP
#define QNN_FUNCTIONAL_SOFTMAX_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/tensor.hpp"

namespace qnn {
namespace functional {

template <typename T>
tensor<quaternion<T>> split_softmax(const tensor<quaternion<T>>& x) {
    tensor<quaternion<T>> y(x.shape());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const quaternion<T>& q = x[i];
        const T m = std::max(std::max(q.w, q.x), std::max(q.y, q.z));
        const quaternion<T> e(std::exp(q.w - m), std::exp(q.x - m),
                              std::exp(q.y - m), std::exp(q.z - m));
        const T s = e.w + e.x + e.y + e.z;
        y[i] = quaternion<T>(e.w / s, e.x / s, e.y / s, e.z / s);
    }
    return y;
}

template <typename T>
tensor<T> softmax(const tensor<T>& x) {
    tensor<T> y(x.shape());
    if (x.size() == 0) return y;
    T m = x[0];
    for (std::size_t i = 1; i < x.size(); ++i) {
        if (x[i] > m) m = x[i];
    }
    for (std::size_t i = 0; i < x.size(); ++i) {
        y[i] = std::exp(x[i] - m);
    }
    T sum{};
    for (std::size_t i = 0; i < y.size(); ++i) {
        sum += y[i];
    }
    for (std::size_t i = 0; i < y.size(); ++i) {
        y[i] /= sum;
    }
    return y;
}

}  // namespace functional
}  // namespace qnn

#endif  // QNN_FUNCTIONAL_SOFTMAX_HPP