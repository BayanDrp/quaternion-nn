#ifndef QNN_LOSS_CROSS_ENTROPY_HPP
#define QNN_LOSS_CROSS_ENTROPY_HPP

#include <cmath>
#include <cstddef>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/tensor.hpp"

namespace qnn {
namespace loss {

template <typename T>
T cross_entropy(const tensor<quaternion<T>>& target, const tensor<quaternion<T>>& pred) {
    T sum{};
    const T e = T(1e-9);
    for (std::size_t i = 0; i < target.size(); ++i) {
        const quaternion<T>& t = target[i];
        const quaternion<T>& p = pred[i];
        sum -= t.w * std::log(p.w + e);
        sum -= t.x * std::log(p.x + e);
        sum -= t.y * std::log(p.y + e);
        sum -= t.z * std::log(p.z + e);
    }
    return sum / T(target.size());
}

template <typename T>
tensor<quaternion<T>> cross_entropy_grad(const tensor<quaternion<T>>& pred,
                                         const tensor<quaternion<T>>& target) {
    tensor<quaternion<T>> g(pred.shape());
    for (std::size_t i = 0; i < pred.size(); ++i) {
        g[i] = pred[i] - target[i];
    }
    return g;
}

}  // namespace loss
}  // namespace qnn

#endif  // QNN_LOSS_CROSS_ENTROPY_HPP