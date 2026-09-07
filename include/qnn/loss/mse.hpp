#ifndef QNN_LOSS_MSE_HPP
#define QNN_LOSS_MSE_HPP

#include <cstddef>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/tensor.hpp"

namespace qnn {
namespace loss {

template <typename T>
T mse(const tensor<quaternion<T>>& y, const tensor<quaternion<T>>& target) {
    T sum{};
    for (std::size_t i = 0; i < y.size(); ++i) {
        const quaternion<T>& a = y[i];
        const quaternion<T>& b = target[i];
        const T dw = a.w - b.w;
        const T dx = a.x - b.x;
        const T dy = a.y - b.y;
        const T dz = a.z - b.z;
        sum += dw * dw + dx * dx + dy * dy + dz * dz;
    }
    return sum / (T(y.size()) * T(4));
}

}  // namespace loss
}  // namespace qnn

#endif  // QNN_LOSS_MSE_HPP