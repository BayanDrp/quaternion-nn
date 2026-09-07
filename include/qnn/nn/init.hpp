#ifndef QNN_NN_INIT_HPP
#define QNN_NN_INIT_HPP

#include <cmath>
#include <cstddef>
#include <random>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/tensor.hpp"

namespace qnn {
namespace nn {

template <typename T>
void uniform(tensor<quaternion<T>>& t, T lo, T hi, std::mt19937& rng) {
    std::uniform_real_distribution<T> dist(lo, hi);
    for (std::size_t i = 0; i < t.size(); ++i) {
        quaternion<T>& q = t[i];
        q = quaternion<T>(dist(rng), dist(rng), dist(rng), dist(rng));
    }
}

template <typename T>
void xavier(tensor<quaternion<T>>& t, std::size_t fan_in, std::size_t fan_out,
            std::mt19937& rng) {
    T bound = std::sqrt(T(6) / T(fan_in + fan_out));
    uniform(t, -bound, bound, rng);
}

}  // namespace nn
}  // namespace qnn

#endif  // QNN_NN_INIT_HPP