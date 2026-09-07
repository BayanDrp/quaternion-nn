#ifndef QNN_FUNCTIONAL_MATMUL_HPP
#define QNN_FUNCTIONAL_MATMUL_HPP

#include <cassert>
#include <cstddef>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/tensor.hpp"

namespace qnn {
namespace functional {

template <typename T>
tensor<T> matmul(const tensor<T>& a, const tensor<T>& b) {
    assert(a.rank() == 2);
    assert(b.rank() == 2);
    assert(a.dim(1) == b.dim(0));

    const std::size_t m = a.dim(0);
    const std::size_t k = a.dim(1);
    const std::size_t n = b.dim(1);

    tensor<T> c(::qnn::shape{m, n});
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            T acc{};
            for (std::size_t t = 0; t < k; ++t) acc = acc + a(i, t) * b(t, j);
            c(i, j) = acc;
        }
    }
    return c;
}

template <typename T>
tensor<T> matvec(const tensor<T>& a, const tensor<T>& v) {
    assert(a.rank() == 2);
    assert(v.rank() == 1);
    assert(a.dim(1) == v.size());

    const std::size_t m = a.dim(0);
    const std::size_t k = a.dim(1);

    tensor<T> c(::qnn::shape{m});
    for (std::size_t i = 0; i < m; ++i) {
        T acc{};
        for (std::size_t t = 0; t < k; ++t) acc = acc + a(i, t) * v[t];
        c[i] = acc;
    }
    return c;
}

}  // namespace functional
}  // namespace qnn

#endif  // QNN_FUNCTIONAL_MATMUL_HPP