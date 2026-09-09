#ifndef QNN_FUNCTIONAL_FLATTEN_HPP
#define QNN_FUNCTIONAL_FLATTEN_HPP

#include <cassert>
#include <cstddef>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"

namespace qnn {
namespace functional {

// Element-preserving view of x with a new shape (row-major order kept).
template <typename T>
tensor<quaternion<T>> reshape(const tensor<quaternion<T>>& x,
                              const ::qnn::shape& s) {
    assert(s.size() == x.size());
    tensor<quaternion<T>> out = x;
    out.reshape(s);
    return out;
}

// Collapse all trailing dims: [d0, d1, ..., dn] -> [d0, d1*...*dn].
// Rank-1 input -> [1, n]. Keeps batch dim 0 so output feeds linear<T>.
template <typename T>
tensor<quaternion<T>> flatten(const tensor<quaternion<T>>& x) {
    assert(x.rank() >= 1);
    const std::size_t outer = (x.rank() >= 2) ? x.dim(0) : 1;
    assert(outer != 0 && x.size() % outer == 0);
    return reshape(x, ::qnn::shape{outer, x.size() / outer});
}

}  // namespace functional
}  // namespace qnn

#endif  // QNN_FUNCTIONAL_FLATTEN_HPP