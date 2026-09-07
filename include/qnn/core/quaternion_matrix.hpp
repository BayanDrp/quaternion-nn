#ifndef QNN_CORE_QUATERNION_MATRIX_HPP
#define QNN_CORE_QUATERNION_MATRIX_HPP

#include <cassert>
#include <cstddef>

#include "qnn/core/tensor.hpp"

namespace qnn {

template <typename T>
class quaternion_matrix {
public:
    explicit quaternion_matrix(tensor<T>& t) : t_(t) { assert(t.rank() == 2); }

    std::size_t rows() const { return t_.dim(0); }
    std::size_t cols() const { return t_.dim(1); }

    T& operator()(std::size_t r, std::size_t c) { return t_(r, c); }
    const T& operator()(std::size_t r, std::size_t c) const { return t_(r, c); }

    T& at(std::size_t r, std::size_t c) { return t_.at({r, c}); }
    const T& at(std::size_t r, std::size_t c) const { return t_.at({r, c}); }

    T* data() { return t_.data(); }
    const T* data() const { return t_.data(); }

    tensor<T>& storage() { return t_; }
    const tensor<T>& storage() const { return t_; }

private:
    tensor<T>& t_;
};

}  // namespace qnn

#endif  // QNN_CORE_QUATERNION_MATRIX_HPP