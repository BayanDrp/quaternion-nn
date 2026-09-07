#ifndef QNN_CORE_QUATERNION_VECTOR_HPP
#define QNN_CORE_QUATERNION_VECTOR_HPP

#include <cassert>
#include <cstddef>

#include "qnn/core/tensor.hpp"

namespace qnn {

template <typename T>
class quaternion_vector {
public:
    explicit quaternion_vector(tensor<T>& t) : t_(t) { assert(t.rank() == 1); }

    std::size_t size() const { return t_.size(); }

    T& operator[](std::size_t i) { return t_[i]; }
    const T& operator[](std::size_t i) const { return t_[i]; }

    T& at(std::size_t i) { return t_.at({i}); }
    const T& at(std::size_t i) const { return t_.at({i}); }

    T* data() { return t_.data(); }
    const T* data() const { return t_.data(); }

    tensor<T>& storage() { return t_; }
    const tensor<T>& storage() const { return t_; }

private:
    tensor<T>& t_;
};

}  // namespace qnn

#endif  // QNN_CORE_QUATERNION_VECTOR_HPP