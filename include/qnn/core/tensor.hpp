#ifndef QNN_CORE_TENSOR_HPP
#define QNN_CORE_TENSOR_HPP

#include <cassert>
#include <cstddef>
#include <vector>

#include "qnn/core/shape.hpp"

namespace qnn {

template <typename T>
class tensor {
public:
    tensor() = default;
    explicit tensor(const ::qnn::shape& s) : shape_(s), data_(s.size()) {}

    const ::qnn::shape& shape() const { return shape_; }
    std::size_t rank() const { return shape_.rank(); }
    std::size_t dim(std::size_t i) const { return shape_.dim(i); }
    std::size_t size() const { return data_.size(); }

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    void reshape(const ::qnn::shape& s) {
        assert(s.size() == data_.size());
        shape_ = s;
    }

    std::size_t flat_index(const std::vector<std::size_t>& idx) const {
        assert(idx.size() == shape_.rank());
        std::size_t n = 0;
        for (std::size_t i = 0; i < shape_.rank(); ++i) n = n * shape_.dim(i) + idx[i];
        return n;
    }

    T& operator[](std::size_t i) { return data_[i]; }
    const T& operator[](std::size_t i) const { return data_[i]; }

    T& at(const std::vector<std::size_t>& idx) { return data_[flat_index(idx)]; }
    const T& at(const std::vector<std::size_t>& idx) const { return data_[flat_index(idx)]; }

    template <typename... Idx>
    T& operator()(Idx... idx) {
        return data_[flat_index({static_cast<std::size_t>(idx)...})];
    }

    template <typename... Idx>
    const T& operator()(Idx... idx) const {
        return data_[flat_index({static_cast<std::size_t>(idx)...})];
    }

private:
    ::qnn::shape shape_;
    std::vector<T> data_;
};

}  // namespace qnn

#endif  // QNN_CORE_TENSOR_HPP