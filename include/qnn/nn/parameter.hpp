#ifndef QNN_NN_PARAMETER_HPP
#define QNN_NN_PARAMETER_HPP

#include <cstddef>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"

namespace qnn {
namespace nn {

template <typename T = float>
class Parameter {
public:
    explicit Parameter(qnn::shape s) : value_(std::move(s)), grad_(value_.shape()) {}

    tensor<quaternion<T>>& value() { return value_; }
    const tensor<quaternion<T>>& value() const { return value_; }
    tensor<quaternion<T>>& grad() { return grad_; }
    const tensor<quaternion<T>>& grad() const { return grad_; }

    void zero_grad() {
        for (std::size_t i = 0; i < grad_.size(); ++i) grad_[i] = quaternion<T>();
    }

private:
    tensor<quaternion<T>> value_;
    tensor<quaternion<T>> grad_;
};

}  // namespace nn
}  // namespace qnn

#endif  // QNN_NN_PARAMETER_HPP