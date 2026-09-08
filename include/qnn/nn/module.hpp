#ifndef QNN_NN_MODULE_HPP
#define QNN_NN_MODULE_HPP

#include <cstddef>
#include <vector>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/tensor.hpp"

namespace qnn {
namespace nn {

template <typename T>
class Module {
public:
    virtual ~Module() = default;

    virtual tensor<quaternion<T>> forward(const tensor<quaternion<T>>& x) = 0;
    virtual tensor<quaternion<T>> backward(const tensor<quaternion<T>>& dy) = 0;

    virtual std::vector<tensor<quaternion<T>>*> parameters() = 0;
    virtual std::vector<tensor<quaternion<T>>*> gradients() = 0;
};

}  // namespace nn
}  // namespace qnn

#endif  // QNN_NN_MODULE_HPP