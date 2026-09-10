#ifndef QNN_NN_LAYERS_FLATTEN_HPP
#define QNN_NN_LAYERS_FLATTEN_HPP

#include <cstddef>
#include <vector>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/functional/flatten.hpp"
#include "qnn/nn/module.hpp"

namespace qnn {
namespace nn {
namespace layers {

// Flattens [N, C, H, W] -> [N, C*H*W] in forward and restores the exact input
// shape in backward. Covered shape convention (see conv2d.hpp): image tensors
// always carry an explicit channel dim, so the collapsed row-major layout is
// the identity permutation and the inverse reshape is exact.
template <typename T>
class flatten : public Module<T> {
public:
    std::vector<tensor<quaternion<T>>*> parameters() override { return {}; }
    std::vector<tensor<quaternion<T>>*> gradients() override { return {}; }

    tensor<quaternion<T>> forward(const tensor<quaternion<T>>& x) override {
        in_shape_ = x.shape();
        return qnn::functional::flatten(x);
    }

    tensor<quaternion<T>> backward(const tensor<quaternion<T>>& dy) override {
        assert(dy.size() == in_shape_.size());
        return qnn::functional::reshape(dy, in_shape_);
    }

private:
    qnn::shape in_shape_;
};

}  // namespace layers
}  // namespace nn
}  // namespace qnn

#endif  // QNN_NN_LAYERS_FLATTEN_HPP