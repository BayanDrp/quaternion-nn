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

    // Default implementations operate on the parameters()/gradients() lists, so
    // parameterless modules (pool, flatten, activators) inherit no-op versions.
    virtual void zero_grad() {
        for (auto* g : gradients()) {
            for (std::size_t i = 0; i < g->size(); ++i) (*g)[i] = quaternion<T>();
        }
    }

    virtual void apply_gradients(T lr) {
        const std::vector<tensor<quaternion<T>>*> ps = parameters();
        const std::vector<tensor<quaternion<T>>*> gs = gradients();
        if (ps.size() != gs.size()) return;
        for (std::size_t i = 0; i < ps.size(); ++i) {
            for (std::size_t j = 0; j < ps[i]->size(); ++j) {
                (*ps[i])[j] = (*ps[i])[j] - (*gs[i])[j] * lr;
            }
        }
        zero_grad();
    }
};

}  // namespace nn
}  // namespace qnn

#endif  // QNN_NN_MODULE_HPP