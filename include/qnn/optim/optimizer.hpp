#ifndef QNN_OPTIM_OPTIMIZER_HPP
#define QNN_OPTIM_OPTIMIZER_HPP

#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/nn/module.hpp"

namespace qnn {
namespace optim {

template <typename T>
class optimizer {
public:
    explicit optimizer(T lr) : lr_(lr) {}
    virtual ~optimizer() = default;

    void add(tensor<quaternion<T>>& param, tensor<quaternion<T>>& grad) {
        params_.emplace_back(&param, &grad);
    }

    void add(qnn::nn::Module<T>& model) {
        const std::vector<tensor<quaternion<T>>*> ps = model.parameters();
        const std::vector<tensor<quaternion<T>>*> gs = model.gradients();
        assert(ps.size() == gs.size());
        for (std::size_t i = 0; i < ps.size(); ++i) {
            add(*ps[i], *gs[i]);
        }
    }

    void zero_grad() {
        for (auto& [param, grad] : params_) {
            for (std::size_t i = 0; i < grad->size(); ++i) {
                (*grad)[i] = quaternion<T>();
            }
        }
    }

    virtual void step() = 0;

protected:
    T lr_;
    std::vector<std::pair<tensor<quaternion<T>>*, tensor<quaternion<T>>*>> params_;
};

}  // namespace optim
}  // namespace qnn

#endif  // QNN_OPTIM_OPTIMIZER_HPP