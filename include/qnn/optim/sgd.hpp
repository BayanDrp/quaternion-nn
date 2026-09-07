#ifndef QNN_OPTIM_SGD_HPP
#define QNN_OPTIM_SGD_HPP

#include <cstddef>

#include "qnn/optim/optimizer.hpp"

namespace qnn {
namespace optim {

template <typename T>
class sgd : public optimizer<T> {
public:
    explicit sgd(T lr) : optimizer<T>(lr) {}

    void step() override {
        for (auto& [param, grad] : this->params_) {
            for (std::size_t i = 0; i < param->size(); ++i) {
                (*param)[i] = (*param)[i] - (*grad)[i] * this->lr_;
            }
        }
    }
};

}  // namespace optim
}  // namespace qnn

#endif  // QNN_OPTIM_SGD_HPP