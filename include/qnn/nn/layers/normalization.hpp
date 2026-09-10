#ifndef QNN_NN_LAYERS_NORMALIZATION_HPP
#define QNN_NN_LAYERS_NORMALIZATION_HPP

#include <cassert>
#include <cstddef>
#include <vector>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/functional/normalization.hpp"
#include "qnn/nn/module.hpp"
#include "qnn/nn/parameter.hpp"

namespace qnn {
namespace nn {
namespace layers {

// Split-component LayerNorm as a Module. Normalizes the w/x/y/z components
// independently over the LAST axis (d_model), then applies the learnable
// per-feature scale/bias:
//   y(j) = gamma(j) . z + beta(j),   z = (x - mean)/sqrt(var + eps)
// (the . is a componentwise product, NOT a Hamilton product). gamma starts at
// (1,1,1,1) and beta at 0, so the layer is the identity at initialization.
template <typename T = float>
class normalization : public Module<T> {
public:
    explicit normalization(std::size_t d_model, T eps = T(1e-5))
        : d_model_(d_model),
          eps_(eps),
          gamma_(::qnn::shape{d_model}),
          beta_(::qnn::shape{d_model}) {
        for (std::size_t j = 0; j < d_model_; ++j) {
            gamma_.value()[j] = quaternion<T>(1, 1, 1, 1);  // identity scale
            beta_.value()[j] = quaternion<T>();             // zero shift
        }
        this->zero_grad();
    }

    std::size_t d_model() const { return d_model_; }

    tensor<quaternion<T>>& gamma() { return gamma_.value(); }
    const tensor<quaternion<T>>& gamma() const { return gamma_.value(); }
    tensor<quaternion<T>>& beta() { return beta_.value(); }
    const tensor<quaternion<T>>& beta() const { return beta_.value(); }

    std::vector<tensor<quaternion<T>>*> parameters() override {
        return {&gamma_.value(), &beta_.value()};
    }

    std::vector<tensor<quaternion<T>>*> gradients() override {
        return {&gamma_.grad(), &beta_.grad()};
    }

    tensor<quaternion<T>> forward(const tensor<quaternion<T>>& x) override {
        assert(x.dim(x.rank() - 1) == d_model_);
        x_ = x;
        const functional::layer_norm_out<T> r =
            functional::layer_norm_forward(x, gamma_.value(), beta_.value(), eps_);
        stats_ = r.stats;
        return r.y;
    }

    tensor<quaternion<T>> backward(const tensor<quaternion<T>>& dy) override {
        assert(dy.shape() == x_.shape());
        const functional::layer_norm_grad<T> g =
            functional::layer_norm_backward(dy, x_, gamma_.value(), stats_);
        for (std::size_t j = 0; j < d_model_; ++j) {
            gamma_.grad()[j] = gamma_.grad()[j] + g.dgamma[j];
            beta_.grad()[j] = beta_.grad()[j] + g.dbeta[j];
        }
        return g.dx;
    }

private:
    std::size_t d_model_;
    T eps_;
    Parameter<T> gamma_;
    Parameter<T> beta_;
    tensor<quaternion<T>> x_;
    functional::layer_norm_stats<T> stats_;
};

}  // namespace layers
}  // namespace nn
}  // namespace qnn

#endif  // QNN_NN_LAYERS_NORMALIZATION_HPP