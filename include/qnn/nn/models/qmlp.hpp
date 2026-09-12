#ifndef QNN_NN_MODELS_QMLP_HPP
#define QNN_NN_MODELS_QMLP_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/nn/layers/activation.hpp"
#include "qnn/nn/layers/linear.hpp"
#include "qnn/nn/module.hpp"
#include "qnn/nn/sequential.hpp"

namespace qnn {
namespace nn {
namespace models {

// Quaternion MLP: hidden blocks of linear + split activation, then a linear
// output head. Built on nn::sequential, so forward / backward / parameters /
// gradients all chain through the component layers.
template <typename T = float>
class qmlp : public Module<T> {
public:
    struct Config {
        std::size_t in_features = 196;  // MNIST loader yields {N,14,14} quats = 196
        std::vector<std::size_t> hidden_features = {128};
        std::size_t out_features = 10;
        typename layers::split_activation<T>::kind activation =
            layers::split_activation<T>::kind::relu;
        std::uint32_t seed = 7;
    };

    explicit qmlp(const Config& cfg) {
        if (cfg.hidden_features.empty()) {
            net_.template add<layers::linear>(cfg.in_features, cfg.out_features, true, cfg.seed);
            return;
        }
        std::size_t prev = cfg.in_features;
        for (const std::size_t h : cfg.hidden_features) {
            net_.template add<layers::linear>(prev, h, true, cfg.seed);
            net_.template add<layers::split_activation>(cfg.activation);
            prev = h;
        }
        net_.template add<layers::linear>(prev, cfg.out_features, true, cfg.seed);
    }

    std::size_t size() const { return net_.size(); }

    tensor<quaternion<T>> forward(const tensor<quaternion<T>>& x) override {
        return net_.forward(x);
    }

    tensor<quaternion<T>> backward(const tensor<quaternion<T>>& dy) override {
        return net_.backward(dy);
    }

    std::vector<tensor<quaternion<T>>*> parameters() override {
        return net_.parameters();
    }

    std::vector<tensor<quaternion<T>>*> gradients() override {
        return net_.gradients();
    }

private:
    qnn::nn::sequential<T> net_;
};

}  // namespace models
}  // namespace nn
}  // namespace qnn

#endif  // QNN_NN_MODELS_QMLP_HPP