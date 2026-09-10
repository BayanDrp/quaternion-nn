

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/nn/layers/activation.hpp"
#include "qnn/nn/layers/linear.hpp"
#include "qnn/nn/sequential.hpp"


namespace qnn {
namespace nn {
namespace models {
template <typename T = float>
class qmlp : public Module<T> {
public:
    struct Config {
        std::size_t in_features = 784;  // input size
        std::vector<std::size_t> hidden_features = {128};  // hidden layer sizes
        std::size_t out_features = 10;  // output size
        typename layers::split_activation<T>::kind activation =
            layers::split_activation<T>::kind::relu;  // activation function
        std::uint32_t seed = 7;  // RNG seed for linear init
    };

    qmlp(const Config& cfg) : net_(), flat_(0) {
        net_.push_back(layers::linear<T>(cfg.in_features, cfg.hidden_features[0], true, cfg.seed));
        for (std::size_t i = 1; i < cfg.hidden_features.size(); ++i)
            net_.push_back(layers::linear<T>(cfg.hidden_features[i - 1], cfg.hidden_features[i], true, cfg.seed));
        net_.push_back(layers::split_activation<T>(cfg.activation));
        net_.push_back(layers::linear<T>(cfg.hidden_features.back(), cfg.out_features, true, cfg.seed));
        flat_ = cfg.out_features;
    }

    std::size_t size() const { return net_.size(); }

    tensor<T> forward(const tensor<T>& x) {
        return net_.forward(x);
    }

    tensor<T> backward(const tensor<T>& dy) {
        return net_.backward(dy);
    }

    std::vector<tensor<T>*> gradients() {
        return net_.gradients();
    }

    std::vector<tensor<T>*> parameters() {
        return net_.parameters();
    }

private:
    Sequential<T> net_;
    std::size_t flat_;
};
}  // namespace models
}  // namespace nn
}  // namespace qnn

