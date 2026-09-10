#ifndef QNN_NN_MODELS_QCNN_HPP
#define QNN_NN_MODELS_QCNN_HPP

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/nn/layers/activation.hpp"
#include "qnn/nn/layers/conv2d.hpp"
#include "qnn/nn/layers/flatten.hpp"
#include "qnn/nn/layers/linear.hpp"
#include "qnn/nn/layers/pool2d.hpp"
#include "qnn/nn/sequential.hpp"

namespace qnn {
namespace nn {
namespace models {

// Clean, configurable quaternion CNN.
//
// Builds the standard convolutional stack:
//   [conv(k,k) -> split-activation -> pool] repeated for each entry in
//   Config::channels, then flatten -> linear(out_features).
//
// Shape convention: images are [N, C, H, W] end to end (never implicit rank-3).
// Config::height/width describe ONE image, channels the per-block conv width.
template <typename T = float>
class qcnn : public Module<T> {
public:
    struct Config {
        std::size_t in_channels = 1;  // image channels (1 for grayscale/RGB-packed)
        std::size_t height = 14;      // input image height (one sample)
        std::size_t width = 14;       // input image width (one sample)
        std::vector<std::size_t> channels = {8, 16};  // conv width per block
        std::size_t kernel = 3;                       // square conv kernel size
        int padding = 1;
        int stride = 1;
        std::size_t pool_size = 2;
        int pool_stride = 2;
        functional::pool_type pool_type =
            functional::pool_type::max;  // max or average per block
        bool use_activation = true;  // false => conv + pool only (linear stack)
        typename layers::split_activation<T>::kind activation =
            layers::split_activation<T>::kind::tanh;
        std::size_t out_features = 10;   // linear head width
        std::uint32_t seed = 7;          // RNG seed for conv/linear init
    };

    explicit qcnn(const Config& cfg) : net_(), flat_(0) {
        std::uint32_t s = cfg.seed;
        std::size_t in = cfg.in_channels == 0 ? 1 : cfg.in_channels;
        std::size_t h = cfg.height, w = cfg.width;

        for (std::size_t out : cfg.channels) {
            assert(out > 0);
            net_.template add<layers::conv2d>(
                out, shape{cfg.kernel, cfg.kernel}, in, cfg.padding,
                cfg.stride, true, s++);
            const auto conv_out_h =
                (static_cast<long long>(h) + 2LL * cfg.padding -
                 static_cast<long long>(cfg.kernel)) /
                    cfg.stride +
                1;
            const auto conv_out_w =
                (static_cast<long long>(w) + 2LL * cfg.padding -
                 static_cast<long long>(cfg.kernel)) /
                    cfg.stride +
                1;
            h = static_cast<std::size_t>(conv_out_h);
            w = static_cast<std::size_t>(conv_out_w);

            if (cfg.use_activation) {
                net_.template add<layers::split_activation>(cfg.activation);
            }
            net_.template add<layers::pool2d>(out, shape{cfg.pool_size, cfg.pool_size},
                                              cfg.pool_stride, cfg.pool_type);
            h = (h - cfg.pool_size) / static_cast<std::size_t>(cfg.pool_stride) + 1;
            w = (w - cfg.pool_size) / static_cast<std::size_t>(cfg.pool_stride) + 1;
            assert(h > 0 && w > 0);
            in = out;
        }

        flat_ = in * h * w;
        net_.template add<layers::flatten>();
        net_.template add<layers::linear>(flat_, cfg.out_features, true, s++);
    }

    std::size_t size() const { return net_.size(); }
    Module<T>& operator[](std::size_t i) { return net_[i]; }
    const Module<T>& operator[](std::size_t i) const { return net_[i]; }
    std::size_t flat_features() const { return flat_; }

    // Total real (scalar) component count across all parameters.
    std::size_t real_parameter_count() {
        std::size_t n = 0;
        for (auto* p : parameters()) n += p->size() * 4;
        return n;
    }

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
    void zero_grad() override { net_.zero_grad(); }
    void apply_gradients(T lr) override { net_.apply_gradients(lr); }

private:
    sequential<T> net_;
    std::size_t flat_;
};

}  // namespace models
}  // namespace nn
}  // namespace qnn

#endif  // QNN_NN_MODELS_QCNN_HPP