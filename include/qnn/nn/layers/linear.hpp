#ifndef QNN_NN_LAYERS_LINEAR_HPP
#define QNN_NN_LAYERS_LINEAR_HPP

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/nn/init.hpp"
#include "qnn/nn/module.hpp"
#include "qnn/nn/parameter.hpp"

namespace qnn {
namespace nn {
namespace layers {

template <typename T = float>
class linear : public qnn::nn::Module<T> {
public:
    linear(std::size_t in_features, std::size_t out_features, bool use_bias = true,
           std::uint32_t seed = 42)
        : in_(in_features),
          out_(out_features),
          use_bias_(use_bias),
          weight_(::qnn::shape{out_features, in_features}),
          bias_(::qnn::shape{out_features}) {
        std::mt19937 rng(seed);
        qnn::nn::xavier(weight_.value(), in_features, out_features, rng);
        zero_grad();
    }

    std::size_t in_features() const { return in_; }
    std::size_t out_features() const { return out_; }
    bool has_bias() const { return use_bias_; }

    tensor<quaternion<T>>& weight() { return weight_.value(); }
    const tensor<quaternion<T>>& weight() const { return weight_.value(); }
    tensor<quaternion<T>>& bias() { return bias_.value(); }
    const tensor<quaternion<T>>& bias() const { return bias_.value(); }
    tensor<quaternion<T>>& dweight() { return weight_.grad(); }
    const tensor<quaternion<T>>& dweight() const { return weight_.grad(); }
    tensor<quaternion<T>>& dbias() { return bias_.grad(); }
    const tensor<quaternion<T>>& dbias() const { return bias_.grad(); }

    std::vector<tensor<quaternion<T>>*> parameters() override {
        std::vector<tensor<quaternion<T>>*> ps;
        ps.push_back(&weight_.value());
        if (use_bias_) ps.push_back(&bias_.value());
        return ps;
    }

    std::vector<tensor<quaternion<T>>*> gradients() override {
        std::vector<tensor<quaternion<T>>*> gs;
        gs.push_back(&weight_.grad());
        if (use_bias_) gs.push_back(&bias_.grad());
        return gs;
    }

    tensor<quaternion<T>> forward(const tensor<quaternion<T>>& x) {
        assert(x.rank() == 2);
        assert(x.dim(1) == in_);
        const std::size_t batch = x.dim(0);

        tensor<quaternion<T>> out(::qnn::shape{batch, out_});
#pragma omp parallel for collapse(2) schedule(static)
        for (std::size_t i = 0; i < batch; ++i) {
            for (std::size_t j = 0; j < out_; ++j) {
                quaternion<T> acc{};
                for (std::size_t k = 0; k < in_; ++k) {
                    acc = acc + weight_.value()(j, k) * x(i, k);
                }
                if (use_bias_) acc = acc + bias_.value()[j];
                out(i, j) = acc;
            }
        }
        x_ = x;
        return out;
    }

    tensor<quaternion<T>> backward(const tensor<quaternion<T>>& dy) {
        assert(dy.rank() == 2);
        assert(dy.dim(0) == x_.dim(0));
        assert(dy.dim(1) == out_);
        const std::size_t batch = x_.dim(0);

        tensor<quaternion<T>> dx(::qnn::shape{batch, in_});
#pragma omp parallel for schedule(static)
        for (std::size_t j = 0; j < out_; ++j) {
            if (use_bias_) {
                quaternion<T> acc{};
                for (std::size_t i = 0; i < batch; ++i) acc = acc + dy(i, j);
                bias_.grad()[j] = bias_.grad()[j] + acc;
            }
            for (std::size_t k = 0; k < in_; ++k) {
                quaternion<T> acc{};
                for (std::size_t i = 0; i < batch; ++i) {
                    acc = acc + dy(i, j) * x_(i, k).conjugate();
                }
                weight_.grad()(j, k) = weight_.grad()(j, k) + acc;
            }
        }
#pragma omp parallel for collapse(2) schedule(static)
        for (std::size_t i = 0; i < batch; ++i) {
            for (std::size_t k = 0; k < in_; ++k) {
                quaternion<T> acc{};
                for (std::size_t j = 0; j < out_; ++j) {
                    acc = acc + weight_.value()(j, k).conjugate() * dy(i, j);
                }
                dx(i, k) = acc;
            }
        }
        return dx;
    }

    void apply_gradients(T lr) {
        for (std::size_t i = 0; i < weight_.value().size(); ++i) {
            weight_.value()[i] = weight_.value()[i] - weight_.grad()[i] * lr;
        }
        if (use_bias_) {
            for (std::size_t i = 0; i < bias_.value().size(); ++i) {
                bias_.value()[i] = bias_.value()[i] - bias_.grad()[i] * lr;
            }
        }
        zero_grad();
    }

    void zero_grad() {
        weight_.zero_grad();
        bias_.zero_grad();
    }

private:
    std::size_t in_;
    std::size_t out_;
    bool use_bias_;
    Parameter<T> weight_;
    Parameter<T> bias_;
    tensor<quaternion<T>> x_;
};

}  // namespace layers
}  // namespace nn
}  // namespace qnn

#endif  // QNN_NN_LAYERS_LINEAR_HPP