// Demonstrates the framework's core API: Module<T> and Parameter<T>.
//
// Build a tiny Module by hand, inspect its Parameters, bind the whole model
// to an optimizer with a single call, and train XOR.
//
// Key calls to notice:
//   model.parameters() / model.gradients()  -- flat lists of the Parameter tensors
//   opt.add(model)                          -- one call binds all Parameters
//   model.forward(x) / model.backward(dy)   -- dispatch through the Module interface

#include <cstddef>
#include <cstdio>
#include <vector>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/loss/mse.hpp"
#include "qnn/nn/layers/activation.hpp"
#include "qnn/nn/layers/linear.hpp"
#include "qnn/nn/module.hpp"
#include "qnn/nn/parameter.hpp"
#include "qnn/optim/sgd.hpp"

using qnn::loss::mse;
using qnn::nn::Module;
using qnn::nn::layers::linear;
using qnn::nn::layers::split_tanh;
using qnn::quaternion;
using qnn::shape;
using qnn::tensor;
using qf = quaternion<float>;

namespace {
constexpr std::size_t kBatch = 4;

qf tanh_prime(const qf& q) {
    float tw = 1 - std::tanh(q.w) * std::tanh(q.w);
    float tx = 1 - std::tanh(q.x) * std::tanh(q.x);
    float ty = 1 - std::tanh(q.y) * std::tanh(q.y);
    float tz = 1 - std::tanh(q.z) * std::tanh(q.z);
    return qf(tw, tx, ty, tz);
}

// ----  the Model  --------------------------------------------------------
// A Module aggregates child layers and flattens their Parameters so that
// `opt.add(model)` binds every weight in one call.

class MLP : public Module<float> {
public:
    MLP() : l1(2, 4, true, 7), l2(4, 1, true, 7), pre1_(shape{kBatch, 4}) {}

    tensor<qf> forward(const tensor<qf>& x) override {
        tensor<qf> z1 = l1.forward(x);
        pre1_ = z1;
        tensor<qf> h(shape{kBatch, 4});
        for (std::size_t i = 0; i < kBatch; ++i)
            for (std::size_t j = 0; j < 4; ++j) h(i, j) = split_tanh(z1(i, j));
        return l2.forward(h);
    }

    tensor<qf> backward(const tensor<qf>& dy) override {
        tensor<qf> dyh = l2.backward(dy);
        for (std::size_t i = 0; i < kBatch; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                const qf gate = tanh_prime(pre1_(i, j));
                const qf g = dyh(i, j);
                dyh(i, j) = qf(g.w * gate.w, g.x * gate.x, g.y * gate.y, g.z * gate.z);
            }
        }
        return l1.backward(dyh);
    }

    // Flatten child Parameters so opt.add(model) sees them all.
    std::vector<tensor<qf>*> parameters() override {
        std::vector<tensor<qf>*> ps = l1.parameters();
        const std::vector<tensor<qf>*> tail = l2.parameters();
        ps.insert(ps.end(), tail.begin(), tail.end());
        return ps;
    }
    std::vector<tensor<qf>*> gradients() override {
        std::vector<tensor<qf>*> gs = l1.gradients();
        const std::vector<tensor<qf>*> tail = l2.gradients();
        gs.insert(gs.end(), tail.begin(), tail.end());
        return gs;
    }

private:
    linear<float> l1, l2;
    tensor<qf> pre1_;
};
}  // namespace

int main() {
    // --- inspect a single Parameter --------------------------------------
    linear<float> layer(3, 2, true, 1);
    std::puts("Parameter introspection:");
    std::printf("  params: %zu  grads: %zu\n", layer.parameters().size(),
                layer.gradients().size());
    std::printf("  weight: %zux%zu  grad: %zux%zu\n", layer.weight().dim(0),
                layer.weight().dim(1), layer.dweight().dim(0), layer.dweight().dim(1));
    qf w00 = layer.weight()(0, 0);
    std::printf("  weight[0,0] = (%.4f %.4f %.4f %.4f)\n", w00.w, w00.x, w00.y, w00.z);

    // --- bind a whole Model to the optimizer ------------------------------
    MLP model;
    qnn::optim::sgd<float> opt(0.6f);
    opt.add(model);
    std::printf("bound %zu parameters to optimizer\n", model.parameters().size());

    // --- XOR data --------------------------------------------------------
    const int bits[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    tensor<qf> x(shape{kBatch, 2});
    tensor<qf> tgt(shape{kBatch, 1});
    for (std::size_t i = 0; i < kBatch; ++i) {
        x(i, 0) = qf(bits[i][0] ? 1.0f : -1.0f, 0, 0, 0);
        x(i, 1) = qf(bits[i][1] ? 1.0f : -1.0f, 0, 0, 0);
        tgt(i, 0) = qf(bits[i][0] ^ bits[i][1] ? 1.0f : -1.0f, 0, 0, 0);
    }

    // --- train -----------------------------------------------------------
    for (int step = 0; step < 3000; ++step) {
        opt.zero_grad();
        tensor<qf> y = model.forward(x);
        float loss = mse(y, tgt);

        const float scale = 2.0f / (kBatch * 1 * 4.0f);
        tensor<qf> dy(shape{kBatch, 1});
        for (std::size_t i = 0; i < kBatch; ++i) dy(i, 0) = (y(i, 0) - tgt(i, 0)) * scale;
        model.backward(dy);
        opt.step();

        if (step % 1000 == 0) std::printf("step %4d  loss %.5f\n", step, loss);
    }

    // --- evaluate --------------------------------------------------------
    tensor<qf> y = model.forward(x);
    int correct = 0;
    for (int i = 0; i < 4; ++i) {
        int pred = y(i, 0).w > 0 ? 1 : 0;
        int want = bits[i][0] ^ bits[i][1];
        if (pred == want) ++correct;
        std::printf("%d XOR %d = %d (want %d)\n", bits[i][0], bits[i][1], pred, want);
    }
    std::printf("accuracy %d/4\n", correct);
    return correct == 4 ? 0 : 1;
}