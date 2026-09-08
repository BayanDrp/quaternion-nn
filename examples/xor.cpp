#include <cmath>
#include <cstdio>
#include <cstddef>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/loss/mse.hpp"
#include "qnn/nn/layers/activation.hpp"
#include "qnn/nn/layers/linear.hpp"
#include "qnn/nn/module.hpp"
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

    std::vector<tensor<qf>*> parameters() override {
        std::vector<tensor<qf>*> ps = l1.parameters();
        const std::vector<tensor<qf>*> ps2 = l2.parameters();
        ps.insert(ps.end(), ps2.begin(), ps2.end());
        return ps;
    }

    std::vector<tensor<qf>*> gradients() override {
        std::vector<tensor<qf>*> gs = l1.gradients();
        const std::vector<tensor<qf>*> gs2 = l2.gradients();
        gs.insert(gs.end(), gs2.begin(), gs2.end());
        return gs;
    }

private:
    linear<float> l1;
    linear<float> l2;
    tensor<qf> pre1_;
};
}  // namespace

int main() {
    const int bits[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

    tensor<qf> x(shape{4, 2});
    tensor<qf> tgt(shape{4, 1});
    for (std::size_t i = 0; i < 4; ++i) {
        x(i, 0) = qf(bits[i][0] ? 1.0f : -1.0f, 0, 0, 0);
        x(i, 1) = qf(bits[i][1] ? 1.0f : -1.0f, 0, 0, 0);
        tgt(i, 0) = qf(bits[i][0] ^ bits[i][1] ? 1.0f : -1.0f, 0, 0, 0);
    }

    MLP model;
    qnn::optim::sgd<float> opt(0.6f);
    opt.add(model);

    const int steps = 3000;

    for (int step = 0; step < steps; ++step) {
        opt.zero_grad();
        tensor<qf> y = model.forward(x);
        float loss = mse(y, tgt);

        const float scale = 2.0f / (4 * 1 * 4.0f);
        tensor<qf> dy(shape{4, 1});
        for (std::size_t i = 0; i < 4; ++i) dy(i, 0) = (y(i, 0) - tgt(i, 0)) * scale;

        model.backward(dy);
        opt.step();

        if (step % 300 == 0) std::printf("step %4d  loss %.5f\n", step, loss);
    }

    tensor<qf> y = model.forward(x);
    int correct = 0;
    for (int i = 0; i < 4; ++i) {
        float pred = y(i, 0).w > 0 ? 1.0f : -1.0f;
        int expected = bits[i][0] ^ bits[i][1];
        if ((pred > 0) == (expected == 1)) ++correct;
        std::printf("%d XOR %d = %+.2f  (expected %d)\n", bits[i][0], bits[i][1],
                    y(i, 0).w, expected);
    }
    std::printf("accuracy %d/4\n", correct);
    return correct == 4 ? 0 : 1;
}