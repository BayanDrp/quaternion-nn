#include <cmath>
#include <cstdio>
#include <cstddef>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/loss/mse.hpp"
#include "qnn/nn/layers/activation.hpp"
#include "qnn/nn/layers/linear.hpp"
#include "qnn/optim/sgd.hpp"

using qnn::loss::mse;
using qnn::nn::layers::linear;
using qnn::nn::layers::split_tanh;
using qnn::quaternion;
using qnn::shape;
using qnn::tensor;
using qf = quaternion<float>;

static qf tanh_prime(const qf& q) {
    float tw = 1 - std::tanh(q.w) * std::tanh(q.w);
    float tx = 1 - std::tanh(q.x) * std::tanh(q.x);
    float ty = 1 - std::tanh(q.y) * std::tanh(q.y);
    float tz = 1 - std::tanh(q.z) * std::tanh(q.z);
    return qf(tw, tx, ty, tz);
}

int main() {
    const int bits[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

    tensor<qf> x(shape{4, 2});
    tensor<qf> tgt(shape{4, 1});
    for (std::size_t i = 0; i < 4; ++i) {
        x(i, 0) = qf(bits[i][0] ? 1.0f : -1.0f, 0, 0, 0);
        x(i, 1) = qf(bits[i][1] ? 1.0f : -1.0f, 0, 0, 0);
        tgt(i, 0) = qf(bits[i][0] ^ bits[i][1] ? 1.0f : -1.0f, 0, 0, 0);
    }

    linear<float> l1(2, 4, true, 7);
    linear<float> l2(4, 1, true, 7);

    qnn::optim::sgd<float> opt(0.6f);
    opt.add(l1.weight(), l1.dweight());
    opt.add(l1.bias(), l1.dbias());
    opt.add(l2.weight(), l2.dweight());
    opt.add(l2.bias(), l2.dbias());

    const int steps = 3000;

    for (int step = 0; step < steps; ++step) {
        opt.zero_grad();

        tensor<qf> y1 = l1.forward(x);

        tensor<qf> h(shape{4, 4});
        for (std::size_t i = 0; i < 4; ++i)
            for (std::size_t j = 0; j < 4; ++j) h(i, j) = split_tanh(y1(i, j));

        tensor<qf> y2 = l2.forward(h);
        float loss = mse(y2, tgt);

        const float scale = 2.0f / (4 * 1 * 4.0f);
        tensor<qf> dy2(shape{4, 1});
        for (std::size_t i = 0; i < 4; ++i) dy2(i, 0) = (y2(i, 0) - tgt(i, 0)) * scale;

        tensor<qf> dyh = l2.backward(dy2);

        tensor<qf> dy1(shape{4, 4});
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                qf t = tanh_prime(y1(i, j));
                qf g = dyh(i, j);
                dy1(i, j) = qf(g.w * t.w, g.x * t.x, g.y * t.y, g.z * t.z);
            }
        }

        l1.backward(dy1);
        opt.step();

        if (step % 300 == 0) std::printf("step %4d  loss %.5f\n", step, loss);
    }

    tensor<qf> y1 = l1.forward(x);
    tensor<qf> h(shape{4, 4});
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j) h(i, j) = split_tanh(y1(i, j));
    tensor<qf> y2 = l2.forward(h);

    int correct = 0;
    for (int i = 0; i < 4; ++i) {
        float pred = y2(i, 0).w > 0 ? 1.0f : -1.0f;
        int expected = bits[i][0] ^ bits[i][1];
        if ((pred > 0) == (expected == 1)) ++correct;
        std::printf("%d XOR %d = %+.2f  (expected %d)\n", bits[i][0], bits[i][1],
                    y2(i, 0).w, expected);
    }
    std::printf("accuracy %d/4\n", correct);
    return correct == 4 ? 0 : 1;
}