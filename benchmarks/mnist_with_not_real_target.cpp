#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>

#include <cstdlib>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/data/mnist.hpp"
#include "qnn/loss/mse.hpp"
#include "qnn/nn/layers/activation.hpp"
#include "qnn/nn/layers/linear.hpp"
#include "qnn/optim/sgd.hpp"

using qnn::data::load_mnist;
using qnn::loss::mse;
using qnn::nn::layers::linear;
using qnn::nn::layers::split_tanh;
using qnn::quaternion;
using qnn::shape;
using qnn::tensor;
using qf = quaternion<float>;

namespace {
constexpr std::size_t kIn = 196;
constexpr std::size_t kOut = 10;

qf tanh_prime(const qf& q) {
    return qf(1 - std::tanh(q.w) * std::tanh(q.w),
              1 - std::tanh(q.x) * std::tanh(q.x),
              1 - std::tanh(q.y) * std::tanh(q.y),
              1 - std::tanh(q.z) * std::tanh(q.z));
}

float dot(const qf& a, const qf& b) {
    return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
}

std::array<qf, kOut> codes() {
    return {qf(-0.612372436f, -0.790569415f, 0.000000000f, 0.000000000f),
            qf(-0.612372436f, 0.263523138f, -0.745355992f, 0.000000000f),
            qf(-0.612372436f, 0.263523138f, 0.372677996f, -0.645497224f),
            qf(-0.612372436f, 0.263523138f, 0.372677996f, 0.645497224f),
            qf(0.408248290f, -0.527046277f, -0.745355992f, 0.000000000f),
            qf(0.408248290f, -0.527046277f, 0.372677996f, -0.645497224f),
            qf(0.408248290f, -0.527046277f, 0.372677996f, 0.645497224f),
            qf(0.408248290f, 0.527046277f, -0.372677996f, -0.645497224f),
            qf(0.408248290f, 0.527046277f, -0.372677996f, 0.645497224f),
            qf(0.408248290f, 0.527046277f, 0.745355992f, 0.000000000f)};
}

void print_codes(const std::array<qf, kOut>& cs) {
    float min_dot = 1.0f;
    for (std::size_t a = 0; a < kOut; ++a)
        for (std::size_t b = a + 1; b < kOut; ++b) {
            const float d = dot(cs[a], cs[b]);
            if (d < min_dot) min_dot = d;
        }
    std::printf("codes: 10 unit quaternions, min pairwise dot %.3f\n", min_dot);
    for (std::size_t c = 0; c < kOut; ++c)
        std::printf("  %zu: (%+.2f, %+.2f, %+.2f, %+.2f)\n", c, cs[c].w, cs[c].x, cs[c].y,
                    cs[c].z);
    std::fflush(stdout);
}
}  // namespace

int main(int argc, char** argv) {
    std::size_t epochs = 8;
    std::size_t hidden = 64;
    std::size_t batch = 128;
    float lr = 0.1f;
    std::size_t max_train = 60000;
    if (argc >= 2) epochs = static_cast<std::size_t>(std::atoi(argv[1]));
    if (argc >= 3) hidden = static_cast<std::size_t>(std::atoi(argv[2]));
    if (argc >= 4) batch = static_cast<std::size_t>(std::atoi(argv[3]));
    if (argc >= 5) lr = static_cast<float>(std::atof(argv[4]));
    if (argc >= 6) max_train = static_cast<std::size_t>(std::atoi(argv[5]));

    const std::array<qf, kOut> cs = codes();
    print_codes(cs);

    std::printf("loading mnist ...\n");
    std::fflush(stdout);
    qnn::data::mnist train = load_mnist("data/mnist/train-images-idx3-ubyte.gz",
                                        "data/mnist/train-labels-idx1-ubyte.gz");
    qnn::data::mnist test = load_mnist("data/mnist/t10k-images-idx3-ubyte.gz",
                                       "data/mnist/t10k-labels-idx1-ubyte.gz");

    linear<float> l1(kIn, hidden, true, 7);
    linear<float> l2(hidden, kOut, true, 7);

    qnn::optim::sgd<float> opt(lr);
    opt.add(l1.weight(), l1.dweight());
    opt.add(l1.bias(), l1.dbias());
    opt.add(l2.weight(), l2.dweight());
    opt.add(l2.bias(), l2.dbias());

    const std::size_t n = train.labels.size() < max_train ? train.labels.size() : max_train;
    const std::size_t n_batches = n / batch;
    std::printf("training on %zu samples (%zu batches/epoch), full-quat targets + dot decode\n", n,
                n_batches);
    std::fflush(stdout);
    const float scale = 2.0f / static_cast<float>(batch * kOut * 4);

    tensor<qf> x(shape{batch, kIn});
    tensor<qf> tgt(shape{batch, kOut});
    tensor<qf> y1(shape{batch, hidden});
    tensor<qf> h(shape{batch, hidden});
    tensor<qf> y2(shape{batch, kOut});
    tensor<qf> dy2(shape{batch, kOut});
    tensor<qf> dyh(shape{batch, hidden});
    tensor<qf> dy1(shape{batch, hidden});

    for (std::size_t ep = 0; ep < epochs; ++ep) {
        double loss_sum = 0;
        for (std::size_t b = 0; b < n_batches; ++b) {
            const std::size_t off = b * batch;
            for (std::size_t i = 0; i < batch; ++i) {
                const std::size_t s = off + i;
                for (std::size_t k = 0; k < kIn; ++k) x(i, k) = train.images(s, k / 14, k % 14);
                for (std::size_t c = 0; c < kOut; ++c) {
                    tgt(i, c) = (train.labels[s] == c) ? cs[c] : qf();
                }
            }

            opt.zero_grad();
            tensor<qf> y1f = l1.forward(x);
            for (std::size_t i = 0; i < batch; ++i)
                for (std::size_t k = 0; k < hidden; ++k) h(i, k) = split_tanh(y1f(i, k));
            tensor<qf> y2f = l2.forward(h);
            loss_sum += mse(y2f, tgt);
            for (std::size_t i = 0; i < batch; ++i)
                for (std::size_t c = 0; c < kOut; ++c) dy2(i, c) = (y2f(i, c) - tgt(i, c)) * scale;
            tensor<qf> dyhf = l2.backward(dy2);
            for (std::size_t i = 0; i < batch; ++i) {
                for (std::size_t k = 0; k < hidden; ++k) {
                    const qf gate = tanh_prime(y1f(i, k));
                    const qf g = dyhf(i, k);
                    dy1(i, k) = qf(g.w * gate.w, g.x * gate.x, g.y * gate.y, g.z * gate.z);
                }
            }
            l1.backward(dy1);
            opt.step();
        }
        std::printf("epoch %zu  avg loss %.4f\n", ep + 1, loss_sum / n_batches);
        std::fflush(stdout);
    }

    std::size_t correct = 0;
    std::size_t seen = 0;
    const std::size_t tn = test.labels.size();
    const std::size_t test_batches = tn / batch;
    for (std::size_t b = 0; b < test_batches; ++b) {
        const std::size_t off = b * batch;
        for (std::size_t i = 0; i < batch; ++i) {
            const std::size_t s = off + i;
            for (std::size_t k = 0; k < kIn; ++k) x(i, k) = test.images(s, k / 14, k % 14);
        }
        tensor<qf> y1f = l1.forward(x);
        for (std::size_t i = 0; i < batch; ++i)
            for (std::size_t k = 0; k < hidden; ++k) h(i, k) = split_tanh(y1f(i, k));
        tensor<qf> y2f = l2.forward(h);
        for (std::size_t i = 0; i < batch; ++i) {
            std::size_t best = 0;
            float best_score = dot(cs[0], y2f(i, 0));
            for (std::size_t c = 1; c < kOut; ++c) {
                const float score = dot(cs[c], y2f(i, c));
                if (score > best_score) {
                    best_score = score;
                    best = c;
                }
            }
            if (best == test.labels[off + i]) ++correct;
            ++seen;
        }
    }
    std::printf("test accuracy: %zu / %zu = %.2f%%\n", correct, seen,
                100.0 * static_cast<double>(correct) / static_cast<double>(seen));
    std::printf("(real-only baseline: 86.67%%; torch T4: 86.43%%)\n");
    return correct * 100 >= seen * 85 ? 0 : 1;
}