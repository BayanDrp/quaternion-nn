#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/data/mnist.hpp"
#include "qnn/functional/flatten.hpp"
#include "qnn/loss/mse.hpp"
#include "qnn/nn/layers/activation.hpp"
#include "qnn/nn/layers/conv2d.hpp"
#include "qnn/nn/layers/linear.hpp"
#include "qnn/nn/layers/pool2d.hpp"
#include "qnn/optim/sgd.hpp"

using qnn::data::load_mnist;
using qnn::functional::pool_type;
using qnn::functional::reshape;
using qnn::loss::mse;
using qnn::nn::layers::conv2d;
using qnn::nn::layers::linear;
using qnn::nn::layers::pool2d;
using qnn::nn::layers::split_tanh;
using qnn::nn::layers::split_tanh_prime;
using qnn::quaternion;
using qnn::shape;
using qnn::tensor;
using qf = quaternion<float>;

namespace {
constexpr std::size_t kInH = 14;
constexpr std::size_t kInW = 14;
constexpr std::size_t kC1 = 8;
constexpr std::size_t kC2 = 16;
constexpr std::size_t kP1 = 7;              // 14 / pool(2,2)
constexpr std::size_t kP2 = 3;              // 7 / pool(2,2)
constexpr std::size_t kDense = kC2 * kP2 * kP2;  // 144
constexpr std::size_t kOut = 10;
}  // namespace

int main(int argc, char** argv) {
    std::size_t epochs = 4;
    std::size_t batch = 64;
    float lr = 0.1f;
    std::size_t max_train = 60000;
    if (argc >= 2) epochs = static_cast<std::size_t>(std::atoi(argv[1]));
    if (argc >= 3) batch = static_cast<std::size_t>(std::atoi(argv[2]));
    if (argc >= 4) lr = static_cast<float>(std::atof(argv[3]));
    if (argc >= 5) max_train = static_cast<std::size_t>(std::atoi(argv[4]));

    std::printf("loading mnist ...\n");
    std::fflush(stdout);
    qnn::data::mnist train = load_mnist("data/mnist/train-images-idx3-ubyte.gz",
                                        "data/mnist/train-labels-idx1-ubyte.gz");
    qnn::data::mnist test = load_mnist("data/mnist/t10k-images-idx3-ubyte.gz",
                                       "data/mnist/t10k-labels-idx1-ubyte.gz");

    // ---- quaternion QCNN (mirrors benchmarks/cnn_torch.py) ----
    conv2d<float> conv1(kC1, shape{3, 3}, 1, 1, 1, true, 7);
    conv2d<float> conv2(kC2, shape{3, 3}, kC1, 1, 1, true, 7);
    pool2d<float> pool1(kC1, shape{2, 2}, 2, pool_type::max);
    pool2d<float> pool2(kC2, shape{2, 2}, 2, pool_type::max);
    linear<float> l2(kDense, kOut, true, 7);

    std::printf("conv1: in=%zu out=%zu kernel=3x3 pad=1 stride=1\n",
                conv1.in_channels(), conv1.out_channels());
    std::printf("conv2: in=%zu out=%zu kernel=3x3 pad=1 stride=1\n",
                conv2.in_channels(), conv2.out_channels());
    std::printf("dense: %zu -> %zu\n", kDense, kOut);
    std::fflush(stdout);

    qnn::optim::sgd<float> opt(lr);
    opt.add(conv1.weight(), conv1.dweight());
    opt.add(conv1.bias(), conv1.dbias());
    opt.add(conv2.weight(), conv2.dweight());
    opt.add(conv2.bias(), conv2.dbias());
    opt.add(l2.weight(), l2.dweight());
    opt.add(l2.bias(), l2.dbias());

    const std::size_t n = train.labels.size() < max_train ? train.labels.size() : max_train;
    const std::size_t n_batches = n / batch;
    std::printf("training on %zu samples (%zu batches/epoch)\n", n, n_batches);
    std::fflush(stdout);
    const float scale = 2.0f / static_cast<float>(batch * kOut * 4);

    tensor<qf> x(shape{batch, 1, kInH, kInW});
    tensor<qf> tgt(shape{batch, kOut});
    tensor<qf> conv1_out(shape{batch, kC1, kInH, kInW});
    tensor<qf> pooled1(shape{batch, kC1, kP1, kP1});
    tensor<qf> conv2_out(shape{batch, kC2, kP1, kP1});
    tensor<qf> pooled2(shape{batch, kC2, kP2, kP2});
    tensor<qf> flat(shape{batch, kDense});
    tensor<qf> dy2(shape{batch, kOut});
    tensor<qf> dy_flat(shape{batch, kDense});
    tensor<qf> dy_pool2(shape{batch, kC2, kP2, kP2});
    tensor<qf> dx_conv2(shape{batch, kC2, kP1, kP1});
    tensor<qf> dy_pool1(shape{batch, kC1, kP1, kP1});
    tensor<qf> dx_conv1(shape{batch, kC1, kInH, kInW});

    for (std::size_t ep = 0; ep < epochs; ++ep) {
        double loss_sum = 0;
        for (std::size_t b = 0; b < n_batches; ++b) {
            const std::size_t off = b * batch;
            for (std::size_t i = 0; i < batch; ++i) {
                const std::size_t s = off + i;
                for (std::size_t r = 0; r < kInH; ++r)
                    for (std::size_t ccol = 0; ccol < kInW; ++ccol)
                        x(i, 0, r, ccol) = train.images(s, r, ccol);
                for (std::size_t c = 0; c < kOut; ++c)
                    tgt(i, c) = (train.labels[s] == c) ? qf(1, 0, 0, 0) : qf();
            }

            opt.zero_grad();

            tensor<qf> y1 = conv1.forward(x);                    // [N,8,14,14]
            for (std::size_t i = 0; i < conv1_out.size(); ++i)
                conv1_out[i] = split_tanh(y1[i]);
            pooled1 = pool1.forward(conv1_out);                  // [N,8,7,7]

            tensor<qf> y2 = conv2.forward(pooled1);              // [N,16,7,7]
            for (std::size_t i = 0; i < conv2_out.size(); ++i)
                conv2_out[i] = split_tanh(y2[i]);
            pooled2 = pool2.forward(conv2_out);                  // [N,16,3,3]

            flat = reshape(pooled2, shape{batch, kDense});       // [N,144]
            tensor<qf> out = l2.forward(flat);
            loss_sum += mse(out, tgt);
            for (std::size_t i = 0; i < batch; ++i)
                for (std::size_t c = 0; c < kOut; ++c)
                    dy2(i, c) = (out(i, c) - tgt(i, c)) * scale;

            tensor<qf> g = l2.backward(dy2);                     // [N,144]
            dy_flat = reshape(g, shape{batch, kC2, kP2, kP2});
            dy_pool2 = pool2.backward(dy_flat);                  // [N,16,7,7]
            for (std::size_t i = 0; i < dx_conv2.size(); ++i)
                dy_pool2[i] = dy_pool2[i] * split_tanh_prime(y2[i]);
            dx_conv2 = conv2.backward(dy_pool2);                 // [N,8,7,7]

            dy_pool1 = pool1.backward(dx_conv2);                 // [N,8,14,14]
            for (std::size_t i = 0; i < dx_conv1.size(); ++i)
                dy_pool1[i] = dy_pool1[i] * split_tanh_prime(y1[i]);
            dx_conv1 = conv1.backward(dy_pool1);

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
            for (std::size_t r = 0; r < kInH; ++r)
                for (std::size_t ccol = 0; ccol < kInW; ++ccol)
                    x(i, 0, r, ccol) = test.images(s, r, ccol);
        }
        tensor<qf> y1 = conv1.forward(x);
        for (std::size_t i = 0; i < conv1_out.size(); ++i) conv1_out[i] = split_tanh(y1[i]);
        pooled1 = pool1.forward(conv1_out);
        tensor<qf> y2 = conv2.forward(pooled1);
        for (std::size_t i = 0; i < conv2_out.size(); ++i) conv2_out[i] = split_tanh(y2[i]);
        pooled2 = pool2.forward(conv2_out);
        flat = reshape(pooled2, shape{batch, kDense});
        tensor<qf> out = l2.forward(flat);
        for (std::size_t i = 0; i < batch; ++i) {
            std::size_t best = 0;
            for (std::size_t c = 1; c < kOut; ++c)
                if (out(i, c).w > out(i, best).w) best = c;
            if (best == test.labels[off + i]) ++correct;
            ++seen;
        }
    }
    std::printf("test accuracy: %zu / %zu = %.2f%%\n", correct, seen,
                100.0 * static_cast<double>(correct) / static_cast<double>(seen));
    return correct * 100 >= seen * 85 ? 0 : 1;
}