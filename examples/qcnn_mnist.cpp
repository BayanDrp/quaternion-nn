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
#include "qnn/nn/layers/conv2d.hpp"
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
constexpr std::size_t kChannels = 8;
constexpr std::size_t kPooled = 7;            // 14 / pool(2,2)
constexpr std::size_t kDense = kChannels * kPooled * kPooled;  // 392
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

    // ---- the conv layer ----
    conv2d<float> conv(kChannels, shape{3, 3}, 1, 1, 1, true, 7);
    std::printf("conv layer: in=%zu out=%zu kernel=%zux%zu pad=1 stride=1\n",
                conv.in_channels(), conv.out_channels(), conv.kernel_h(),
                conv.kernel_w());
    std::printf("conv params=%zu (kernel %s, bias %s)\n", conv.parameters().size(),
                conv.weight().shape().dims().size() ? "" : "",
                conv.has_bias() ? "on" : "off");
    std::fflush(stdout);

    linear<float> l2(kDense, kOut, true, 7);
    pool2d<float> pool(kChannels, shape{2, 2}, 2, pool_type::max);

    qnn::optim::sgd<float> opt(lr);
    opt.add(conv.weight(), conv.dweight());
    opt.add(conv.bias(), conv.dbias());
    opt.add(l2.weight(), l2.dweight());
    opt.add(l2.bias(), l2.dbias());

    const std::size_t n = train.labels.size() < max_train ? train.labels.size() : max_train;
    const std::size_t n_batches = n / batch;
    std::printf("training on %zu samples (%zu batches/epoch)\n", n, n_batches);
    std::fflush(stdout);
    const float scale = 2.0f / static_cast<float>(batch * kOut * 4);

    tensor<qf> x(shape{batch, kInH, kInW});
    tensor<qf> tgt(shape{batch, kOut});
    tensor<qf> conv_out(shape{batch, kChannels, kInH, kInW});
    tensor<qf> pooled(shape{batch, kChannels, kPooled, kPooled});
    tensor<qf> flat(shape{batch, kDense});
    tensor<qf> dy2(shape{batch, kOut});
    tensor<qf> dy_flat(shape{batch, kDense});
    tensor<qf> grad_conv(shape{batch, kChannels, kInH, kInW});

    for (std::size_t ep = 0; ep < epochs; ++ep) {
        double loss_sum = 0;
        for (std::size_t b = 0; b < n_batches; ++b) {
            const std::size_t off = b * batch;
            for (std::size_t i = 0; i < batch; ++i) {
                const std::size_t s = off + i;
                for (std::size_t r = 0; r < kInH; ++r)
                    for (std::size_t ccol = 0; ccol < kInW; ++ccol)
                        x(i, r, ccol) = train.images(s, r, ccol);
                for (std::size_t c = 0; c < kOut; ++c)
                    tgt(i, c) = (train.labels[s] == c) ? qf(1, 0, 0, 0) : qf();
            }

            opt.zero_grad();
            tensor<qf> y1f = conv.forward(x);                    // [N,8,14,14]
            for (std::size_t i = 0; i < conv_out.size(); ++i)
                conv_out[i] = split_tanh(y1f[i]);
            pooled = pool.forward(conv_out);                     // [N,8,7,7]
            flat = reshape(pooled, shape{batch, kDense});
            tensor<qf> y2f = l2.forward(flat);
            loss_sum += mse(y2f, tgt);
            for (std::size_t i = 0; i < batch; ++i)
                for (std::size_t c = 0; c < kOut; ++c)
                    dy2(i, c) = (y2f(i, c) - tgt(i, c)) * scale;

            tensor<qf> dy_flatf = l2.backward(dy2);              // [N,392]
            dy_flat = reshape(dy_flatf, shape{batch, kChannels, kPooled, kPooled});
            grad_conv = pool.backward(dy_flat);                  // [N,8,14,14]
            for (std::size_t i = 0; i < grad_conv.size(); ++i) {
                const qf gate = split_tanh_prime(y1f[i]);
                const qf g = grad_conv[i];
                grad_conv[i] = qf(g.w * gate.w, g.x * gate.x, g.y * gate.y, g.z * gate.z);
            }
            conv.backward(grad_conv);
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
                    x(i, r, ccol) = test.images(s, r, ccol);
        }
        tensor<qf> y1f = conv.forward(x);
        for (std::size_t i = 0; i < conv_out.size(); ++i) conv_out[i] = split_tanh(y1f[i]);
        pool_forward(conv_out, pooled);
        flat = reshape(pooled, shape{batch, kDense});
        tensor<qf> y2f = l2.forward(flat);
        for (std::size_t i = 0; i < batch; ++i) {
            std::size_t best = 0;
            for (std::size_t c = 1; c < kOut; ++c)
                if (y2f(i, c).w > y2f(i, best).w) best = c;
            if (best == test.labels[off + i]) ++correct;
            ++seen;
        }
    }
    std::printf("test accuracy: %zu / %zu = %.2f%%\n", correct, seen,
                100.0 * static_cast<double>(correct) / static_cast<double>(seen));
    return correct * 100 >= seen * 85 ? 0 : 1;
}