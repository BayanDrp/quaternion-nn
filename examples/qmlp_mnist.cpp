#include "qnn/qnn.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

using qf = qnn::quaternion<float>;
using qnn::shape;
using qnn::tensor;

namespace {
constexpr std::size_t kIn = 196;  // loader: 28x28 real px -> {14,14} quaternions
constexpr std::size_t kOut = 10;

qnn::nn::models::qmlp<float> make_model() {
    qnn::nn::models::qmlp<float>::Config cfg;
    cfg.in_features = kIn;
    cfg.hidden_features = {128, 64};
    cfg.out_features = kOut;
    cfg.activation = qnn::nn::layers::split_activation<float>::kind::relu;
    return qnn::nn::models::qmlp<float>(cfg);
}
}  // namespace

int main(int argc, char** argv) {
    std::size_t epochs = 8;
    std::size_t batch = 128;
    float lr = 0.1f;
    if (argc >= 2) epochs = static_cast<std::size_t>(std::atoi(argv[1]));
    if (argc >= 3) batch = static_cast<std::size_t>(std::atoi(argv[2]));
    if (argc >= 4) lr = static_cast<float>(std::atof(argv[3]));

    std::printf("loading mnist ...\n");
    std::fflush(stdout);
    qnn::data::mnist train = qnn::data::load_mnist("data/mnist/train-images-idx3-ubyte.gz",
                                                   "data/mnist/train-labels-idx1-ubyte.gz");
    qnn::data::mnist test = qnn::data::load_mnist("data/mnist/t10k-images-idx3-ubyte.gz",
                                                  "data/mnist/t10k-labels-idx1-ubyte.gz");

    qnn::nn::models::qmlp<float> net = make_model();
    std::printf("model: %zu modules\n", net.size());
    std::fflush(stdout);

    qnn::optim::sgd<float> opt(lr);
    opt.add(net);  // binds every parameter + grad of every layer

    const std::size_t n = train.labels.size();
    const std::size_t n_batches = n / batch;
    const float scale = 2.0f / static_cast<float>(batch * kOut * 4);

    tensor<qf> x(shape{batch, kIn});
    tensor<qf> tgt(shape{batch, kOut});

    for (std::size_t ep = 0; ep < epochs; ++ep) {
        double loss_sum = 0;
        for (std::size_t b = 0; b < n_batches; ++b) {
            const std::size_t off = b * batch;
            for (std::size_t i = 0; i < batch; ++i) {
                const std::size_t s = off + i;
                for (std::size_t k = 0; k < kIn; ++k) x(i, k) = train.images(s, k / 14, k % 14);
                for (std::size_t c = 0; c < kOut; ++c) {
                    tgt(i, c) = (train.labels[s] == c) ? qf(1, 0, 0, 0) : qf();
                }
            }

            opt.zero_grad();
            tensor<qf> y = net.forward(x);
            loss_sum += qnn::loss::mse(y, tgt);
            tensor<qf> dy(shape{batch, kOut});
            for (std::size_t i = 0; i < batch * kOut; ++i) {
                dy[i] = (y[i] - tgt[i]) * scale;
            }
            net.backward(dy);
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
        tensor<qf> y = net.forward(x);
        for (std::size_t i = 0; i < batch; ++i) {
            std::size_t best = 0;
            for (std::size_t c = 1; c < kOut; ++c)
                if (y(i, c).w > y(i, best).w) best = c;
            if (best == test.labels[off + i]) ++correct;
            ++seen;
        }
    }
    std::printf("test accuracy: %zu / %zu = %.2f%%\n", correct, seen,
                100.0 * static_cast<double>(correct) / static_cast<double>(seen));
    return correct * 100 >= seen * 85 ? 0 : 1;
}