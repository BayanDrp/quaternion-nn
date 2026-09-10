#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/nn/layers/activation.hpp"
#include "qnn/nn/layers/linear.hpp"
#include "qnn/optim/sgd.hpp"

using qnn::nn::layers::linear;
using qnn::nn::layers::split_tanh;
using qnn::nn::layers::split_tanh_prime;
using qnn::quaternion;
using qnn::shape;
using qnn::tensor;
using qf = quaternion<float>;

namespace {
constexpr std::size_t kIn = 3;    // three vectors packed as quaternions (z=0)
constexpr std::size_t kH1 = 32;
constexpr std::size_t kH2 = 32;
constexpr std::size_t kOut = 1;   // single output quaternion (the rotation)

// Rotation data (benchmarks/gen_rotation_data.py):
//   int32 n | n * 9 float32 (v1,v2,v3 rotated basis) | n * 4 float32 (unit quat)
struct rotation_set {
    bool load(const char* path) {
        std::FILE* f = std::fopen(path, "rb");
        if (!f) return false;
        std::int32_t n = 0;
        if (std::fread(&n, 4, 1, f) != 1) return false;
        const std::size_t total_n = static_cast<std::size_t>(n);
        inputs.resize(total_n * 9);
        targets.resize(total_n * 4);
        if (total_n > 0 &&
            std::fread(inputs.data(), 4, inputs.size(), f) != inputs.size()) {
            std::fclose(f);
            return false;
        }
        if (total_n > 0 &&
            std::fread(targets.data(), 4, targets.size(), f) != targets.size()) {
            std::fclose(f);
            return false;
        }
        std::fclose(f);
        return true;
    }

    std::size_t size() const { return targets.size() / 4; }

    std::vector<float> inputs;    // [9] per sample
    std::vector<float> targets;   // [4] per sample
};
}  // namespace

int main(int argc, char** argv) {
    std::size_t epochs = 30;
    std::size_t batch = 256;
    float lr = 0.025f;
    if (argc >= 2) epochs = static_cast<std::size_t>(std::atoi(argv[1]));
    if (argc >= 3) batch = static_cast<std::size_t>(std::atoi(argv[2]));
    if (argc >= 4) lr = static_cast<float>(std::atof(argv[3]));

    std::printf("loading rotation data ...\n");
    std::fflush(stdout);
    rotation_set train, test;
    if (!train.load("data/rotation/train.bin") || !test.load("data/rotation/test.bin")) {
        std::printf("missing data/rotation/{train,test}.bin "
                    "(run benchmarks/gen_rotation_data.py)\n");
        return 2;
    }

    linear<float> l1(kIn, kH1, true, 7);
    linear<float> l2(kH1, kH2, true, 7);
    linear<float> l3(kH2, kOut, true, 7);

    std::printf("mlp: %zu quats -> %zu -> %zu -> 1 quaternion\n", kIn, kH1, kH2);
    std::fflush(stdout);

    qnn::optim::sgd<float> opt(lr);
    opt.add(l1.weight(), l1.dweight());
    opt.add(l1.bias(), l1.dbias());
    opt.add(l2.weight(), l2.dweight());
    opt.add(l2.bias(), l2.dbias());
    opt.add(l3.weight(), l3.dweight());
    opt.add(l3.bias(), l3.dbias());

    const std::size_t n = train.size();
    const std::size_t n_batches = n / batch;
    std::printf("training on %zu samples (%zu batches/epoch)\n", n, n_batches);
    std::fflush(stdout);

    tensor<qf> x(shape{batch, kIn});
    tensor<qf> tgt(shape{batch, kOut});
    tensor<qf> y1(shape{batch, kH1});
    tensor<qf> a1(shape{batch, kH1});
    tensor<qf> y2(shape{batch, kH2});
    tensor<qf> a2(shape{batch, kH2});
    tensor<qf> qhat(shape{batch, kOut});
    tensor<qf> dy_out(shape{batch, kOut});
    tensor<qf> dy_a2(shape{batch, kH2});
    tensor<qf> dy_y2(shape{batch, kH2});
    tensor<qf> dy_a1(shape{batch, kH1});
    tensor<qf> dy_y1(shape{batch, kH1});

    // squared-cosine rotation loss
    //   L_i = 1 - <qhat,q>^2 / (|qhat|^2 * 4)     (|q| = 1)
    // batch loss = mean_i L_i
    const auto loss_and_grad = [&](std::size_t i) {
        const qf& o = qhat(i, 0);
        const qf& q = tgt(i, 0);
        const float c = o.w * q.w + o.x * q.x + o.y * q.y + o.z * q.z;
        const float n2 = o.w * o.w + o.x * o.x + o.y * o.y + o.z * o.z;
        const float den = n2 > 0.0f ? n2 : 1.0f;
        const float L = 1.0f - c * c / den;
        const float k = 2.0f * c;
        const float scale = k / (batch * den * den);
        // dL/dqhat = (1/batch) * (2c/den^2) * (c*qhat - q*den)
        dy_out(i, 0) = qf(scale * (c * o.w - q.w * den),
                          scale * (c * o.x - q.x * den),
                          scale * (c * o.y - q.y * den),
                          scale * (c * o.z - q.z * den));
        return L;
    };

    for (std::size_t ep = 0; ep < epochs; ++ep) {
        double loss_sum = 0;
        for (std::size_t b = 0; b < n_batches; ++b) {
            const std::size_t off = b * batch;
            for (std::size_t i = 0; i < batch; ++i) {
                const std::size_t s = off + i;
                // pack the three vectors v1,v2,v3 as quaternions (z=0)
                for (std::size_t j = 0; j < kIn; ++j) {
                    const float* v = &train.inputs[(s * 9) + j * 3];
                    x(i, j) = qf(v[0], v[1], v[2], 0.0f);
                }
                tgt(i, 0) = qf(train.targets[s * 4],
                              train.targets[s * 4 + 1],
                              train.targets[s * 4 + 2],
                              train.targets[s * 4 + 3]);
            }

            opt.zero_grad();

            y1 = l1.forward(x);                    // [N,32] quats
            for (std::size_t i = 0; i < a1.size(); ++i) a1[i] = split_tanh(y1[i]);
            y2 = l2.forward(a1);                   // [N,32]
            for (std::size_t i = 0; i < a2.size(); ++i) a2[i] = split_tanh(y2[i]);
            qhat = l3.forward(a2);                 // [N,1]

            for (std::size_t i = 0; i < batch; ++i) loss_sum += loss_and_grad(i);

            tensor<qf> g3 = l3.backward(dy_out);   // [N,32] = dL/da2
            for (std::size_t i = 0; i < dy_y2.size(); ++i)
                dy_y2[i] = g3[i] * split_tanh_prime(y2[i]);
            tensor<qf> g2 = l2.backward(dy_y2);    // [N,32] = dL/da1
            for (std::size_t i = 0; i < dy_y1.size(); ++i)
                dy_y1[i] = g2[i] * split_tanh_prime(y1[i]);
            l1.backward(dy_y1);

            opt.step();
        }
        std::printf("epoch %zu  avg loss %.5f\n", ep + 1,
                    loss_sum / static_cast<double>(n_batches * batch));
        std::fflush(stdout);

        if ((ep + 1) % 5 == 0 || ep + 1 == epochs) {
            // eval: normalize qhat, angular error in degrees
            std::vector<float> angles;
            angles.reserve(test.size());
            double sum_ang = 0;
            const std::size_t tn = test.size();
            const std::size_t tb = tn / batch;
            for (std::size_t b = 0; b < tb; ++b) {
                const std::size_t off = b * batch;
                for (std::size_t i = 0; i < batch; ++i) {
                    const std::size_t s = off + i;
                    for (std::size_t j = 0; j < kIn; ++j) {
                        const float* v = &test.inputs[(s * 9) + j * 3];
                        x(i, j) = qf(v[0], v[1], v[2], 0.0f);
                    }
                    tgt(i, 0) = qf(test.targets[s * 4],
                                   test.targets[s * 4 + 1],
                                   test.targets[s * 4 + 2],
                                   test.targets[s * 4 + 3]);
                }
                y1 = l1.forward(x);
                for (std::size_t i = 0; i < a1.size(); ++i) a1[i] = split_tanh(y1[i]);
                y2 = l2.forward(a1);
                for (std::size_t i = 0; i < a2.size(); ++i) a2[i] = split_tanh(y2[i]);
                qhat = l3.forward(a2);
                for (std::size_t i = 0; i < batch; ++i) {
                    const qf o = qhat(i, 0);
                    const float n2 = std::sqrt(o.w * o.w + o.x * o.x +
                                               o.y * o.y + o.z * o.z);
                    const qf u = n2 > 0.0f
                                     ? qf(o.w / n2, o.x / n2, o.y / n2, o.z / n2)
                                     : qf();
                    const qf& q = tgt(i, 0);
                    float dot = std::fabs(u.w * q.w + u.x * q.x + u.y * q.y +
                                          u.z * q.z);
                    if (dot > 1.0f) dot = 1.0f;
                    const float deg = 2.0f * std::acos(dot) * 180.0f /
                                      3.14159265358979323846f;
                    angles.push_back(deg);
                    sum_ang += deg;
                }
            }
            std::sort(angles.begin(), angles.end());
            const float mean = static_cast<float>(sum_ang) /
                               static_cast<float>(angles.size());
            const float median = angles[angles.size() / 2];
            std::size_t l1c = 0, l5c = 0, l15c = 0;
            for (float a : angles) {
                if (a < 1.0f) ++l1c;
                if (a < 5.0f) ++l5c;
                if (a < 15.0f) ++l15c;
            }
            const float pct = 100.0f / static_cast<float>(angles.size());
            std::printf("  eval ep %zu: mean %.3f deg  median %.3f deg  "
                        "<1deg %.2f%%  <5deg %.2f%%  <15deg %.2f%%\n",
                        ep + 1, mean, median,
                        static_cast<float>(l1c) * pct,
                        static_cast<float>(l5c) * pct,
                        static_cast<float>(l15c) * pct);
            std::fflush(stdout);
        }
    }
    return 0;
}