#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/functional/flatten.hpp"
#include "qnn/loss/mse.hpp"
#include "qnn/nn/layers/activation.hpp"
#include "qnn/nn/layers/conv2d.hpp"
#include "qnn/nn/layers/linear.hpp"
#include "qnn/nn/layers/pool2d.hpp"
#include "qnn/optim/sgd.hpp"

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
constexpr std::size_t kInH = 32;
constexpr std::size_t kInW = 32;
constexpr std::size_t kC1 = 8;
constexpr std::size_t kC2 = 16;
constexpr std::size_t kP1 = 16;                   // 32 / pool(2,2)
constexpr std::size_t kP2 = 8;                    // 16 / pool(2,2)
constexpr std::size_t kDense = kC2 * kP2 * kP2;   // 1024 pixels
constexpr std::size_t kOut = 10;
constexpr double kPi = 3.14159265358979323846;

// 4D quaternion class codes: (cos t, sin t, cos 2t, sin 2t) / sqrt(2),
// t = 2*pi*k/10  -- matches benchmark/cnn_cifar_torch.py --4d.
std::vector<qf> codes() {
    std::vector<qf> c;
    c.reserve(kOut);
    for (std::size_t k = 0; k < kOut; ++k) {
        const double t = 2.0 * kPi * static_cast<double>(k) / 10.0;
        const double s = 1.0 / std::sqrt(2.0);
        c.push_back(qf(static_cast<float>(s * std::cos(t)),
                       static_cast<float>(s * std::sin(t)),
                       static_cast<float>(s * std::cos(2.0 * t)),
                       static_cast<float>(s * std::sin(2.0 * t))));
    }
    return c;
}

// CIFAR-10: one RGB image per sample, dumped by benchmarks/prep_cifar10.py:
//   int32 n | n * 3072 bytes (row-major 32x32x3 RGB) | n bytes labels
struct cifar {
    bool load(const char* path) {
        std::FILE* f = std::fopen(path, "rb");
        if (!f) return false;
        std::int32_t n = 0;
        if (std::fread(&n, 4, 1, f) != 1) return false;
        const std::size_t total_n = static_cast<std::size_t>(n);
        pixels.assign(total_n * 3072, 0);
        labels.resize(total_n);
        if (total_n > 0 &&
            std::fread(pixels.data(), 1, pixels.size(), f) != pixels.size()) {
            std::fclose(f);
            return false;
        }
        if (total_n > 0 &&
            std::fread(labels.data(), 1, labels.size(), f) != labels.size()) {
            std::fclose(f);
            return false;
        }
        std::fclose(f);
        return true;
    }

    std::size_t size() const { return labels.size(); }

    // RGB -> quaternion (w=red, x=green, y=blue, z=0), per-channel
    // normalized: (raw/255 - mean) / std  -- identical to cnn_cifar_torch.py
    qf pixel(std::size_t s, std::size_t r, std::size_t ccol) const {
        const std::size_t off = (s * 32 + r) * 32 + ccol;
        const float r0 = static_cast<float>(pixels[off * 3 + 0]) / 255.0f;
        const float g0 = static_cast<float>(pixels[off * 3 + 1]) / 255.0f;
        const float b0 = static_cast<float>(pixels[off * 3 + 2]) / 255.0f;
        const float rv = (r0 - 0.49140f) / 0.24705f;
        const float gv = (g0 - 0.48235f) / 0.24352f;
        const float bv = (b0 - 0.44653f) / 0.26159f;
        return qf(rv, gv, bv, 0.0f);
    }

    std::vector<std::uint8_t> labels;
    std::vector<std::uint8_t> pixels;
};
}  // namespace

int main(int argc, char** argv) {
    std::size_t epochs = 4;
    std::size_t batch = 64;
    float lr = 0.1f;
    std::size_t max_train = 8000;
    bool use_4d = false;
    if (argc >= 2) epochs = static_cast<std::size_t>(std::atoi(argv[1]));
    if (argc >= 3) batch = static_cast<std::size_t>(std::atoi(argv[2]));
    if (argc >= 4) lr = static_cast<float>(std::atof(argv[3]));
    if (argc >= 5) max_train = static_cast<std::size_t>(std::atoi(argv[4]));
    if (argc >= 6) use_4d = std::string(argv[5]) == "4d";

    const std::size_t head_in = use_4d ? 1 : kOut;  // 4d: single 4-vector code

    std::printf("loading cifar-10 ...\n");
    std::fflush(stdout);
    cifar train, test;
    if (!train.load("data/cifar10/train.bin") || !test.load("data/cifar10/test.bin")) {
        std::printf("missing data/cifar10/{train,test}.bin (run benchmarks/prep_cifar10.py)\n");
        return 2;
    }

    // ---- quaternion QCNN on real RGB (w,x,y = R,G,B, z=0) ----
    conv2d<float> conv1(kC1, shape{3, 3}, 1, 1, 1, true, 7);
    conv2d<float> conv2(kC2, shape{3, 3}, kC1, 1, 1, true, 7);
    pool2d<float> pool1(kC1, shape{2, 2}, 2, pool_type::max);
    pool2d<float> pool2(kC2, shape{2, 2}, 2, pool_type::max);
    linear<float> l2(kDense, head_in, true, 7);

    std::printf("conv1: in=1 out=%zu kernel=3x3 pad=1 stride=1 (RGB as quaternion)\n",
                conv1.out_channels());
    std::printf("conv2: in=%zu out=%zu kernel=3x3 pad=1 stride=1\n",
                conv2.in_channels(), conv2.out_channels());
    if (use_4d) {
        std::printf("dense: %zu -> 1 (quaternion 4D class code)\n", kDense);
    } else {
        std::printf("dense: %zu -> %zu (one-hot, evaluated on w)\n", kDense, kOut);
    }
    std::fflush(stdout);

    qnn::optim::sgd<float> opt(lr);
    opt.add(conv1.weight(), conv1.dweight());
    opt.add(conv1.bias(), conv1.dbias());
    opt.add(conv2.weight(), conv2.dweight());
    opt.add(conv2.bias(), conv2.dbias());
    opt.add(l2.weight(), l2.dweight());
    opt.add(l2.bias(), l2.dbias());

    const std::size_t n = train.size() < max_train ? train.size() : max_train;
    const std::size_t n_batches = n / batch;
    std::printf("training on %zu samples (%zu batches/epoch)\n", n, n_batches);
    std::fflush(stdout);
    const float scale = 2.0f / static_cast<float>(batch * head_in * 4);

    tensor<qf> x(shape{batch, 1, kInH, kInW});
    tensor<qf> tgt(shape{batch, head_in});
    tensor<qf> conv1_out(shape{batch, kC1, kInH, kInW});
    tensor<qf> pooled1(shape{batch, kC1, kP1, kP1});
    tensor<qf> conv2_out(shape{batch, kC2, kP1, kP1});
    tensor<qf> pooled2(shape{batch, kC2, kP2, kP2});
    tensor<qf> flat(shape{batch, kDense});
    tensor<qf> dy2(shape{batch, head_in});
    tensor<qf> dy_flat(shape{batch, kDense});
    tensor<qf> dy_pool2(shape{batch, kC2, kP2, kP2});
    tensor<qf> dx_conv2(shape{batch, kC2, kP1, kP1});
    tensor<qf> dy_pool1(shape{batch, kC1, kP1, kP1});
    tensor<qf> dx_conv1(shape{batch, kC1, kInH, kInW});

    const std::vector<qf> class_code = codes();

    struct eval_result {
        std::size_t correct;
        std::size_t correct_q4;
        std::size_t seen;
    };
    const auto eval_test = [&]() {
        eval_result r{0, 0, 0};
        const std::size_t tn = test.size();
        const std::size_t test_batches = tn / batch;
        for (std::size_t tb = 0; tb < test_batches; ++tb) {
            const std::size_t off = tb * batch;
            for (std::size_t i = 0; i < batch; ++i) {
                const std::size_t s = off + i;
                for (std::size_t rr = 0; rr < kInH; ++rr)
                    for (std::size_t cc = 0; cc < kInW; ++cc)
                        x(i, 0, rr, cc) = test.pixel(s, rr, cc);
            }
            tensor<qf> fy1 = conv1.forward(x);
            for (std::size_t i = 0; i < conv1_out.size(); ++i) conv1_out[i] = split_tanh(fy1[i]);
            pooled1 = pool1.forward(conv1_out);
            tensor<qf> fy2 = conv2.forward(pooled1);
            for (std::size_t i = 0; i < conv2_out.size(); ++i) conv2_out[i] = split_tanh(fy2[i]);
            pooled2 = pool2.forward(conv2_out);
            flat = reshape(pooled2, shape{batch, kDense});
            tensor<qf> out = l2.forward(flat);
            for (std::size_t i = 0; i < batch; ++i) {
                std::size_t best = 0;
                if (use_4d) {
                    const qf& o = out(i, 0);
                    const float n = std::sqrt(o.w * o.w + o.x * o.x + o.y * o.y +
                                              o.z * o.z);
                    const qf u = n > 0.0f
                                     ? qf(o.w / n, o.x / n, o.y / n, o.z / n)
                                     : qf();
                    std::size_t bc = 0;
                    float best_dot = -1.0f;
                    for (std::size_t c = 0; c < kOut; ++c) {
                        const qf& t = class_code[c];
                        const float dot = u.w * t.w + u.x * t.x + u.y * t.y + u.z * t.z;
                        if (dot > best_dot) { best_dot = dot; bc = c; }
                    }
                    best = bc;
                } else {
                    // metric 1: argmax over w (torch-style, ignores x,y,z)
                    for (std::size_t c = 1; c < kOut; ++c)
                        if (out(i, c).w > out(i, best).w) best = c;
                }
                if (!use_4d) {
                    // metric 2: argmin quaternion distance to the one-hot target
                    // quats { (1,0,0,0) for class, zeros otherwise }
                    std::size_t bc = 0;
                    float best_d = 1e30f;
                    for (std::size_t c = 0; c < kOut; ++c) {
                        const qf& o = out(i, c);
                        const float d = (c == test.labels[off + i])
                                            ? (o.w - 1) * (o.w - 1) + o.x * o.x +
                                                  o.y * o.y + o.z * o.z
                                            : o.w * o.w + o.x * o.x + o.y * o.y +
                                                  o.z * o.z;
                        if (d < best_d) { best_d = d; bc = c; }
                    }
                    if (bc == test.labels[off + i]) ++r.correct_q4;
                }
                if (best == test.labels[off + i]) ++r.correct;
                ++r.seen;
            }
        }
        return r;
    };

    for (std::size_t ep = 0; ep < epochs; ++ep) {
        double loss_sum = 0;
        for (std::size_t b = 0; b < n_batches; ++b) {
            const std::size_t off = b * batch;
            for (std::size_t i = 0; i < batch; ++i) {
                const std::size_t s = off + i;
                for (std::size_t r = 0; r < kInH; ++r)
                    for (std::size_t ccol = 0; ccol < kInW; ++ccol)
                        x(i, 0, r, ccol) = train.pixel(s, r, ccol);
                for (std::size_t c = 0; c < head_in; ++c)
                    tgt(i, c) = use_4d ? class_code[train.labels[s]]
                                       : ((train.labels[s] == c) ? qf(1, 0, 0, 0) : qf());
            }

            opt.zero_grad();

            tensor<qf> y1 = conv1.forward(x);                    // [N,8,32,32]
            for (std::size_t i = 0; i < conv1_out.size(); ++i)
                conv1_out[i] = split_tanh(y1[i]);
            pooled1 = pool1.forward(conv1_out);                  // [N,8,16,16]

            tensor<qf> y2 = conv2.forward(pooled1);              // [N,16,16,16]
            for (std::size_t i = 0; i < conv2_out.size(); ++i)
                conv2_out[i] = split_tanh(y2[i]);
            pooled2 = pool2.forward(conv2_out);                  // [N,16,8,8]

            flat = reshape(pooled2, shape{batch, kDense});       // [N,1024]
            tensor<qf> out = l2.forward(flat);
            loss_sum += mse(out, tgt);
            if (!std::isfinite(loss_sum)) return 3;
            for (std::size_t i = 0; i < batch; ++i)
                for (std::size_t c = 0; c < head_in; ++c)
                    dy2(i, c) = (out(i, c) - tgt(i, c)) * scale;

            tensor<qf> g = l2.backward(dy2);                     // [N,1024]
            dy_flat = reshape(g, shape{batch, kC2, kP2, kP2});
            dy_pool2 = pool2.backward(dy_flat);                  // [N,16,16,16]
            for (std::size_t i = 0; i < dx_conv2.size(); ++i)
                dy_pool2[i] = dy_pool2[i] * split_tanh_prime(y2[i]);
            dx_conv2 = conv2.backward(dy_pool2);                 // [N,8,16,16]

            dy_pool1 = pool1.backward(dx_conv2);                 // [N,8,32,32]
            for (std::size_t i = 0; i < dx_conv1.size(); ++i)
                dy_pool1[i] = dy_pool1[i] * split_tanh_prime(y1[i]);
            dx_conv1 = conv1.backward(dy_pool1);

            opt.step();
        }
        std::printf("epoch %zu  avg loss %.4f\n", ep + 1, loss_sum / n_batches);
        std::fflush(stdout);
        if ((ep + 1) % 2 == 0 || ep + 1 == epochs) {
            const eval_result r = eval_test();
            std::printf("  test after epoch %zu: %zu / %zu = %.2f%%\n",
                        ep + 1, r.correct, r.seen,
                        100.0 * static_cast<double>(r.correct) /
                            static_cast<double>(r.seen));
            std::fflush(stdout);
        }
    }

    const eval_result r = eval_test();
    std::printf("test accuracy (w argmax):    %zu / %zu = %.2f%%\n", r.correct, r.seen,
                100.0 * static_cast<double>(r.correct) / static_cast<double>(r.seen));
    if (!use_4d)
        std::printf("test accuracy (quat distance): %zu / %zu = %.2f%%\n",
                    r.correct_q4, r.seen,
                    100.0 * static_cast<double>(r.correct_q4) /
                        static_cast<double>(r.seen));
    return 0;
}