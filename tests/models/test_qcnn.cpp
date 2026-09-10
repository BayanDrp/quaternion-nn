#include <cstddef>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/loss/mse.hpp"
#include "qnn/nn/models/qcnn.hpp"
#include "qnn/nn/layers/conv2d.hpp"
#include "qnn/nn/layers/linear.hpp"
#include "qnn/optim/sgd.hpp"

#include "../test_gradcheck.hpp"
#include "../test_util.hpp"

using qnn::quaternion;
using qnn::shape;
using qnn::tensor;
using qnn::nn::models::qcnn;
using qf = quaternion<float>;
using qd = quaternion<double>;

static void fill(tensor<qd>& t) {
    // Distinct monotone per-component values keep max-pool argmaxs immune to
    // eps perturbation, so the FD input grad through the whole stack is exact.
    std::size_t i = 0;
    for (std::size_t a = 0; a < t.dim(0); ++a)
        for (std::size_t b = 0; b < t.dim(1); ++b)
            for (std::size_t c = 0; c < t.dim(2); ++c)
                for (std::size_t d = 0; d < t.dim(3); ++d, ++i)
                    t(a, b, c, d) =
                        qd(0.5 * i + 3, -0.7 * i + 1, 1.3 * i - 2, -0.9 * i + 4);
}

int main() {
    // ---- structure: 8x8 in, 2 conv blocks ({2,2}, 3x3, pad1, pool2), head 2 ----
    {
        typename qcnn<double>::Config cfg;
        cfg.in_channels = 1;
        cfg.height = 8;
        cfg.width = 8;
        cfg.channels = {2, 2};
        cfg.out_features = 2;
        cfg.seed = 7;
        // FD checks need a smooth pipeline: max-pool argmaxs can tie under a
        // +/- eps probe even with separated inputs once convs mix them. Max
        // routing itself is covered by test_pool2d; here we verify wiring.
        cfg.pool_type = qnn::functional::pool_type::average;
        qcnn<double> model(cfg);

        // 8 -> conv(3,pad1) 8 -> pool 4 -> conv 4 -> pool 2 -> flat 2*2*2 = 8
        CHECK(model.flat_features() == 8);
        // per block: conv + activation + pool, then flat + linear
        CHECK(model.size() == 3 * cfg.channels.size() + 2);
        CHECK(model.real_parameter_count() ==
              4 * ((2 * (1 * 3 * 3) + 2) + (2 * (2 * 3 * 3) + 2) + (8 * 2 + 2)));

        tensor<qd> x(shape{2, 1, 8, 8});
        fill(x);
        tensor<qd> y = model.forward(x);
        CHECK((y.shape() == shape{2, 2}));

        tensor<qd> dy(shape{2, 2});
        for (std::size_t i = 0; i < dy.size(); ++i) dy[i] = qd(1, -1, 0.5, 0.25);

        model.zero_grad();
        tensor<qd> dx = model.backward(dy);
        CHECK(dx.shape() == x.shape());

        // end-to-end finite differences through the whole stack
        const double eps = 1e-5, tol = 1e-4;
        CHECK(g_check_params(model, x, dy, eps, tol));
        CHECK(g_check_input(model, x, dy, dx, eps, tol));
    }

    // ---- MNIST-shaped build (14x14, {8,16}, head 10): shapes + param count ----
    {
        typename qcnn<float>::Config cfg;
        cfg.in_channels = 1;
        cfg.height = 14;
        cfg.width = 14;
        cfg.channels = {8, 16};
        cfg.out_features = 10;
        cfg.seed = 7;
        qcnn<float> model(cfg);
        CHECK(model.flat_features() == 16 * 3 * 3);

        tensor<qf> x(shape{4, 1, 14, 14});
        for (std::size_t i = 0; i < x.size(); ++i) x[i] = qf(0.001f * i, 0, 0, 0);
        tensor<qf> y = model.forward(x);
        CHECK((y.shape() == shape{4, 10}));
        tensor<qf> dy(shape{4, 10});
        for (std::size_t i = 0; i < dy.size(); ++i) dy[i] = qf(1, 0, 0, 0);
        model.zero_grad();
        tensor<qf> dx = model.backward(dy);
        CHECK(dx.shape() == x.shape());
    }

    // ---- trains: SGD bound to the whole qcnn reduces MSE on a toy image task ----
    {
        typename qcnn<float>::Config cfg;
        cfg.in_channels = 1;
        cfg.height = 8;
        cfg.width = 8;
        cfg.channels = {2, 2};
        cfg.out_features = 2;
        cfg.seed = 11;
        qcnn<float> model(cfg);
        qnn::optim::sgd<float> opt(0.05f);
        opt.add(model);

        tensor<qf> x(shape{4, 1, 8, 8});
        tensor<qf> t(shape{4, 2});
        for (std::size_t i = 0; i < 4; ++i) {
            t(i, 0) = qf(1, 0, 0, 0);
            t(i, 1) = qf(0, 0, 0, 0);
            for (std::size_t j = 0; j < 8; ++j)
                for (std::size_t c = 0; c < 8; ++c)
                    x(i, 0, j, c) = qf(((i + j + c) % 3) * 0.1f, 0, 0, 0);
        }

        float before = qnn::loss::mse(model.forward(x), t);
        for (int step = 0; step < 100; ++step) {
            opt.zero_grad();
            tensor<qf> y = model.forward(x);
            const float s = 2.0f / (4 * 2 * 4);
            tensor<qf> dy(shape{4, 2});
            for (std::size_t i = 0; i < dy.size(); ++i) dy[i] = (y[i] - t[i]) * s;
            model.backward(dy);
            opt.step();
        }
        float after = qnn::loss::mse(model.forward(x), t);
        CHECK(after < before);
    }

    DONE();
}