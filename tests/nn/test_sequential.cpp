#include <cmath>
#include <cstddef>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/loss/mse.hpp"
#include "qnn/nn/layers/flatten.hpp"
#include "qnn/nn/sequential.hpp"
#include "qnn/nn/layers/activation.hpp"
#include "qnn/nn/layers/linear.hpp"
#include "qnn/optim/sgd.hpp"

#include "../test_gradcheck.hpp"
#include "../test_util.hpp"

using qnn::quaternion;
using qnn::shape;
using qnn::tensor;
using qnn::nn::layers::flatten;
using qnn::nn::layers::linear;
using qnn::nn::layers::split_activation;
using qnn::nn::sequential;
using qd = quaternion<double>;
using qf = quaternion<float>;

int main() {
    // ---- flatten layer: [N,C,H,W] -> [N, C*H*W] and back ----
    {
        flatten<double> f;
        tensor<qd> x(shape{2, 3, 2, 2});
        for (std::size_t i = 0; i < x.size(); ++i) x[i] = qd(i, -i, 2 * i, i + 1);
        tensor<qd> y = f.forward(x);
        CHECK((y.shape() == shape{2, 12}));
        CHECK(y(0, 0).w == x(0, 0, 0, 0).w && y(1, 11).w == x(1, 2, 1, 1).w);
        CHECK(f.parameters().empty() && f.gradients().empty());
        tensor<qd> dy(shape{2, 12});
        for (std::size_t i = 0; i < dy.size(); ++i) dy[i] = qd(i + 1, 1, 1, 1);
        tensor<qd> dx = f.backward(dy);
        CHECK(dx.shape() == x.shape());
        CHECK(dx(1, 2, 1, 1).w == dy(1, 11).w);
    }

    // ---- split activation module: forward tanh, backward component gate ----
    {
        split_activation<double> act(split_activation<double>::kind::tanh);
        tensor<qd> x(shape{2, 2});
        x(0, 0) = qd(1, -1, 2, 0.5);
        x(0, 1) = qd(0, 0.1, -2, 3);
        x(1, 0) = qd(0.2, 5, 0, -0.3);
        x(1, 1) = qd(-1, 1, 1, 1);
        tensor<qd> y = act.forward(x);
        CHECK(std::fabs(y(0, 0).w - std::tanh(1.0)) < 1e-12);
        CHECK(std::fabs(y(0, 0).x - std::tanh(-1.0)) < 1e-12);
        tensor<qd> dy(shape{2, 2});
        for (std::size_t i = 0; i < dy.size(); ++i) dy[i] = qd(1, -1, 0.5, 2);
        tensor<qd> dx = act.backward(dy);
        // componentwise: dx = dy * tanh'(x)
        CHECK(std::fabs(dx(0, 0).w - 1.0 * (1 - std::tanh(1.0) * std::tanh(1.0))) < 1e-12);
        CHECK(std::fabs(dx(0, 0).x - (-1.0) * (1 - std::tanh(-1.0) * std::tanh(-1.0))) < 1e-12);
        CHECK(act.parameters().empty() && act.gradients().empty());
    }

    // ---- sequential: linear + tanh + linear, gradcheck through the chain ----
    {
        sequential<double> net;
        net.add<linear>(3, 4, true, 1);
        net.add<split_activation>(split_activation<double>::kind::tanh);
        net.add<linear>(4, 2, true, 2);

        const std::vector<tensor<qd>*> ps = net.parameters();
        const std::vector<tensor<qd>*> gs = net.gradients();
        CHECK(ps.size() == gs.size());
        CHECK(ps.size() == 4);  // l1 w,b | l2 w,b
        CHECK(net.size() == 3);

        tensor<qd> x(shape{2, 3});
        std::size_t k = 1;
        for (std::size_t i = 0; i < x.size(); ++i) {
            x[i] = qd(0.1 * k, -0.2 * k, 0.3 * k, 0.05 * k);
            ++k;
        }
        tensor<qd> y = net.forward(x);
        CHECK((y.shape() == shape{2, 2}));
        tensor<qd> dy(shape{2, 2});
        for (std::size_t i = 0; i < dy.size(); ++i) dy[i] = qd(1, -1, 0.5, 0.25);

        net.zero_grad();
        tensor<qd> dx = net.backward(dy);
        CHECK((dx.shape() == shape{2, 3}));

        const double eps = 1e-6, tol = 1e-5;
        CHECK(g_check_params(net, x, dy, eps, tol));
        CHECK(g_check_input(net, x, dy, dx, eps, tol));
    }

    // ---- sgd bound to the whole sequential reduces MSE ----
    {
        sequential<float> net;
        net.add<linear>(3, 4, true, 5);
        net.add<split_activation>(split_activation<float>::kind::tanh);
        net.add<linear>(4, 2, true, 6);

        qnn::optim::sgd<float> opt(0.01f);
        opt.add(net);

        tensor<qf> x(shape{4, 3});
        for (std::size_t i = 0; i < x.size(); ++i) x[i] = qf(i, -i, i + 1, 0);
        tensor<qf> t(shape{4, 2});
        for (std::size_t i = 0; i < 4; ++i)
            for (std::size_t j = 0; j < 2; ++j) t(i, j) = qf(j, j, 0, 0);

        float before = 0;
        tensor<qf> y0 = net.forward(x);
        before = qnn::loss::mse(y0, t);
        for (int step = 0; step < 50; ++step) {
            opt.zero_grad();
            tensor<qf> y = net.forward(x);
            const float s = 2.0f / (4 * 2 * 4);
            tensor<qf> dy(shape{4, 2});
            for (std::size_t i = 0; i < dy.size(); ++i) dy[i] = (y[i] - t[i]) * s;
            net.backward(dy);
            opt.step();
        }
        float after = qnn::loss::mse(net.forward(x), t);
        CHECK(after < before);
    }

    DONE();
}