#include <cmath>
#include <cstddef>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/nn/layers/conv2d.hpp"

#include "../test_util.hpp"

using qnn::quaternion;
using qnn::shape;
using qnn::tensor;
using qd = quaternion<double>;

static double comp_of(const qd& q, int comp) {
    return comp == 0 ? q.w : comp == 1 ? q.x : comp == 2 ? q.y : q.z;
}

static double probe_loss(const tensor<qd>& y, const tensor<qd>& dy) {
    double s = 0;
    for (std::size_t i = 0; i < y.size(); ++i)
        s += y[i].w * dy[i].w + y[i].x * dy[i].x + y[i].y * dy[i].y +
             y[i].z * dy[i].z;
    return s;
}

static void fill(tensor<qd>& t) {
    std::size_t i = 0;
    for (std::size_t a = 0; a < t.dim(0); ++a)
        for (std::size_t b = 0; b < t.dim(1); ++b)
            for (std::size_t c = 0; c < t.dim(2); ++c)
                for (std::size_t d = 0; d < t.dim(3); ++d, ++i)
                    t(a, b, c, d) =
                        qd(std::sin(0.3 * i), std::cos(0.5 * i + 1) * 2,
                           -0.7 * i + 1.5, 0.2 * i * i - 3);
}

int main() {
    // ---- forward shape with batch + channels; default bias on ----
    qnn::nn::layers::conv2d<double> layer(2, shape{2, 2}, 2, 0, 1);
    CHECK(layer.in_channels() == 2 && layer.out_channels() == 2);
    CHECK(layer.kernel_h() == 2 && layer.kernel_w() == 2);
    CHECK(layer.parameters().size() == 2);   // kernel + bias
    CHECK(layer.gradients().size() == 2);

    tensor<qd> x(shape{2, 2, 4, 4});
    fill(x);
    auto y = layer.forward(x);
    CHECK(y.rank() == 4);
    CHECK(y.dim(0) == 2 && y.dim(1) == 2 && y.dim(2) == 3 && y.dim(3) == 3);

    // ---- backward shape matches input ----
    tensor<qd> dy(shape{y.dim(0), y.dim(1), y.dim(2), y.dim(3)});
    for (std::size_t i = 0; i < dy.size(); ++i) dy[i] = qd(1, -1, 0.5, 0.25);
    layer.zero_grad();
    auto dx = layer.backward(dy);
    CHECK(dx.rank() == 4);
    CHECK(dx.dim(0) == 2 && dx.dim(1) == 2 && dx.dim(2) == 4 && dx.dim(3) == 4);

    // ---- finite differences: kernel gradient (double, all components) ----
    const double h = 1e-6;
    for (int comp = 0; comp < 4; ++comp) {
        for (std::size_t idx = 0; idx < layer.weight().size(); ++idx) {
            qd wp = layer.weight()[idx], wm = layer.weight()[idx];
            double& a = comp == 0 ? wp.w : comp == 1 ? wp.x : comp == 2 ? wp.y : wp.z;
            double& b = comp == 0 ? wm.w : comp == 1 ? wm.x : comp == 2 ? wm.y : wm.z;
            a += h; b -= h;
            tensor<qd> k1 = layer.weight(), k2 = layer.weight();
            k1[idx] = wp; k2[idx] = wm;
            // rebuild layer view via swap trick is heavier; instead perturb through
            // fresh layers sharing the same kernel dims and copy weights in.
            qnn::nn::layers::conv2d<double> l1(2, shape{2, 2}, 2, 0, 1);
            qnn::nn::layers::conv2d<double> l2(2, shape{2, 2}, 2, 0, 1);
            for (std::size_t i = 0; i < layer.weight().size(); ++i) {
                l1.weight()[i] = k1[i];
                l2.weight()[i] = k2[i];
            }
            for (std::size_t i = 0; i < layer.bias().size(); ++i) {
                l1.bias()[i] = layer.bias()[i];
                l2.bias()[i] = layer.bias()[i];
            }
            double num = (probe_loss(l1.forward(x), dy) - probe_loss(l2.forward(x), dy)) / (2 * h);
            CHECK(std::fabs(comp_of(layer.dweight()[idx], comp) - num) < 1e-5);
        }
    }

    // ---- finite differences: bias gradient ----
    for (int comp = 0; comp < 4; ++comp) {
        for (std::size_t idx = 0; idx < layer.bias().size(); ++idx) {
            qd bp = layer.bias()[idx], bm = layer.bias()[idx];
            double& a = comp == 0 ? bp.w : comp == 1 ? bp.x : comp == 2 ? bp.y : bp.z;
            double& b = comp == 0 ? bm.w : comp == 1 ? bm.x : comp == 2 ? bm.y : bm.z;
            a += h; b -= h;
            qnn::nn::layers::conv2d<double> l1(2, shape{2, 2}, 2, 0, 1);
            qnn::nn::layers::conv2d<double> l2(2, shape{2, 2}, 2, 0, 1);
            for (std::size_t i = 0; i < layer.weight().size(); ++i) {
                l1.weight()[i] = layer.weight()[i];
                l2.weight()[i] = layer.weight()[i];
            }
            for (std::size_t i = 0; i < layer.bias().size(); ++i) {
                l1.bias()[i] = i == idx ? bp : layer.bias()[i];
                l2.bias()[i] = i == idx ? bm : layer.bias()[i];
            }
            double num = (probe_loss(l1.forward(x), dy) - probe_loss(l2.forward(x), dy)) / (2 * h);
            CHECK(std::fabs(comp_of(layer.dbias()[idx], comp) - num) < 1e-5);
        }
    }

    // ---- no-bias: single parameter, explicit single-channel input [N,1,H,W] ----
    {
        qnn::nn::layers::conv2d<double> nb(1, shape{3, 3}, 1, 0, 1, false);
        CHECK(nb.parameters().size() == 1);
        CHECK(nb.gradients().size() == 1);
        tensor<qd> x1(shape{2, 1, 5, 5});
        for (std::size_t i = 0; i < x1.size(); ++i)
            x1[i] = qd(0.01 * i, -0.02 * i, 0, 0.03 * i);
        auto y1 = nb.forward(x1);
        CHECK(y1.rank() == 4);
        CHECK(y1.dim(0) == 2 && y1.dim(1) == 1 && y1.dim(2) == 3 && y1.dim(3) == 3);
        tensor<qd> dy1(shape{y1.dim(0), y1.dim(1), y1.dim(2), y1.dim(3)});
        for (std::size_t i = 0; i < dy1.size(); ++i) dy1[i] = qd(1, 0, 0, 0);
        nb.zero_grad();
        auto dx1 = nb.backward(dy1);
        CHECK(dx1.rank() == 4);
        CHECK(dx1.dim(0) == 2 && dx1.dim(1) == 1 && dx1.dim(2) == 5 && dx1.dim(3) == 5);
    }

    // ---- apply_gradients moves kernel and bias ----
    {
        qd w0 = layer.weight()[0];
        qd b0 = layer.bias()[0];
        layer.zero_grad();
        layer.backward(dy);
        layer.apply_gradients(0.1);
        CHECK((layer.weight()[0] - w0).norm() > 0);
        CHECK((layer.bias()[0] - b0).norm() > 0);
        CHECK(layer.dweight()[0].norm() == 0);  // gradients cleared
        CHECK(layer.dbias()[0].norm() == 0);
    }

    DONE();
}