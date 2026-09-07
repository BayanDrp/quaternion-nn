#include <cmath>
#include <cstddef>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/loss/mse.hpp"
#include "qnn/nn/layers/linear.hpp"

#include "../test_util.hpp"

using qnn::loss::mse;
using qnn::nn::layers::linear;
using qnn::quaternion;
using qnn::shape;
using qnn::tensor;
using qf = quaternion<float>;

static float loss_at(const linear<float>& src, const tensor<qf>& x, const tensor<qf>& t) {
    linear<float> copy = src;
    tensor<qf> y = copy.forward(x);
    return mse(y, t);
}

static bool approx(float a, float b, float tol, float scale = 1.0f) {
    return std::fabs(a - b) <= tol * (std::fabs(b) + scale);
}

int main() {
    linear<float> l(3, 2, true, 1);
    CHECK(l.in_features() == 3);
    CHECK(l.out_features() == 2);
    CHECK(l.has_bias());

    tensor<qf> x(shape{2, 3});
    x(0, 0) = qf(1, 0, 0, 0);
    x(0, 1) = qf(0, 1, 0, 0);
    x(0, 2) = qf(0, 0, -1, 0);
    x(1, 0) = qf(0, 0, 1, 0);
    x(1, 1) = qf(1, 1, 0, 0);
    x(1, 2) = qf(0, 0, 0, -2);

    tensor<qf> t(shape{2, 2});
    t(0, 0) = qf(1, 0, 0, 0);
    t(0, 1) = qf(0, 0, 1, 0);
    t(1, 0) = qf(0, 0, 0, 1);
    t(1, 1) = qf(1, 0, 0, 0);

    tensor<qf> y = l.forward(x);
    CHECK((y.shape() == shape{2, 2}));
    CHECK((l.weight().shape() == shape{2, 3}));

    const float scale = 2.0f / (2 * 2 * 4.0f);
    tensor<qf> dy(shape{2, 2});
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j) dy(i, j) = (y(i, j) - t(i, j)) * scale;

    tensor<qf> dx = l.backward(dy);

    const float eps = 1e-2f;
    const float tol = 2e-2f;

    bool w_ok = true;
    for (std::size_t j = 0; j < 2 && w_ok; ++j) {
        for (std::size_t k = 0; k < 3; ++k) {
            for (int c = 0; c < 4; ++c) {
                linear<float> plus = l;
                qf qp = plus.weight()(j, k);
                if (c == 0) qp.w += eps;
                if (c == 1) qp.x += eps;
                if (c == 2) qp.y += eps;
                if (c == 3) qp.z += eps;
                plus.weight()(j, k) = qp;

                linear<float> minus = l;
                qf qm = minus.weight()(j, k);
                if (c == 0) qm.w -= eps;
                if (c == 1) qm.x -= eps;
                if (c == 2) qm.y -= eps;
                if (c == 3) qm.z -= eps;
                minus.weight()(j, k) = qm;

                float numer = (loss_at(plus, x, t) - loss_at(minus, x, t)) / (2 * eps);

                qf gw = l.dweight()(j, k);
                float anal = (c == 0) ? gw.w : (c == 1) ? gw.x : (c == 2) ? gw.y : gw.z;
                if (!approx(numer, anal, tol)) w_ok = false;
            }
        }
    }
    CHECK(w_ok);

    bool b_ok = true;
    for (std::size_t j = 0; j < 2 && b_ok; ++j) {
        for (int c = 0; c < 4; ++c) {
            linear<float> plus = l;
            qf qp = plus.bias()(j);
            if (c == 0) qp.w += eps;
            if (c == 1) qp.x += eps;
            if (c == 2) qp.y += eps;
            if (c == 3) qp.z += eps;
            plus.bias()(j) = qp;

            linear<float> minus = l;
            qf qm = minus.bias()(j);
            if (c == 0) qm.w -= eps;
            if (c == 1) qm.x -= eps;
            if (c == 2) qm.y -= eps;
            if (c == 3) qm.z -= eps;
            minus.bias()(j) = qm;

            float numer = (loss_at(plus, x, t) - loss_at(minus, x, t)) / (2 * eps);

            qf gb = l.dbias()(j);
            float anal = (c == 0) ? gb.w : (c == 1) ? gb.x : (c == 2) ? gb.y : gb.z;
            if (!approx(numer, anal, tol)) b_ok = false;
        }
    }
    CHECK(b_ok);

    bool x_ok = true;
    for (std::size_t i = 0; i < 2 && x_ok; ++i) {
        for (std::size_t k = 0; k < 3; ++k) {
            for (int c = 0; c < 4; ++c) {
                tensor<qf> xp = x;
                qf qp = xp(i, k);
                if (c == 0) qp.w += eps;
                if (c == 1) qp.x += eps;
                if (c == 2) qp.y += eps;
                if (c == 3) qp.z += eps;
                xp(i, k) = qp;

                tensor<qf> xm = x;
                qf qm = xm(i, k);
                if (c == 0) qm.w -= eps;
                if (c == 1) qm.x -= eps;
                if (c == 2) qm.y -= eps;
                if (c == 3) qm.z -= eps;
                xm(i, k) = qm;

                float numer = (loss_at(l, xp, t) - loss_at(l, xm, t)) / (2 * eps);

                qf gx = dx(i, k);
                float anal = (c == 0) ? gx.w : (c == 1) ? gx.x : (c == 2) ? gx.y : gx.z;
                if (!approx(numer, anal, tol)) x_ok = false;
            }
        }
    }
    CHECK(x_ok);

    linear<float> l2(2, 2, false, 3);
    tensor<qf> x2(shape{1, 2});
    x2(0, 0) = qf(0, 1, 0, 0);
    x2(0, 1) = qf(1, 0, 1, 0);
    tensor<qf> y2 = l2.forward(x2);
    CHECK((y2.shape() == shape{1, 2}));
    CHECK(!l2.has_bias());
    tensor<qf> dy2(shape{1, 2});
    dy2(0, 0) = qf(1, 0, 0, 0);
    dy2(0, 1) = qf(0, 1, 0, 0);
    tensor<qf> dx2 = l2.backward(dy2);
    CHECK((dx2.shape() == shape{1, 2}));
    bool db_zero = true;
    for (std::size_t i = 0; i < l2.dbias().size(); ++i)
        if (l2.dbias()[i] != qf(0, 0, 0, 0)) db_zero = false;
    CHECK(db_zero);

    linear<float> l3 = l;
    float before = loss_at(l3, x, t);
    l3.backward(dy);
    l3.apply_gradients(0.05f);
    float after = loss_at(l3, x, t);
    CHECK(after < before);

    DONE();
}