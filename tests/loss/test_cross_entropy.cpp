#include <cmath>
#include <cstddef>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/functional/softmax.hpp"
#include "qnn/loss/cross_entropy.hpp"

#include "../test_util.hpp"

using qnn::functional::split_softmax;
using qnn::loss::cross_entropy;
using qnn::loss::cross_entropy_grad;
using qnn::quaternion;
using qnn::shape;
using qnn::tensor;
using qf = quaternion<float>;

int main() {
    tensor<qf> t(shape{2});
    t[0] = qf(1, 0, 0, 0);
    t[1] = qf(0, 1, 0, 0);

    tensor<qf> p_perfect = t;
    CHECK(std::fabs(cross_entropy(t, p_perfect)) < 1e-4f);

    tensor<qf> p_uniform(shape{2});
    p_uniform[0] = qf(0.25f, 0.25f, 0.25f, 0.25f);
    p_uniform[1] = qf(0.25f, 0.25f, 0.25f, 0.25f);
    CHECK(std::fabs(cross_entropy(t, p_uniform) - std::log(4.0f)) < 1e-5f);

    tensor<qf> g0 = cross_entropy_grad(p_perfect, t);
    CHECK(g0[0] == qf(0, 0, 0, 0));
    CHECK(g0[1] == qf(0, 0, 0, 0));

    tensor<qf> lw(shape{2});
    lw[0] = qf(0.4f, 0.2f, 0.2f, 0.2f);
    lw[1] = qf(0.25f, 0.3f, 0.2f, 0.25f);
    tensor<qf> g = cross_entropy_grad(lw, t);
    CHECK(g[0].w < 0);
    CHECK(g[0].x > 0);
    CHECK(g[1].x < 0);
    CHECK(g[1].w > 0);
    float s0 = g[0].w + g[0].x + g[0].y + g[0].z;
    float s1 = g[1].w + g[1].x + g[1].y + g[1].z;
    CHECK(std::fabs(s0) < 1e-6f);
    CHECK(std::fabs(s1) < 1e-6f);

    tensor<qf> logits(shape{2});
    logits[0] = qf(1, 2, 3, 4);
    logits[1] = qf(4, 3, 2, 1);
    tensor<qf> probs = split_softmax(logits);
    tensor<qf> t_end(shape{2});
    t_end[0] = qf(0, 0, 0, 1);
    t_end[1] = qf(1, 0, 0, 0);
    float ce = cross_entropy(t_end, probs);
    CHECK(ce > 0);
    CHECK(ce < std::log(4.0f));
    CHECK(std::fabs(ce - 0.4403f) < 2e-2f);

    const quaternion<double> dp(0.25, 0.25, 0.25, 0.25);
    tensor<quaternion<double>> dt(shape{1});
    dt[0] = quaternion<double>(1, 0, 0, 0);
    tensor<quaternion<double>> dpred(shape{1});
    dpred[0] = dp;
    double dce = cross_entropy(dt, dpred);
    CHECK(std::fabs(dce - std::log(4.0)) < 1e-6);

    DONE();
}