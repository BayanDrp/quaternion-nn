#include <cmath>
#include <cstddef>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/optim/adam.hpp"
#include "qnn/optim/sgd.hpp"

#include "../test_util.hpp"

using qnn::optim::adam;
using qnn::optim::sgd;
using qnn::quaternion;
using qnn::shape;
using qnn::tensor;
using qf = quaternion<float>;

int main() {
    tensor<qf> p(shape{2});
    p[0] = qf(8, 4, 2, 8);
    p[1] = qf(0, 0, 0, 0);
    tensor<qf> g(shape{2});
    g[0] = qf(2, 1, -1, -1);
    g[1] = qf(0, 0, 0, 0);

    sgd<float> opt(0.5f);
    opt.add(p, g);

    opt.step();
    CHECK(p[0] == qf(7, 3.5f, 2.5f, 8.5f));
    CHECK(p[1] == qf(0, 0, 0, 0));

    opt.step();
    CHECK(p[0] == qf(6, 3, 3, 9));

    opt.zero_grad();
    CHECK(g[0] == qf(0, 0, 0, 0));
    CHECK(g[1] == qf(0, 0, 0, 0));

    opt.step();
    CHECK(p[0] == qf(6, 3, 3, 9));

    tensor<qf> pa(shape{1});
    pa[0] = qf(1, 1, 1, 1);
    tensor<qf> ga(shape{1});
    ga[0] = qf(1, 0.5f, 0.1f, 0);

    adam<float> ad(0.1f);
    ad.add(pa, ga);

    float prev = 1.0f;
    bool monotonic = true;
    for (int i = 0; i < 50; ++i) {
        ad.step();
        if (pa[0].w >= prev) monotonic = false;
        prev = pa[0].w;
    }
    CHECK(monotonic);
    CHECK(pa[0].w < 0.5f);
    CHECK(pa[0].w > -5.0f);
    CHECK(pa[0].z == 1.0f);
    CHECK(std::isfinite(pa[0].x));
    CHECK(std::isfinite(pa[0].y));

    tensor<qf> pb(shape{2});
    pb[0] = qf(0, 0, 0, 0);
    pb[1] = qf(2, 0, -1, 0);
    tensor<qf> gb(shape{2});
    gb[0] = qf(0, 0, 0, 0);
    gb[1] = qf(0.5f, 0, 0, 0);

    sgd<float> opt2(0.1f);
    opt2.add(pb, gb);
    opt2.step();
    CHECK(pb[0] == qf(0, 0, 0, 0));
    CHECK((pb[1].w < 2.0f));

    DONE();
}