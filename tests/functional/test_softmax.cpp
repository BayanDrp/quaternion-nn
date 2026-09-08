#include <cmath>
#include <cstddef>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/functional/softmax.hpp"

#include "../test_util.hpp"

using qnn::quaternion;
using qnn::shape;
using qnn::tensor;
using qnn::functional::softmax;
using qnn::functional::split_softmax;
using qf = quaternion<float>;

int main() {
    tensor<qf> x(shape{2});
    x[0] = qf(1, 2, 3, 4);
    x[1] = qf(0, 0, 0, 0);

    tensor<qf> y = split_softmax(x);
    CHECK((y.shape() == shape{2}));

    const float denom = std::exp(1) + std::exp(2) + std::exp(3) + std::exp(4);
    CHECK(std::fabs(y[0].w - std::exp(1) / denom) < 1e-5f);
    CHECK(std::fabs(y[0].z - std::exp(4) / denom) < 1e-5f);

    bool sorted = y[0].w < y[0].x && y[0].x < y[0].y && y[0].y < y[0].z;
    CHECK(sorted);

    bool positive = y[0].w > 0 && y[0].x > 0 && y[0].y > 0 && y[0].z > 0;
    CHECK(positive);

    float s0 = y[0].w + y[0].x + y[0].y + y[0].z;
    CHECK(std::fabs(s0 - 1.0f) < 1e-6f);
    CHECK(std::fabs(y[1].w - 0.25f) < 1e-6f);
    float s1 = y[1].w + y[1].x + y[1].y + y[1].z;
    CHECK(std::fabs(s1 - 1.0f) < 1e-6f);

    tensor<qf> big(shape{1});
    big[0] = qf(1000, 1000, 1001, 1000);
    tensor<qf> yb = split_softmax(big);
    CHECK(std::isfinite(yb[0].w));
    float sb = yb[0].w + yb[0].x + yb[0].y + yb[0].z;
    CHECK(std::fabs(sb - 1.0f) < 1e-6f);

    tensor<float> r(shape{3});
    r[0] = 1;
    r[1] = 2;
    r[2] = 3;
    tensor<float> yr = softmax(r);
    const float sdenom = std::exp(1) + std::exp(2) + std::exp(3);
    CHECK(std::fabs(yr[0] - std::exp(1) / sdenom) < 1e-5f);
    CHECK(std::fabs(yr[2] - std::exp(3) / sdenom) < 1e-5f);
    CHECK(yr[0] < yr[1] && yr[1] < yr[2]);
    CHECK(std::fabs((yr[0] + yr[1] + yr[2]) - 1.0f) < 1e-5f);

    tensor<float> rbig(shape{2});
    rbig[0] = 1000;
    rbig[1] = 1001;
    tensor<float> yrb = softmax(rbig);
    CHECK(std::isfinite(yrb[0]));
    CHECK(std::fabs((yrb[0] + yrb[1]) - 1.0f) < 1e-6f);

    tensor<float> empty(shape{0});
    tensor<float> ye = softmax(empty);
    CHECK(ye.size() == 0);

    DONE();
}