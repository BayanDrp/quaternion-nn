#include <cmath>

#include "qnn/core/quaternion.hpp"
#include "qnn/nn/layers/activation.hpp"

#include "../test_util.hpp"

using qnn::quaternion;
using qf = quaternion<float>;

int main() {
    CHECK(qnn::nn::layers::split_relu(qf(1, -2, 0, 3)) == qf(1, 0, 0, 3));
    CHECK(qnn::nn::layers::split_relu(qf(-0.5f, 0, 0.1f, -3)) == qf(0, 0, 0.1f, 0));
    CHECK(qnn::nn::layers::split_relu(qf(0, 0, 0, 0)) == qf(0, 0, 0, 0));

    qf sg = qnn::nn::layers::split_sigmoid(qf(0, 0, 0, 0));
    CHECK(std::fabs(sg.w - 0.5f) < 1e-6f);
    CHECK(sg.x == sg.w);
    CHECK(sg.y == sg.w);
    CHECK(sg.z == sg.w);

    qf sp = qnn::nn::layers::split_sigmoid(qf(2, 0, 0, 0));
    CHECK(sp.w > 0.8f);
    qf sn = qnn::nn::layers::split_sigmoid(qf(-2, 0, 0, 0));
    CHECK(sn.w < 0.2f);

    qf th = qnn::nn::layers::split_tanh(qf(1, 0, 0, 0));
    CHECK(std::fabs(th.w - 0.7616f) < 1e-3f);
    qf tz = qnn::nn::layers::split_tanh(qf(0, 0, 0, 0));
    CHECK(tz == qf(0, 0, 0, 0));
    qf tm = qnn::nn::layers::split_tanh(qf(-1, 1, -1, 1));
    CHECK(std::fabs(tm.w - (-0.7616f)) < 1e-3f);
    CHECK(tm.y == tm.w);

    DONE();
}