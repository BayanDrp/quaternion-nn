#include <cmath>
#include <random>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/nn/init.hpp"

#include "../test_util.hpp"

using qnn::quaternion;
using qnn::shape;
using qnn::tensor;
using qf = quaternion<float>;

int main() {
    tensor<qf> t(shape{4, 3});

    std::mt19937 rng(42);
    qnn::nn::uniform(t, -0.5f, 0.5f, rng);

    bool in_uniform = true;
    for (std::size_t i = 0; i < t.size(); ++i) {
        const qf& q = t[i];
        if (q.w < -0.5f || q.w > 0.5f || q.x < -0.5f || q.x > 0.5f ||
            q.y < -0.5f || q.y > 0.5f || q.z < -0.5f || q.z > 0.5f)
            in_uniform = false;
    }
    CHECK(in_uniform);

    bool all_nonzero = false;
    float sum_all = 0;
    for (std::size_t i = 0; i < t.size(); ++i) {
        const qf& q = t[i];
        sum_all += q.w + q.x + q.y + q.z;
        if (q.w != 0 || q.x != 0 || q.y != 0 || q.z != 0) all_nonzero = true;
    }
    CHECK(all_nonzero);
    CHECK(std::fabs(sum_all / (t.size() * 4)) < 0.2f);

    const float fan_in = 16;
    const float fan_out = 16;
    const float bound = std::sqrt(6.0f / (fan_in + fan_out));
    qnn::nn::xavier(t, 16, 16, rng);

    bool in_xavier = true;
    float mean = 0;
    for (std::size_t i = 0; i < t.size(); ++i) {
        const qf& q = t[i];
        mean += q.w + q.x + q.y + q.z;
        if (std::fabs(q.w) > bound || std::fabs(q.x) > bound ||
            std::fabs(q.y) > bound || std::fabs(q.z) > bound)
            in_xavier = false;
    }
    CHECK(in_xavier);
    CHECK(std::fabs(mean / (t.size() * 4)) < 0.1f);
    CHECK(bound > 0.4f);
    CHECK(bound <= 0.5f);

    std::mt19937 rng2(7);
    tensor<qf> t2(shape{2, 2});
    qnn::nn::uniform(t2, 1.0f, 2.0f, rng2);
    bool in_pos = true;
    for (std::size_t i = 0; i < t2.size(); ++i) {
        const qf& q = t2[i];
        if (q.w < 1.0f || q.w > 2.0f || q.x < 1.0f || q.x > 2.0f)
            in_pos = false;
    }
    CHECK(in_pos);

    DONE();
}