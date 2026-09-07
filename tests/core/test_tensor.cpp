#include <vector>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/quaternion_matrix.hpp"
#include "qnn/core/quaternion_vector.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"

#include "../test_util.hpp"

using qnn::quaternion;
using qnn::quaternion_matrix;
using qnn::quaternion_vector;
using qnn::shape;
using qnn::tensor;

int main() {
    tensor<float> t(shape{2, 3});
    CHECK(t.rank() == 2);
    CHECK(t.dim(0) == 2);
    CHECK(t.dim(1) == 3);
    CHECK(t.size() == 6);
    CHECK((t.shape() == shape{2, 3}));

    t(0, 0) = 1;
    t(0, 2) = 5;
    t(1, 1) = -3;
    CHECK(t[0] == 1);
    CHECK(t[2] == 5);
    CHECK(t[4] == -3);
    CHECK(t.at({1, 1}) == -3);
    CHECK(t.data()[4] == -3);

    t.reshape(shape{3, 2});
    CHECK((t.shape() == shape{3, 2}));
    CHECK(t(2, 0) == -3);
    CHECK(t.size() == 6);

    tensor<float> scalar(shape{});
    CHECK(scalar.size() == 1);
    scalar() = 42;
    CHECK(scalar[0] == 42);

    tensor<quaternion<float>> qt(shape{2, 2});
    qt(0, 0) = quaternion<float>(1, 0, 0, 0);
    qt(1, 1) = quaternion<float>(0, 1, 0, 0);
    CHECK(qt(0, 0) == quaternion<float>(1, 0, 0, 0));
    CHECK(qt(1, 1) == quaternion<float>(0, 1, 0, 0));
    CHECK(qt.size() == 4);

    tensor<quaternion<float>> vdata(shape{3});
    quaternion_vector<quaternion<float>> v(vdata);
    CHECK(v.size() == 3);
    v[0] = quaternion<float>(0, 1, 0, 0);
    v[2] = quaternion<float>(2, 0, -1, 0);
    CHECK(vdata[0] == quaternion<float>(0, 1, 0, 0));
    CHECK(v.at(2) == quaternion<float>(2, 0, -1, 0));
    CHECK(&v.storage() == &vdata);

    tensor<quaternion<float>> mdata(shape{2, 3});
    quaternion_matrix<quaternion<float>> m(mdata);
    CHECK(m.rows() == 2);
    CHECK(m.cols() == 3);
    m(1, 2) = quaternion<float>(1, 2, 3, 4);
    CHECK(mdata(1, 2) == quaternion<float>(1, 2, 3, 4));
    CHECK(m.at(1, 2) == quaternion<float>(1, 2, 3, 4));
    m(0, 0) = quaternion<float>::identity();
    CHECK(m.data()[0] == quaternion<float>::identity());

    DONE();
}