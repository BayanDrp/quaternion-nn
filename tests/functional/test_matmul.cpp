#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/functional/matmul.hpp"

#include "../test_util.hpp"

using qnn::quaternion;
using qnn::shape;
using qnn::tensor;
using qnn::functional::matmul;
using qnn::functional::matvec;

template <typename T>
tensor<T> fill_matrix(std::size_t m, std::size_t n,
                      const std::vector<std::vector<T>>& rows) {
    tensor<T> t(shape{m, n});
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) t(i, j) = rows[i][j];
    return t;
}

template <typename T>
tensor<quaternion<T>> identity_matrix(std::size_t n) {
    tensor<quaternion<T>> t(shape{n, n});
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            t(i, j) = (i == j) ? quaternion<T>(1, 0, 0, 0) : quaternion<T>();
    return t;
}

int main() {
    using qf = quaternion<float>;
    const qf I(0, 1, 0, 0), J(0, 0, 1, 0), K(0, 0, 0, 1), ONE(1, 0, 0, 0);

    tensor<qf> a = fill_matrix<qf>(2, 2, {{I, J}, {K, ONE}});
    tensor<qf> b = fill_matrix<qf>(2, 2, {{K, I}, {J, ONE}});

    tensor<qf> c = matmul(a, b);
    CHECK((c.shape() == shape{2, 2}));
    CHECK(c(0, 0) == qf(-1, 0, -1, 0));
    CHECK(c(0, 1) == qf(-1, 0, 1, 0));
    CHECK(c(1, 0) == qf(-1, 0, 1, 0));
    CHECK(c(1, 1) == qf(1, 0, 1, 0));

    tensor<qf> id = identity_matrix<float>(3);
    tensor<qf> m = fill_matrix<qf>(3, 3,
        {{qf(1, 0, 2, 0), qf(0, 1, 0, 0), qf(1, 1, 1, 1)},
         {qf(0, 0, 1, 0), qf(2, 0, 0, 0), qf(0, 0, 0, 3)},
         {qf(1, 0, 0, 1), qf(0, 2, 0, 0), qf(0, 0, 0, 0)}});
    tensor<qf> mid = matmul(id, m);
    CHECK((mid.shape() == shape{3, 3}));
    CHECK(mid(0, 0) == m(0, 0));
    CHECK(mid(1, 2) == m(1, 2));
    CHECK(mid(2, 1) == m(2, 1));

    tensor<float> ar = fill_matrix<float>(2, 2, {{1, 2}, {3, 4}});
    tensor<float> br = fill_matrix<float>(2, 2, {{5, 6}, {7, 8}});
    tensor<float> cr = matmul(ar, br);
    CHECK(cr(0, 0) == 19);
    CHECK(cr(0, 1) == 22);
    CHECK(cr(1, 0) == 43);
    CHECK(cr(1, 1) == 50);

    tensor<float> v = fill_matrix<float>(1, 2, {{5, 6}});
    tensor<float> vv(shape{2});
    vv[0] = 5;
    vv[1] = 6;
    tensor<float> mvr = matvec(ar, vv);
    CHECK((mvr.shape() == shape{2}));
    CHECK(mvr[0] == 17);
    CHECK(mvr[1] == 39);

    tensor<qf> vq(shape{2});
    vq[0] = I;
    vq[1] = ONE;
    tensor<qf> mvq = matvec(a, vq);
    CHECK(mvq[0] == qf(-1, 0, 1, 0));
    CHECK(mvq[1] == qf(1, 0, 1, 0));

    DONE();
}