#include <cstddef>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/functional/flatten.hpp"

#include "../test_util.hpp"

using qnn::quaternion;
using qnn::shape;
using qnn::tensor;
using qf = quaternion<float>;

int main() {
    // ---- flatten keeps batch dim, collapses the rest, order preserved ----
    {
        tensor<qf> x(shape{2, 3, 4, 5});
        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] = qf((float)i, -float(i), 2.0f * i, 0.5f * i);
        auto f = qnn::functional::flatten(x);
        CHECK(f.rank() == 2);
        CHECK(f.dim(0) == 2);
        CHECK(f.dim(1) == 3 * 4 * 5);
        CHECK(f.size() == x.size());
        for (std::size_t i = 0; i < x.size(); ++i) CHECK(f[i] == x[i]);
    }

    // ---- flatten of a matrix is identity ----
    {
        tensor<qf> x(shape{4, 6});
        for (std::size_t i = 0; i < x.size(); ++i) x[i] = qf((float)i, 1, -1, 0);
        auto f = qnn::functional::flatten(x);
        CHECK(f.dim(0) == 4 && f.dim(1) == 6);
        for (std::size_t i = 0; i < x.size(); ++i) CHECK(f[i] == x[i]);
    }

    // ---- flatten of a rank-1 tensor -> [1, n] ----
    {
        tensor<qf> x(shape{7});
        for (std::size_t i = 0; i < x.size(); ++i) x[i] = qf(3.0f * i, -i, 1, 0);
        auto f = qnn::functional::flatten(x);
        CHECK(f.rank() == 2);
        CHECK(f.dim(0) == 1 && f.dim(1) == 7);
        for (std::size_t i = 0; i < x.size(); ++i) CHECK(f[i] == x[i]);
    }

    // ---- reshape round-trip ----
    {
        tensor<qf> x(shape{2, 3, 2, 3});
        for (std::size_t i = 0; i < x.size(); ++i) x[i] = qf((float)i, 0, 0, 0);
        auto r = qnn::functional::reshape(x, shape{2, 18});
        CHECK(r.dim(0) == 2 && r.dim(1) == 18);
        auto back = qnn::functional::reshape(r, shape{2, 3, 2, 3});
        CHECK(back.rank() == 4);
        CHECK(back.dim(0) == 2 && back.dim(1) == 3 && back.dim(2) == 2 &&
              back.dim(3) == 3);
        for (std::size_t i = 0; i < x.size(); ++i) CHECK(back[i] == x[i]);
    }

    // ---- reshape row-major view: element at same flat index preserved ----
    {
        tensor<qf> x(shape{2, 3, 4});
        for (std::size_t i = 0; i < x.size(); ++i) x[i] = qf((float)i, (float)i, (float)i, (float)i);
        auto f = qnn::functional::flatten(x);
        const std::size_t n = x.size() / 2;
        CHECK(f(1, n - 1) == x(1, 2, 3));
        CHECK(f(1, 0) == x(1, 0, 0));
    }

    DONE();
}