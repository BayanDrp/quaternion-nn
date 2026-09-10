// pool2d: finite-difference input-gradient checks for max and average pooling
// (overlapping, strided and rectangular windows) plus hand-checked argmax
// routing and the average-pool 1/count gradient split.
#include <cmath>
#include <cstddef>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/functional/pool.hpp"
#include "qnn/nn/layers/pool2d.hpp"

#include "../test_gradcheck.hpp"
#include "../test_util.hpp"

using qnn::quaternion;
using qnn::shape;
using qnn::tensor;
using qf = quaternion<float>;
using qd = quaternion<double>;

static void fill(tensor<qd>& t) {
    // Distinct strictly monotone per-component values: a window can never have
    // two near-equal entries, so max-pool argmax stays stable under eps
    // perturbation (avoids FD flips at ties).
    std::size_t i = 0;
    for (std::size_t a = 0; a < t.dim(0); ++a)
        for (std::size_t b = 0; b < t.dim(1); ++b)
            for (std::size_t c = 0; c < t.dim(2); ++c)
                for (std::size_t d = 0; d < t.dim(3); ++d, ++i)
                    t(a, b, c, d) =
                        qd(0.5 * i + 3, -0.7 * i + 1, 1.3 * i - 2, -0.9 * i + 4);
}

static void fill_dy(tensor<qd>& t) {
    for (std::size_t i = 0; i < t.size(); ++i) t[i] = qd(0.7, -0.4, 0.9, 0.25);
}

// FD gradcheck of the input gradient for one (type, stride) configuration.
static bool run_case(qnn::functional::pool_type type, std::size_t c,
                     std::size_t h, std::size_t w, std::size_t ph,
                     std::size_t pw, int stride) {
    qnn::nn::layers::pool2d<double> pool(c, shape{ph, pw}, stride, type);
    tensor<qd> x(shape{2, c, h, w});
    fill(x);

    tensor<qd> y = pool.forward(x);
    tensor<qd> dy(y.shape());
    fill_dy(dy);

    tensor<qd> dx = pool.backward(dy);
    if (dx.shape() != x.shape()) return false;

    const double eps = 1e-5, tol = 1e-5;
    return g_check_input(pool, x, dy, dx, eps, tol);
}

int main() {
    // max pool, full overlap stride
    CHECK(run_case(qnn::functional::pool_type::max, 2, 6, 6, 2, 2, 2));
    // max pool, stride 1 (overlapping windows)
    CHECK(run_case(qnn::functional::pool_type::max, 1, 5, 5, 2, 2, 1));
    // average pool
    CHECK(run_case(qnn::functional::pool_type::average, 3, 6, 6, 2, 2, 2));
    // rectangular pool window
    CHECK(run_case(qnn::functional::pool_type::average, 1, 6, 4, 3, 2, 2));

    // shape + routing semantics in a tiny hand-checked case
    {
        qnn::nn::layers::pool2d<float> pool(1, shape{2, 2}, 2,
                                            qnn::functional::pool_type::max);
        tensor<qf> x(shape{1, 1, 2, 2});
        x(0, 0, 0, 0) = qf(3, 1, 0, 0);
        x(0, 0, 0, 1) = qf(2, 5, 0, 0);
        x(0, 0, 1, 0) = qf(1, 4, 0, 0);
        x(0, 0, 1, 1) = qf(0, 2, 0, 0);
        tensor<qf> y = pool.forward(x);
        // w-axis max is the (0,0) pixel, x-axis max the (0,1) pixel
        CHECK(y(0, 0, 0, 0).w == 3 && y(0, 0, 0, 0).x == 5);
        tensor<qf> dy(shape{1, 1, 1, 1});
        dy(0, 0, 0, 0) = qf(1, -1, 0, 0);
        tensor<qf> dx = pool.backward(dy);
        // dy.w routed to the w-argmax pixel, dy.x to the x-argmax pixel
        CHECK(dx(0, 0, 0, 0).w == 1 && dx(0, 0, 0, 1).x == -1);
        CHECK(dx(0, 0, 1, 0).w == 0 && dx(0, 0, 1, 1).x == 0);
    }

    // average pool divides the gradient evenly across the window
    {
        qnn::nn::layers::pool2d<float> pool(1, shape{2, 2}, 2,
                                            qnn::functional::pool_type::average);
        tensor<qf> x(shape{1, 1, 2, 2});
        for (std::size_t i = 0; i < x.size(); ++i) x[i] = qf(1, 1, 1, 1);
        tensor<qf> y = pool.forward(x);
        CHECK(y(0, 0, 0, 0).w == 1);  // (1+1+1+1)/4
        tensor<qf> dy(shape{1, 1, 1, 1});
        dy(0, 0, 0, 0) = qf(4, 4, 4, 4);
        tensor<qf> dx = pool.backward(dy);
        CHECK(dx(0, 0, 0, 0).w == 1 && dx(0, 0, 1, 1).w == 1);
        CHECK(dx(0, 0, 0, 0).z == 1);
    }

    DONE();
}