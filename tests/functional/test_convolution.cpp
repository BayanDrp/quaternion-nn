#include <cmath>
#include <cstddef>
#include <utility>

#include "qnn/functional/convolution.hpp"
#include "qnn/functional/pool.hpp"
#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"

#include "../test_util.hpp"

using qnn::functional::backward_conv2d;
using qnn::functional::backward_pool2d;
using qnn::functional::conv2d;
using qnn::functional::pool2d;
using qnn::functional::pool_type;
using qnn::quaternion;
using qnn::shape;
using qnn::tensor;
using qf = quaternion<float>;
using qd = quaternion<double>;

static bool approx(const qf& a, const qf& b, float tol = 1e-4f) {
    return std::fabs(a.w - b.w) < tol && std::fabs(a.x - b.x) < tol &&
           std::fabs(a.y - b.y) < tol && std::fabs(a.z - b.z) < tol;
}

// Scalar loss assembled from a fixed probe dy (component-wise linear + quaternion-free).
static double probe_loss(const tensor<qd>& y, const tensor<qd>& dy) {
    double s = 0;
    for (std::size_t i = 0; i < y.size(); ++i)
        s += y[i].w * dy[i].w + y[i].x * dy[i].x + y[i].y * dy[i].y + y[i].z * dy[i].z;
    return s;
}

int main() {
    // ---- add_padding ----
    {
        tensor<qf> in(shape{1, 2, 2});
        in(0, 0, 0) = qf(1, 0, 0, 0);
        in(0, 0, 1) = qf(2, 0, 0, 0);
        in(0, 1, 0) = qf(3, 0, 0, 0);
        in(0, 1, 1) = qf(4, 0, 0, 0);
        auto p = qnn::functional::add_padding(in, 1);
        CHECK(p.rank() == 3);
        CHECK(p.dim(0) == 1 && p.dim(1) == 4 && p.dim(2) == 4);
        CHECK(p(0, 0, 0) == qf(0, 0, 0, 0));
        CHECK(p(0, 1, 1) == qf(1, 0, 0, 0));
        CHECK(p(0, 2, 2) == qf(4, 0, 0, 0));
    }

    // ---- conv2d: single channel golden (kernel taps = real, i, j, k) ----
    {
        // Hamilton: i*(a,b,c,d)=(-b,a,-d,c), j*(a,b,c,d)=(-c,d,a,-b),
        //            k*(a,b,c,d)=(-d,-c,b,a)
        tensor<qf> x(shape{1, 3, 3});
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) x(0, r, c) = qf(r * 3 + c + 1, 0, 0, 0);
        tensor<qf> k(shape{1, 1, 2, 2});
        k(0, 0, 0, 0) = qf(1, 0, 0, 0);
        k(0, 0, 0, 1) = qf(0, 1, 0, 0);
        k(0, 0, 1, 0) = qf(0, 0, 1, 0);
        k(0, 0, 1, 1) = qf(0, 0, 0, 1);
        auto y = conv2d(x, k, 0, 1);
        CHECK(y.rank() == 3);
        CHECK(y.dim(0) == 1 && y.dim(1) == 2 && y.dim(2) == 2);
        CHECK(approx(y(0, 0, 0), qf(1, 2, 4, 5)));
        CHECK(approx(y(0, 0, 1), qf(2, 3, 5, 6)));
        CHECK(approx(y(0, 1, 0), qf(4, 5, 7, 8)));
        CHECK(approx(y(0, 1, 1), qf(5, 6, 8, 9)));
    }

    // ---- conv2d: multi-channel input/output, non-square (1x1) kernel ----
    {
        // v(r,c) = r*3 + c + 1, both input channels identical; every kernel tap real 1
        tensor<qf> x2(shape{1, 2, 3, 3});
        for (int ci = 0; ci < 2; ++ci)
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) x2(0, ci, r, c) = qf(r * 3 + c + 1, 0, 0, 0);
        tensor<qf> k2(shape{2, 2, 1, 1});  // 2 out, 2 in, 1x1 tap
        for (int co = 0; co < 2; ++co)
            for (int ci = 0; ci < 2; ++ci) k2(co, ci, 0, 0) = qf(1, 0, 0, 0);
        auto y = conv2d(x2, k2, 0, 1);
        CHECK(y.dim(0) == 2 && y.dim(1) == 3 && y.dim(2) == 3);
        CHECK(approx(y(0, 0, 0), qf(2, 0, 0, 0)));
        CHECK(approx(y(0, 1, 1), qf(10, 0, 0, 0)));
        CHECK(approx(y(0, 2, 2), qf(18, 0, 0, 0)));
        CHECK(approx(y(1, 0, 2), qf(6, 0, 0, 0)));   // identical out channels
    }

    // ---- conv2d: padding shape ----
    {
        tensor<qf> x3(shape{1, 1, 3, 3});
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) x3(0, 0, r, c) = qf(r * 3 + c + 1, 0, 0, 0);
        tensor<qf> k3(shape{2, 1, 2, 2});  // all-real summing taps
        for (int co = 0; co < 2; ++co)
            for (int r = 0; r < 2; ++r)
                for (int c = 0; c < 2; ++c) k3(co, 0, r, c) = qf(1, 0, 0, 0);
        auto y = conv2d(x3, k3, 1, 1);  // 3x3 + pad 1 -> 4x4
        CHECK(y.dim(0) == 2 && y.dim(1) == 4 && y.dim(2) == 4);
        CHECK(approx(y(0, 0, 0), qf(1, 0, 0, 0)));    // single corner pixel
        CHECK(approx(y(0, 0, 1), qf(3, 0, 0, 0)));    // x(0,0)+x(0,1)
        CHECK(y(1, 2, 2) == y(0, 2, 2));              // identical kernels
    }

    // ---- pool2d: max (split) and average ----
    {
        tensor<qf> in(shape{1, 2, 2});
        in(0, 0, 0) = qf(1, 5, 2, -4);
        in(0, 0, 1) = qf(3, -2, -4, 7);
        in(0, 1, 0) = qf(-2, 4, 6, 1);
        in(0, 1, 1) = qf(0, 1, -3, 0);
        auto m = pool2d(in, pool_type::max, 2, 2, 1);
        CHECK(approx(m(0, 0, 0), qf(3, 5, 6, 7), 1e-4f));
        auto a = pool2d(in, pool_type::average, 2, 2, 1);
        CHECK(approx(a(0, 0, 0), qf(0.5f, 2, 0.25f, 1), 1e-4f));
    }

    // ---- pool2d: negative-only window (max must not default to zero) ----
    {
        tensor<qf> in(shape{1, 2, 2});
        in(0, 0, 0) = qf(-1, -2, -3, -4);
        in(0, 0, 1) = qf(-5, -6, -7, -8);
        in(0, 1, 0) = qf(-9, -10, -11, -12);
        in(0, 1, 1) = qf(-13, -14, -15, -16);
        auto m = pool2d(in, pool_type::max, 2, 2, 1);
        CHECK(m(0, 0, 0) == qf(-1, -2, -3, -4));
    }

    // ---- backward_conv2d: finite differences (double) ----
    {
        tensor<qd> x(shape{1, 1, 4, 4});
        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] = qd(0.1 * i, 0.2 * i + 1, 0.3 * i - 2, 0.05 * i + 3);
        tensor<qd> k(shape{1, 1, 2, 2});
        k[0] = qd(1, 2, -1, 0.5);
        k[1] = qd(-0.5, 1, 3, 2);
        k[2] = qd(2, -1, 0.5, 1);
        k[3] = qd(0.5, 0.5, -2, -1);

        auto y = conv2d(x, k, 0, 2);
        tensor<qd> dy(shape{y.dim(0), y.dim(1), y.dim(2)});
        for (std::size_t i = 0; i < y.size(); ++i) dy[i] = qd(1, 1, 1, 1);

        auto [gx, gk] = backward_conv2d(x, k, dy, 0, 2);

        auto num_grad_x = [&](std::size_t r, std::size_t c, int comp, double h) {
            qd xp = x(0, 0, r, c), xm = x(0, 0, r, c);
            double& a = comp == 0 ? xp.w : comp == 1 ? xp.x : comp == 2 ? xp.y : xp.z;
            double& b = comp == 0 ? xm.w : comp == 1 ? xm.x : comp == 2 ? xm.y : xm.z;
            a += h; b -= h;
            tensor<qd> x1 = x, x2 = x;
            x1(0, 0, r, c) = xp; x2(0, 0, r, c) = xm;
            return (probe_loss(conv2d(x1, k, 0, 2), dy) -
                    probe_loss(conv2d(x2, k, 0, 2), dy)) / (2 * h);
        };
        auto num_grad_k = [&](std::size_t idx, int comp, double h) {
            qd kp = k[idx], km = k[idx];
            double& a = comp == 0 ? kp.w : comp == 1 ? kp.x : comp == 2 ? kp.y : kp.z;
            double& b = comp == 0 ? km.w : comp == 1 ? km.x : comp == 2 ? km.y : km.z;
            a += h; b -= h;
            tensor<qd> k1 = k, k2 = k;
            k1[idx] = kp; k2[idx] = km;
            return (probe_loss(conv2d(x, k1, 0, 2), dy) -
                    probe_loss(conv2d(x, k2, 0, 2), dy)) / (2 * h);
        };
        auto comp_of = [](const qd& q, int comp) {
            return comp == 0 ? q.w : comp == 1 ? q.x : comp == 2 ? q.y : q.z;
        };

        const double h = 1e-6;
        for (int comp = 0; comp < 4; ++comp) {
            for (std::size_t idx = 0; idx < 4; ++idx) {
                CHECK(std::fabs(comp_of(gk[idx], comp) - num_grad_k(idx, comp, h)) < 1e-5);
            }
            for (std::size_t r = 0; r < 2; ++r) {
                for (std::size_t c = 0; c < 2; ++c) {
                    CHECK(std::fabs(comp_of(gx(0, 0, r, c), comp) - num_grad_x(r, c, comp, h)) < 1e-5);
                }
            }
        }
    }

    // ---- backward_pool2d: max routing + average overlap (finite differences) ----
    {
        tensor<qd> x(shape{2, 5, 4});
        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] = qd(std::sin(0.3 * i), std::cos(0.5 * i + 1) * 2, -0.7 * i + 1.5, 0.2 * i * i - 3);

        const int ph = 3, pw = 2, stride = 2;
        const pool_type types[2] = {pool_type::max, pool_type::average};
        for (int t = 0; t < 2; ++t) {
            auto y = pool2d(x, types[t], ph, pw, stride);
            tensor<qd> dy(shape{y.dim(0), y.dim(1), y.dim(2)});
            for (std::size_t i = 0; i < y.size(); ++i) dy[i] = qd(i % 3, (i + 1) % 5, (i + 2) % 4, (i + 3) % 2);
            auto g = backward_pool2d(x, dy, types[t], ph, pw, stride);
            CHECK(g.dim(0) == 2 && g.dim(1) == 5 && g.dim(2) == 4);

            const double h = 1e-6;
            for (std::size_t cidx = 0; cidx < 2; ++cidx)
                for (std::size_t r = 0; r < 5; ++r)
                    for (std::size_t cc = 0; cc < 4; ++cc)
                        for (int comp = 0; comp < 4; ++comp) {
                            qd xp = x(cidx, r, cc), xm = x(cidx, r, cc);
                            double& a = comp == 0 ? xp.w : comp == 1 ? xp.x
                                                : comp == 2 ? xp.y : xp.z;
                            double& b = comp == 0 ? xm.w : comp == 1 ? xm.x
                                                : comp == 2 ? xm.y : xm.z;
                            a += h; b -= h;
                            tensor<qd> x1 = x, x2 = x;
                            x1(cidx, r, cc) = xp; x2(cidx, r, cc) = xm;
                            double num = (probe_loss(pool2d(x1, types[t], ph, pw, stride), dy) -
                                          probe_loss(pool2d(x2, types[t], ph, pw, stride), dy)) /
                                         (2 * h);
                            const qd& gg = g(cidx, r, cc);
                            double ana = comp == 0 ? gg.w : comp == 1 ? gg.x
                                                   : comp == 2 ? gg.y : gg.z;
                            CHECK(std::fabs(num - ana) < 1e-5);
                        }
        }
    }

    DONE();
}