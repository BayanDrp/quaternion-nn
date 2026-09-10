#ifndef QNN_TEST_GRADCHECK_HPP
#define QNN_TEST_GRADCHECK_HPP

#include <cmath>
#include <cstddef>
#include <vector>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/nn/module.hpp"

// Finite-difference gradient checks for any qnn::nn::Module<T>.
//
// Loss probe: L(y) = <y, dy> = sum over elements of the component-wise dot
// product with a fixed dy. For any layer, dL/dparam must equal the analytic
// gradient accumulated in module->gradients() after forward + backward.
//
// All checks are done in double to keep central-difference noise negligible.
//
// FD tuning (learned the hard way):
//   * eps ~ 1e-5..1e-4 works best for double. 1e-6 makes roundoff dominate for
//     elements whose gradient is small relative to the probe scale (e.g. a
//     conv where an = 0.003 while the probe sums to ~2e4).
//   * max-pool has a non-differentiable argmax: if two window entries tie to
//     within eps, +/-eps probes flip the routing and the FD value is garbage.
//     Feed pools inputs with strictly monotone per-component values, or test
//     max routing separately and use average pooling for model-level checks.
//   * don't judge near-zero gradients by their ratio to the FD difference;
//     use a symmetric relative criterion + an absolute roundoff floor.

template <typename T>
static T g_comp_of(const qnn::quaternion<T>& q, std::size_t c) {
    return c == 0 ? q.w : c == 1 ? q.x : c == 2 ? q.y : q.z;
}

template <typename T>
static void g_set_comp(qnn::quaternion<T>& q, std::size_t c, T v) {
    if (c == 0) q.w = v;
    else if (c == 1) q.x = v;
    else if (c == 2) q.y = v;
    else q.z = v;
}

template <typename T>
static T g_probe(const qnn::tensor<qnn::quaternion<T>>& y,
                 const qnn::tensor<qnn::quaternion<T>>& dy) {
    T s = 0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        s += y[i].w * dy[i].w + y[i].x * dy[i].x + y[i].y * dy[i].y +
             y[i].z * dy[i].z;
    }
    return s;
}

// Checks backward's input gradient dx == dL/dx by perturbing x elementwise and
// central-differenceing the probe. dx must already be produced by
// m.backward(dy) (the caller controls zero_grad/accumulation ordering).
// Pass/fail uses a symmetric relative tolerance plus an absolute roundoff
// floor so elements whose true gradient is ~0 aren't judged by their ratio.
template <typename M, typename T>
static bool g_check_input(M& m, const qnn::tensor<qnn::quaternion<T>>& x,
                          const qnn::tensor<qnn::quaternion<T>>& dy,
                          const qnn::tensor<qnn::quaternion<T>>& dx, T eps,
                          T tol, T abs_floor = T(1e-9)) {
    bool ok = true;
    for (std::size_t i = 0; i < x.size() && ok; ++i) {
        for (std::size_t c = 0; c < 4 && ok; ++c) {
            qnn::tensor<qnn::quaternion<T>> xp = x;
            qnn::tensor<qnn::quaternion<T>> xm = x;
            qnn::quaternion<T> qp = xp[i], qm = xm[i];
            g_set_comp(qp, c, g_comp_of(qp, c) + eps);
            g_set_comp(qm, c, g_comp_of(qm, c) - eps);
            xp[i] = qp;
            xm[i] = qm;
            const T fp = g_probe(m.forward(xp), dy);
            const T fm = g_probe(m.forward(xm), dy);
            const T num = (fp - fm) / (2 * eps);
            const T anal = g_comp_of(dx[i], c);
            if (std::fabs(num - anal) >
                tol * (std::fabs(anal) + std::fabs(num)) + abs_floor) ok = false;
        }
    }
    return ok;
}

// Checks all parameter gradients (module->gradients() vs module->parameters())
// by perturbing each parameter component in place, re-forwarding, and restoring.
// The analytic gradients must already be accumulated in the module.
template <typename M, typename T>
static bool g_check_params(M& m, const qnn::tensor<qnn::quaternion<T>>& x,
                           const qnn::tensor<qnn::quaternion<T>>& dy, T eps,
                           T tol) {
    const std::vector<qnn::tensor<qnn::quaternion<T>>*> ps = m.parameters();
    const std::vector<qnn::tensor<qnn::quaternion<T>>*> gs = m.gradients();
    if (ps.size() != gs.size()) return false;
    bool ok = true;
    for (std::size_t k = 0; k < ps.size() && ok; ++k) {
        qnn::tensor<qnn::quaternion<T>>& p = *ps[k];
        for (std::size_t i = 0; i < p.size() && ok; ++i) {
            for (std::size_t c = 0; c < 4 && ok; ++c) {
                const qnn::quaternion<T> save = p[i];
                qnn::quaternion<T> qp = save, qm = save;
                g_set_comp(qp, c, g_comp_of(qp, c) + eps);
                g_set_comp(qm, c, g_comp_of(qm, c) - eps);
                p[i] = qp;
                const T fp = g_probe(m.forward(x), dy);
                p[i] = qm;
                const T fm = g_probe(m.forward(x), dy);
                p[i] = save;
                const T num = (fp - fm) / (2 * eps);
                const T anal = g_comp_of((*gs[k])[i], c);
                if (std::fabs(num - anal) >
                    tol * (std::fabs(anal) + std::fabs(num)) + tol * T(1e-9))
                    ok = false;
            }
        }
    }
    return ok;
}

#endif  // QNN_TEST_GRADCHECK_HPP