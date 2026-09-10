#ifndef QNN_FUNCTIONAL_NORMALIZATION_HPP
#define QNN_FUNCTIONAL_NORMALIZATION_HPP

#include <cmath>
#include <cstddef>
#include <vector>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"

namespace qnn {
namespace functional {

// Split-component LayerNorm primitives. Every operation is componentwise on
// w/x/y/z (the "split" law used by split_tanh, split_relu, split_softmax).
// All statistics reduce over the LAST (feature / d_model) axis only, so one
// mean + variance is produced per (sample, position).

// Per-component mean over the last axis. Returns a flat {groups} tensor; group
// p = i / last_dim holds the quaternion (mu_w, mu_x, mu_y, mu_z) of that row.
template <typename T>
tensor<quaternion<T>> mean(const tensor<quaternion<T>>& x) {
    const std::size_t d = x.dim(x.rank() - 1);
    const std::size_t groups = x.size() / d;
    tensor<quaternion<T>> m{::qnn::shape{groups}};
    for (std::size_t p = 0; p < groups; ++p) {
        T sw{0}, sx{0}, sy{0}, sz{0};
        for (std::size_t j = 0; j < d; ++j) {
            const quaternion<T>& q = x[p * d + j];
            sw += q.w;
            sx += q.x;
            sy += q.y;
            sz += q.z;
        }
        m[p] = quaternion<T>(sw / T(d), sx / T(d), sy / T(d), sz / T(d));
    }
    return m;
}

// Per-component variance over the last axis, using a precomputed mean.
template <typename T>
tensor<quaternion<T>> variance(const tensor<quaternion<T>>& x,
                               const tensor<quaternion<T>>& mean) {
    const std::size_t d = x.dim(x.rank() - 1);
    tensor<quaternion<T>> v(mean.shape());
    const std::size_t groups = x.size() / d;
    for (std::size_t p = 0; p < groups; ++p) {
        const quaternion<T> mu = mean[p];
        T sw{0}, sx{0}, sy{0}, sz{0};
        for (std::size_t j = 0; j < d; ++j) {
            const quaternion<T>& q = x[p * d + j];
            const T dw = q.w - mu.w, dx = q.x - mu.x, dy = q.y - mu.y, dz = q.z - mu.z;
            sw += dw * dw;
            sx += dx * dx;
            sy += dy * dy;
            sz += dz * dz;
        }
        v[p] = quaternion<T>(sw / T(d), sx / T(d), sy / T(d), sz / T(d));
    }
    return v;
}

// z = (x - mean) * inv_std, componentwise; inv_std = 1 / sqrt(variance + eps).
template <typename T>
quaternion<T> normalize(const quaternion<T>& x, const quaternion<T>& mean,
                        const quaternion<T>& variance, T eps) {
    const T iw = T(1) / std::sqrt(variance.w + eps);
    const T ix = T(1) / std::sqrt(variance.x + eps);
    const T iy = T(1) / std::sqrt(variance.y + eps);
    const T iz = T(1) / std::sqrt(variance.z + eps);
    return quaternion<T>((x.w - mean.w) * iw, (x.x - mean.x) * ix,
                         (x.y - mean.y) * iy, (x.z - mean.z) * iz);
}

// y = gamma * x + beta, componentwise (NOT a Hamilton product).
template <typename T>
quaternion<T> affine(const quaternion<T>& x, const quaternion<T>& gamma,
                     const quaternion<T>& beta) {
    return quaternion<T>(gamma.w * x.w + beta.w, gamma.x * x.x + beta.x,
                         gamma.y * x.y + beta.y, gamma.z * x.z + beta.z);
}

// Cached per-position statistics needed by the backward pass.
template <typename T>
struct layer_norm_stats {
    tensor<quaternion<T>> mean;
    tensor<quaternion<T>> inv_std;
};

// Forward output: the normalized+affine activations and the cached stats.
template <typename T>
struct layer_norm_out {
    tensor<quaternion<T>> y;
    layer_norm_stats<T> stats;
};

// y(p,j) = gamma(j) . z(p,j) + beta(j),  z = (x - mean) * inv_std.
template <typename T>
layer_norm_out<T> layer_norm_forward(const tensor<quaternion<T>>& x,
                                     const tensor<quaternion<T>>& gamma,
                                     const tensor<quaternion<T>>& beta,
                                     T eps = T(1e-5)) {
    const std::size_t d = x.dim(x.rank() - 1);
    const tensor<quaternion<T>> m = mean(x);
    const tensor<quaternion<T>> v = variance(x, m);

    layer_norm_out<T> out;
    out.stats.mean = m;
    out.stats.inv_std = tensor<quaternion<T>>(m.shape());
    const std::size_t groups = x.size() / d;
    for (std::size_t p = 0; p < groups; ++p) {
        out.stats.inv_std[p] = quaternion<T>(
            T(1) / std::sqrt(v[p].w + eps), T(1) / std::sqrt(v[p].x + eps),
            T(1) / std::sqrt(v[p].y + eps), T(1) / std::sqrt(v[p].z + eps));
    }

    out.y = tensor<quaternion<T>>(x.shape());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const quaternion<T> z = normalize(x[i], m[i / d], v[i / d], eps);
        out.y[i] = affine(z, gamma[i % d], beta[i % d]);
    }
    return out;
}

// Backward gradients. Affine is componentwise, so gamma folds into dy:
//   yhat(p,j) = gamma(j) . dy(p,j)
//   dbeta_j   = sum_p dy(p,j)
//   dgamma_j  = sum_p dy(p,j) . z(p,j)
//   dx(p,j)   = s_p * ( yhat(p,j) - mean_j(yhat_p) - z(p,j) * mean_j(yhat . z)_p )
// (the mean terms must use yhat; using raw dy gives the wrong gradient)
template <typename T>
struct layer_norm_grad {
    tensor<quaternion<T>> dx;
    tensor<quaternion<T>> dgamma;
    tensor<quaternion<T>> dbeta;
};

template <typename T>
layer_norm_grad<T> layer_norm_backward(const tensor<quaternion<T>>& dy,
                                       const tensor<quaternion<T>>& x,
                                       const tensor<quaternion<T>>& gamma,
                                       const layer_norm_stats<T>& st) {
    const std::size_t d = x.dim(x.rank() - 1);
    const std::size_t groups = x.size() / d;
    const std::size_t dmodel = gamma.size();

    // Recompute the normalized z (feature index is i % d for any rank).
    tensor<quaternion<T>> z(x.shape());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const quaternion<T>& m = st.mean[i / d];
        const quaternion<T>& s = st.inv_std[i / d];
        z[i] = quaternion<T>((x[i].w - m.w) * s.w, (x[i].x - m.x) * s.x,
                             (x[i].y - m.y) * s.y, (x[i].z - m.z) * s.z);
    }

    std::vector<T> gw(dmodel, 0), gx(dmodel, 0), gy(dmodel, 0), gz(dmodel, 0);
    std::vector<T> bw(dmodel, 0), bx(dmodel, 0), by(dmodel, 0), bz(dmodel, 0);
    tensor<quaternion<T>> dx(x.shape());

    for (std::size_t p = 0; p < groups; ++p) {
        // Per-position means over the feature axis, with gamma folded into dy:
        // yhat_j = gamma_j . dy_j. The mean terms must use yhat, not dy.
        T mw{0}, mx{0}, my{0}, mz{0};
        T zw{0}, zx{0}, zy{0}, zz{0};
        for (std::size_t j = 0; j < d; ++j) {
            const quaternion<T> g = dy[p * d + j];
            const quaternion<T> gb = gamma[j];
            const quaternion<T> q = z[p * d + j];
            const T yw = g.w * gb.w, yx = g.x * gb.x, yy = g.y * gb.y, yz = g.z * gb.z;
            mw += yw;
            mx += yx;
            my += yy;
            mz += yz;
            zw += yw * q.w;
            zx += yx * q.x;
            zy += yy * q.y;
            zz += yz * q.z;
        }
        const T iwd = mw / T(d), ixd = mx / T(d), iyd = my / T(d), izd = mz / T(d);
        const T jwd = zw / T(d), jxd = zx / T(d), jyd = zy / T(d), jzd = zz / T(d);
        const quaternion<T> s = st.inv_std[p];

        for (std::size_t j = 0; j < d; ++j) {
            const quaternion<T> g = dy[p * d + j];
            const quaternion<T> gb = gamma[j];
            const quaternion<T> q = z[p * d + j];
            // yhat = gamma . dy; dx = s * (yhat - mean(yhat) - z * mean(yhat.z))
            const T yw = g.w * gb.w, yx = g.x * gb.x, yy = g.y * gb.y, yz = g.z * gb.z;
            dx[p * d + j] = quaternion<T>(
                (yw - iwd - q.w * jwd) * s.w, (yx - ixd - q.x * jxd) * s.x,
                (yy - iyd - q.y * jyd) * s.y, (yz - izd - q.z * jzd) * s.z);
            gw[j] += g.w * q.w;
            gx[j] += g.x * q.x;
            gy[j] += g.y * q.y;
            gz[j] += g.z * q.z;
            bw[j] += g.w;
            bx[j] += g.x;
            by[j] += g.y;
            bz[j] += g.z;
        }
    }

    layer_norm_grad<T> out;
    out.dx = std::move(dx);
    out.dgamma = tensor<quaternion<T>>(::qnn::shape{dmodel});
    out.dbeta = tensor<quaternion<T>>(::qnn::shape{dmodel});
    for (std::size_t j = 0; j < dmodel; ++j) {
        out.dgamma[j] = quaternion<T>(gw[j], gx[j], gy[j], gz[j]);
        out.dbeta[j] = quaternion<T>(bw[j], bx[j], by[j], bz[j]);
    }
    return out;
}

}  // namespace functional
}  // namespace qnn

#endif  // QNN_FUNCTIONAL_NORMALIZATION_HPP