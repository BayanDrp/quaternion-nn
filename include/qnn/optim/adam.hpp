#ifndef QNN_OPTIM_ADAM_HPP
#define QNN_OPTIM_ADAM_HPP

#include <cmath>
#include <cstddef>
#include <vector>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/optim/optimizer.hpp"

namespace qnn {
namespace optim {

template <typename T>
class adam : public optimizer<T> {
public:
    adam(T lr, T beta1 = T(0.9), T beta2 = T(0.999), T eps = T(1e-8))
        : optimizer<T>(lr), beta1_(beta1), beta2_(beta2), eps_(eps), t_(0) {}

    void step() override {
        ++t_;
        const T bc1 = T(1) / (T(1) - std::pow(beta1_, t_));
        const T bc2 = T(1) / (T(1) - std::pow(beta2_, t_));

        for (std::size_t k = 0; k < this->params_.size(); ++k) {
            tensor<quaternion<T>>& p = *this->params_[k].first;
            tensor<quaternion<T>>& g = *this->params_[k].second;

            if (m_.size() <= k) {
                m_.emplace_back(p.shape());
                v_.emplace_back(p.shape());
            }

            for (std::size_t i = 0; i < p.size(); ++i) {
                const quaternion<T>& gc = g[i];
                quaternion<T>& m = m_[k][i];
                quaternion<T>& v = v_[k][i];

                m = quaternion<T>(beta1_ * m.w + (T(1) - beta1_) * gc.w,
                                  beta1_ * m.x + (T(1) - beta1_) * gc.x,
                                  beta1_ * m.y + (T(1) - beta1_) * gc.y,
                                  beta1_ * m.z + (T(1) - beta1_) * gc.z);
                v = quaternion<T>(beta2_ * v.w + (T(1) - beta2_) * gc.w * gc.w,
                                  beta2_ * v.x + (T(1) - beta2_) * gc.x * gc.x,
                                  beta2_ * v.y + (T(1) - beta2_) * gc.y * gc.y,
                                  beta2_ * v.z + (T(1) - beta2_) * gc.z * gc.z);

                const T mhat_w = m.w * bc1;
                const T mhat_x = m.x * bc1;
                const T mhat_y = m.y * bc1;
                const T mhat_z = m.z * bc1;

                const T vhat_w = v.w * bc2;
                const T vhat_x = v.x * bc2;
                const T vhat_y = v.y * bc2;
                const T vhat_z = v.z * bc2;

                const T denom_w = std::sqrt(vhat_w) + eps_;
                const T denom_x = std::sqrt(vhat_x) + eps_;
                const T denom_y = std::sqrt(vhat_y) + eps_;
                const T denom_z = std::sqrt(vhat_z) + eps_;

                p[i] = quaternion<T>(p[i].w - this->lr_ * mhat_w / denom_w,
                                     p[i].x - this->lr_ * mhat_x / denom_x,
                                     p[i].y - this->lr_ * mhat_y / denom_y,
                                     p[i].z - this->lr_ * mhat_z / denom_z);
            }
        }
    }

private:
    T beta1_;
    T beta2_;
    T eps_;
    T t_;
    std::vector<tensor<quaternion<T>>> m_;
    std::vector<tensor<quaternion<T>>> v_;
};

}  // namespace optim
}  // namespace qnn

#endif  // QNN_OPTIM_ADAM_HPP