#ifndef QNN_CORE_QUATERNION_HPP
#define QNN_CORE_QUATERNION_HPP

#include <cmath>
#include <ostream>

namespace qnn {

template <typename T = float>
class quaternion {
public:
    T w{};
    T x{};
    T y{};
    T z{};

    quaternion() = default;
    quaternion(T w_, T x_, T y_, T z_) : w(w_), x(x_), y(y_), z(z_) {}

    static quaternion identity() { return quaternion(1, 0, 0, 0); }

    quaternion conjugate() const { return quaternion(w, -x, -y, -z); }

    T norm_squared() const { return w * w + x * x + y * y + z * z; }

    T norm() const { return static_cast<T>(std::sqrt(norm_squared())); }

    quaternion inverse() const {
        T n2 = norm_squared();
        quaternion c = conjugate();
        return quaternion(c.w / n2, c.x / n2, c.y / n2, c.z / n2);
    }

    quaternion operator+(const quaternion& o) const {
        return quaternion(w + o.w, x + o.x, y + o.y, z + o.z);
    }

    quaternion operator-(const quaternion& o) const {
        return quaternion(w - o.w, x - o.x, y - o.y, z - o.z);
    }

    quaternion operator-() const { return quaternion(-w, -x, -y, -z); }

    quaternion operator*(const quaternion& o) const {
        return quaternion(w * o.w - x * o.x - y * o.y - z * o.z,
                          w * o.x + x * o.w + y * o.z - z * o.y,
                          w * o.y - x * o.z + y * o.w + z * o.x,
                          w * o.z + x * o.y - y * o.x + z * o.w);
    }

    quaternion operator/(const quaternion& o) const { return (*this) * o.inverse(); }

    quaternion operator*(T s) const { return quaternion(w * s, x * s, y * s, z * s); }

    bool operator==(const quaternion& o) const {
        return w == o.w && x == o.x && y == o.y && z == o.z;
    }

    bool operator!=(const quaternion& o) const { return !(*this == o); }
};

template <typename T>
quaternion<T> operator*(T s, const quaternion<T>& q) {
    return q * s;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const quaternion<T>& q) {
    return os << '(' << q.w << ", " << q.x << ", " << q.y << ", " << q.z << ')';
}

}  // namespace qnn

#endif  // QNN_CORE_QUATERNION_HPP