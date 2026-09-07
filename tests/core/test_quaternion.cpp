#include <cstdio>
#include <sstream>

#include "qnn/core/quaternion.hpp"

#include "../test_util.hpp"

using qnn::quaternion;
using qf = quaternion<float>;

int main() {
    const qf I(0, 1, 0, 0);
    const qf J(0, 0, 1, 0);
    const qf K(0, 0, 0, 1);
    const qf ONE(1, 0, 0, 0);
    const qf NEG_ONE(-1, 0, 0, 0);

    CHECK(I * J == K);
    CHECK(J * K == I);
    CHECK(K * I == J);
    CHECK(J * I == -K);
    CHECK(K * J == -I);
    CHECK(I * K == -J);
    CHECK(I * I == NEG_ONE);
    CHECK(J * J == NEG_ONE);
    CHECK(K * K == NEG_ONE);
    CHECK((I * J) != (J * I));

    const qf A(1, 2, 3, 4);
    const qf B(5, 6, 7, 8);
    CHECK(A * B == qf(-60, 12, 30, 24));
    CHECK(B * A == qf(-60, 20, 14, 32));
    CHECK(A * B != B * A);

    CHECK(A * ONE == A);
    CHECK(ONE * A == A);
    CHECK(qf() == qf(0, 0, 0, 0));
    CHECK(qf::identity() == ONE);

    CHECK(A + A == A * 2);
    CHECK(A - A == qf(0, 0, 0, 0));
    CHECK(A * 2 == qf(2, 4, 6, 8));
    CHECK(2.0f * A == qf(2, 4, 6, 8));

    CHECK(A.conjugate() == qf(1, -2, -3, -4));
    CHECK(qf(1, 2, 2, 0).norm_squared() == 9);
    CHECK(qf(1, 2, 2, 0).norm() == 3);

    const qf Q(0, 2, 0, 0);
    CHECK(Q * Q.inverse() == ONE);
    CHECK(Q / Q == ONE);
    CHECK(A / ONE == A);

    const quaternion<double> DI(0, 1, 0, 0);
    const quaternion<double> DJ(0, 0, 1, 0);
    CHECK(DI * DJ == quaternion<double>(0, 0, 0, 1));

    std::ostringstream ss;
    ss << A;
    CHECK(!ss.str().empty());

    DONE();
}