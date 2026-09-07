#include <sstream>

#include "qnn/core/shape.hpp"

#include "../test_util.hpp"

using qnn::shape;

int main() {
    shape s{2, 3, 4};
    CHECK(s.rank() == 3);
    CHECK(s.dim(0) == 2);
    CHECK(s.dim(1) == 3);
    CHECK(s.dim(2) == 4);
    CHECK(s.size() == 24);

    shape scalar{};
    CHECK(scalar.rank() == 0);
    CHECK(scalar.size() == 1);

    shape rank1{7};
    CHECK(rank1.rank() == 1);
    CHECK(rank1.size() == 7);

    CHECK((shape{1, 2} == shape{1, 2}));
    CHECK((shape{1, 2} != shape{2, 1}));
    CHECK((shape{1, 2, 3}.size() == 6));

    std::ostringstream ss;
    ss << shape{2, 3};
    CHECK(ss.str() == "[2, 3]");

    DONE();
}