#include <cstddef>
#include <cstdint>
#include <string>

#include "qnn/core/quaternion.hpp"
#include "qnn/data/mnist.hpp"

#include "../test_util.hpp"

using qnn::data::load_mnist;
using qnn::quaternion;
using qnn::shape;

int main() {
    const std::string dir = "data/mnist/";
    qnn::data::mnist train = load_mnist(dir + "train-images-idx3-ubyte.gz",
                                        dir + "train-labels-idx1-ubyte.gz");
    qnn::data::mnist test = load_mnist(dir + "t10k-images-idx3-ubyte.gz",
                                       dir + "t10k-labels-idx1-ubyte.gz");

    CHECK((train.images.shape() == shape{60000, 14, 14}));
    CHECK((test.images.shape() == shape{10000, 14, 14}));
    CHECK((train.labels.shape() == shape{60000}));
    CHECK((test.labels.shape() == shape{10000}));
    CHECK(train.images.size() == 60000 * 196);
    CHECK(test.images.size() == 10000 * 196);
    CHECK(train.labels.size() == 60000);
    CHECK(test.labels.size() == 10000);

    CHECK(train.labels[0] == 5);
    CHECK(train.labels[7] == 3);
    CHECK(test.labels[0] == 7);

    CHECK(train.images(0, 0, 0) == quaternion<float>(0, 0, 0, 0));

    float sum = 0;
    for (std::size_t k = 0; k < 196; ++k) {
        const quaternion<float> q = train.images[k];
        CHECK(q.w >= 0 && q.w <= 1);
        CHECK(q.x >= 0 && q.x <= 1);
        CHECK(q.y >= 0 && q.y <= 1);
        CHECK(q.z >= 0 && q.z <= 1);
        sum += q.w + q.x + q.y + q.z;
    }
    CHECK(sum > 1);

    CHECK(qnn::data::count_labels(train, 5) == 5421);
    CHECK(qnn::data::count_labels(test, 7) == 1028);

    bool ok = true;
    for (std::size_t i = 0; i < test.labels.size(); ++i)
        if (test.labels[i] > 9) ok = false;
    CHECK(ok);

    DONE();
}