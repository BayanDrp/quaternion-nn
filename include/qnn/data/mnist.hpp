#ifndef QNN_DATA_MNIST_HPP
#define QNN_DATA_MNIST_HPP

#include <cstdint>
#include <string>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"

namespace qnn {
namespace data {

struct mnist {
    tensor<quaternion<float>> images;
    tensor<std::uint8_t> labels;
};

mnist load_mnist(const std::string& images_path, const std::string& labels_path);

std::uint32_t count_labels(const mnist& d, std::uint32_t label);

}  // namespace data
}  // namespace qnn

#endif  // QNN_DATA_MNIST_HPP