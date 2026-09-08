#ifndef QNN_DATA_MNIST_HPP
#define QNN_DATA_MNIST_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <zlib.h>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"

namespace qnn {
namespace data {

struct mnist {
    std::vector<tensor<quaternion<float>>> images;
    std::vector<std::size_t> labels;
};

inline std::vector<std::uint8_t> read_gz(const std::string& path) {
    gzFile f = gzopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("mnist: cannot open " + path);
    std::vector<std::uint8_t> out;
    std::array<char, 65536> buf{};
    int n;
    while ((n = gzread(f, buf.data(), static_cast<unsigned>(buf.size()))) > 0) {
        out.insert(out.end(), buf.begin(), buf.begin() + n);
    }
    gzclose(f);
    return out;
}

inline std::uint32_t read_be32(const std::uint8_t* p) {
    return (static_cast<std::uint32_t>(p[0]) << 24) |
           (static_cast<std::uint32_t>(p[1]) << 16) |
           (static_cast<std::uint32_t>(p[2]) << 8) |
           static_cast<std::uint32_t>(p[3]);
}

inline mnist load_mnist(const std::string& images_path, const std::string& labels_path) {
    std::vector<std::uint8_t> imgs = read_gz(images_path);
    std::vector<std::uint8_t> lbls = read_gz(labels_path);
    if (imgs.size() < 16 || lbls.size() < 8)
        throw std::runtime_error("mnist: truncated files");
    if (read_be32(imgs.data()) != 2051)
        throw std::runtime_error("mnist: bad image magic");
    if (read_be32(lbls.data()) != 2049)
        throw std::runtime_error("mnist: bad label magic");
    const std::uint32_t n_img = read_be32(imgs.data() + 4);
    const std::uint32_t rows = read_be32(imgs.data() + 8);
    const std::uint32_t cols = read_be32(imgs.data() + 12);
    const std::uint32_t n_lbl = read_be32(lbls.data() + 4);
    if (rows != 28 || cols != 28 || n_img != n_lbl || imgs.size() != 16 + std::size_t(n_img) * 784)
        throw std::runtime_error("mnist: unexpected dimensions");

    mnist out;
    out.images.reserve(n_img);
    out.labels.reserve(n_lbl);
    for (std::uint32_t n = 0; n < n_img; ++n) {
        tensor<quaternion<float>> img(::qnn::shape{14, 14});
        const std::size_t base = std::size_t(n) * 784;
        for (std::size_t r = 0; r < 14; ++r) {
            for (std::size_t c = 0; c < 14; ++c) {
                const std::size_t tl = base + (2 * r) * 28 + (2 * c);
                const std::size_t tr = base + (2 * r) * 28 + (2 * c) + 1;
                const std::size_t bl = base + (2 * r + 1) * 28 + (2 * c);
                const std::size_t br = base + (2 * r + 1) * 28 + (2 * c) + 1;
                img(r, c) = quaternion<float>(imgs[tl] / 255.0f, imgs[tr] / 255.0f,
                                              imgs[bl] / 255.0f, imgs[br] / 255.0f);
            }
        }
        out.images.push_back(img);
        out.labels.push_back(lbls[8 + n]);
    }
    return out;
}

inline std::uint32_t count_labels(const mnist& d, std::uint32_t label) {
    std::uint32_t c = 0;
    for (std::size_t i = 0; i < d.labels.size(); ++i)
        if (d.labels[i] == label) ++c;
    return c;
}

}  // namespace data
}  // namespace qnn

#endif  // QNN_DATA_MNIST_HPP