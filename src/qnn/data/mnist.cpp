#include "qnn/data/mnist.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <zlib.h>

namespace qnn {
namespace data {

namespace {

std::vector<std::uint8_t> read_gz(const std::string& path) {
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

std::uint32_t read_be32(const std::uint8_t* p) {
    return (static_cast<std::uint32_t>(p[0]) << 24) |
           (static_cast<std::uint32_t>(p[1]) << 16) |
           (static_cast<std::uint32_t>(p[2]) << 8) |
           static_cast<std::uint32_t>(p[3]);
}

}  // namespace

mnist load_mnist(const std::string& images_path, const std::string& labels_path) {
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
    out.images = tensor<quaternion<float>>(::qnn::shape{std::size_t(n_img), 14, 14});
    out.labels = tensor<std::uint8_t>(::qnn::shape{std::size_t(n_lbl)});
    for (std::uint32_t n = 0; n < n_img; ++n) {
        const std::size_t base = std::size_t(n) * 784;
        for (std::size_t r = 0; r < 14; ++r) {
            for (std::size_t c = 0; c < 14; ++c) {
                const std::size_t tl = base + (2 * r) * 28 + (2 * c);
                const std::size_t tr = base + (2 * r) * 28 + (2 * c) + 1;
                const std::size_t bl = base + (2 * r + 1) * 28 + (2 * c);
                const std::size_t br = base + (2 * r + 1) * 28 + (2 * c) + 1;
                out.images(n, r, c) = quaternion<float>(imgs[tl] / 255.0f, imgs[tr] / 255.0f,
                                                        imgs[bl] / 255.0f, imgs[br] / 255.0f);
            }
        }
        out.labels[n] = lbls[8 + n];
    }
    return out;
}

std::uint32_t count_labels(const mnist& d, std::uint32_t label) {
    std::uint32_t c = 0;
    for (std::size_t i = 0; i < d.labels.size(); ++i)
        if (d.labels[i] == label) ++c;
    return c;
}

}  // namespace data
}  // namespace qnn