#include <cmath>
#include <cstddef>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/nn/layers/conv2d.hpp"

#include "../test_gradcheck.hpp"
#include "../test_util.hpp"

using qnn::quaternion;
using qnn::shape;
using qnn::tensor;
using qd = quaternion<double>;

// Deterministic [N, C, H, W] fill with distinct values per element/component.
static void fill(tensor<qd>& t) {
    std::size_t i = 0;
    for (std::size_t a = 0; a < t.dim(0); ++a)
        for (std::size_t b = 0; b < t.dim(1); ++b)
            for (std::size_t c = 0; c < t.dim(2); ++c)
                for (std::size_t d = 0; d < t.dim(3); ++d, ++i)
                    t(a, b, c, d) =
                        qd(std::sin(0.3 * i), std::cos(0.5 * i + 1) * 2,
                           -0.7 * i + 1.5, 0.2 * i * i - 3);
}

static void fill_dy(tensor<qd>& t) {
    for (std::size_t i = 0; i < t.size(); ++i) t[i] = qd(1, -1, 0.5, 0.25);
}

static bool run_case(std::size_t in_c, std::size_t out_c, std::size_t h,
                     std::size_t w, std::size_t kh, std::size_t kw, int pad,
                     int stride, bool bias) {
    qnn::nn::layers::conv2d<double> conv(out_c, shape{kh, kw}, in_c, pad, stride,
                                         bias, 3);
    tensor<qd> x(shape{2, in_c, h, w});
    fill(x);

    tensor<qd> y = conv.forward(x);
    tensor<qd> dy(y.shape());
    fill_dy(dy);

    conv.zero_grad();
    tensor<qd> dx = conv.backward(dy);
    if (dx.shape() != x.shape()) return false;

    // eps fine enough to keep truncation error tiny (conv is linear, so central
// diff is exact), large enough to keep FD roundoff below tol even where the
// input gradient is small relative to the probe scale.
    const double eps = 1e-4, tol = 1e-5;
    if (!g_check_input(conv, x, dy, dx, eps, tol)) return false;
    return g_check_params(conv, x, dy, eps, tol);
}

int main() {
    // multi-channel, padding, same spatial size
    CHECK(run_case(2, 2, 4, 4, 2, 2, 1, 1, true));
    // stride 2 without padding
    CHECK(run_case(1, 2, 5, 5, 2, 2, 0, 2, true));
    // odd kernel, no bias
    CHECK(run_case(1, 2, 4, 4, 3, 3, 0, 1, false));
    // padding that enlarges output
    CHECK(run_case(1, 1, 4, 4, 3, 3, 2, 1, true));
    // rectangular kernel + stride, no bias
    CHECK(run_case(2, 1, 6, 6, 3, 2, 0, 2, false));

    DONE();
}