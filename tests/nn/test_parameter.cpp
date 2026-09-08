#include <cstddef>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/nn/layers/linear.hpp"
#include "qnn/nn/module.hpp"
#include "qnn/nn/parameter.hpp"

#include "../test_util.hpp"

using qnn::nn::Parameter;
using qnn::nn::layers::linear;
using qnn::quaternion;
using qnn::shape;
using qnn::tensor;
using qf = quaternion<float>;

int main() {
    {
        Parameter<float> p(shape{2, 3});
        CHECK((p.value().shape() == shape{2, 3}));
        CHECK((p.grad().shape() == shape{2, 3}));
        for (std::size_t i = 0; i < p.value().size(); ++i) {
            CHECK(p.value()[i] == qf(0, 0, 0, 0));
            CHECK(p.grad()[i] == qf(0, 0, 0, 0));
        }
        p.value()(0, 0) = qf(1, 2, 3, 4);
        p.grad()(0, 1) = qf(4, 3, 2, 1);
        p.zero_grad();
        CHECK(p.value()(0, 0) == qf(1, 2, 3, 4));
        for (std::size_t i = 0; i < p.grad().size(); ++i) {
            CHECK(p.grad()[i] == qf(0, 0, 0, 0));
        }
    }

    {
        linear<float> l(3, 2, true, 1);
        tensor<qf>* wp = l.parameters()[0];
        tensor<qf>* bp = l.parameters()[1];
        tensor<qf>* wg = l.gradients()[0];
        tensor<qf>* bg = l.gradients()[1];
        CHECK(wp == &l.weight());
        CHECK(bp == &l.bias());
        CHECK(wg == &l.dweight());
        CHECK(bg == &l.dbias());
        CHECK((wp->shape() == shape{2, 3}));
        CHECK((wg->shape() == shape{2, 3}));
        CHECK((bp->shape() == shape{2}));
        CHECK((bg->shape() == shape{2}));

        tensor<qf> x(shape{1, 3});
        x(0, 0) = qf(1, 0, 0, 0);
        x(0, 1) = qf(0, 1, 0, 0);
        x(0, 2) = qf(0, 0, 1, 0);
        tensor<qf> y = l.forward(x);
        tensor<qf> dy(shape{1, 2});
        dy(0, 0) = qf(1, 0, 0, 0);
        dy(0, 1) = qf(0, 1, 0, 0);
        l.backward(dy);

        bool grads_nonzero = false;
        for (std::size_t i = 0; i < wg->size(); ++i) {
            if ((*wg)[i] != qf(0, 0, 0, 0)) grads_nonzero = true;
        }
        CHECK(grads_nonzero);

        tensor<qf> w_before = *wp;
        l.apply_gradients(0.1f);
        bool moved = false;
        for (std::size_t i = 0; i < wp->size(); ++i) {
            if ((*wp)[i] != w_before[i]) moved = true;
        }
        CHECK(moved);
        bool gz = true;
        for (std::size_t i = 0; i < wg->size(); ++i) {
            if ((*wg)[i] != qf(0, 0, 0, 0)) gz = false;
        }
        CHECK(gz);
    }

    DONE();
}