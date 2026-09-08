#include <cstddef>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/shape.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/nn/layers/linear.hpp"
#include "qnn/nn/module.hpp"
#include "qnn/optim/sgd.hpp"

#include "../test_util.hpp"

using qnn::nn::layers::linear;
using qnn::quaternion;
using qnn::shape;
using qnn::tensor;
using qf = quaternion<float>;

namespace {
class MLP : public qnn::nn::Module<float> {
public:
    MLP()
        : l1(2, 3, true, 1),
          l2(3, 1, false, 2),
          h_(shape{1, 3}) {}

    tensor<qf> forward(const tensor<qf>& x) override {
        tensor<qf> z = l1.forward(x);
        h_ = z;
        return l2.forward(z);
    }

    tensor<qf> backward(const tensor<qf>& dy) override {
        tensor<qf> dyh = l2.backward(dy);
        tensor<qf> out(h_.shape());
        for (std::size_t i = 0; i < h_.size(); ++i) out[i] = dyh[i];
        return l1.backward(out);
    }

    std::vector<tensor<qf>*> parameters() override {
        std::vector<tensor<qf>*> ps = l1.parameters();
        const std::vector<tensor<qf>*> ps2 = l2.parameters();
        ps.insert(ps.end(), ps2.begin(), ps2.end());
        return ps;
    }

    std::vector<tensor<qf>*> gradients() override {
        std::vector<tensor<qf>*> gs = l1.gradients();
        const std::vector<tensor<qf>*> gs2 = l2.gradients();
        gs.insert(gs.end(), gs2.begin(), gs2.end());
        return gs;
    }

private:
    linear<float> l1;
    linear<float> l2;
    tensor<qf> h_;
};
}  // namespace

int main() {
    {
        linear<float> l(3, 2, true, 1);
        std::vector<tensor<qf>*> ps = l.parameters();
        std::vector<tensor<qf>*> gs = l.gradients();
        CHECK(ps.size() == 2);
        CHECK(gs.size() == 2);
        for (std::size_t i = 0; i < ps.size(); ++i) {
            CHECK(ps[i]->shape() == gs[i]->shape());
        }
        CHECK(ps[0] == &l.weight());
        CHECK(ps[1] == &l.bias());
        CHECK(gs[0] == &l.dweight());
        CHECK(gs[1] == &l.dbias());
    }

    {
        linear<float> l(3, 2, false, 1);
        CHECK(l.parameters().size() == 1);
        CHECK(l.gradients().size() == 1);
        CHECK(l.parameters()[0] == &l.weight());
        CHECK(l.gradients()[0] == &l.dweight());
    }

    {
        MLP model;
        qnn::nn::Module<float>& base = model;
        tensor<qf> x(shape{1, 2});
        x(0, 0) = qf(1, 0, 0, 0);
        x(0, 1) = qf(0, 1, 0, 0);
        tensor<qf> y = base.forward(x);
        CHECK((y.shape() == shape{1, 1}));

        tensor<qf> dy(shape{1, 1});
        dy(0, 0) = qf(1, 0, 0, 0);
        tensor<qf> dx = base.backward(dy);
        CHECK((dx.shape() == shape{1, 2}));

        std::vector<tensor<qf>*> ps = model.parameters();
        std::vector<tensor<qf>*> gs = model.gradients();
        CHECK(ps.size() == 3);
        CHECK(gs.size() == 3);
        for (std::size_t i = 0; i < ps.size(); ++i) {
            CHECK(ps[i]->shape() == gs[i]->shape());
        }

        tensor<qf> before[3] = {*ps[0], *ps[1], *ps[2]};

        qnn::optim::sgd<float> opt(0.5f);
        opt.add(model);
        opt.step();

        for (std::size_t i = 0; i < ps.size(); ++i) {
            bool moved = false;
            for (std::size_t j = 0; j < ps[i]->size(); ++j) {
                if ((*ps[i])[j] != before[i][j]) moved = true;
            }
            CHECK(moved);
        }

        opt.zero_grad();
        bool gz = true;
        for (std::size_t i = 0; i < gs.size() && gz; ++i) {
            for (std::size_t j = 0; j < gs[i]->size(); ++j) {
                if ((*gs[i])[j] != qf(0, 0, 0, 0)) gz = false;
            }
        }
        CHECK(gz);
    }

    DONE();
}