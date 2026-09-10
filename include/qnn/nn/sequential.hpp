#ifndef QNN_NN_SEQUENTIAL_HPP
#define QNN_NN_SEQUENTIAL_HPP

#include <cassert>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "qnn/core/quaternion.hpp"
#include "qnn/core/tensor.hpp"
#include "qnn/nn/module.hpp"

namespace qnn {
namespace nn {

// A module that runs its children in order: forward is a left-to-right feed,
// backward propagates dy right-to-left. parameters()/gradients() are the
// concatenated child lists (kept in the same order, so optimizer bindings
// stay aligned). Children are owned; the class works with any Module<T>.
template <typename T>
class sequential : public Module<T> {
public:
    // Construct and own a child of a layer class template, e.g.
    //   seq.add<layers::linear>(64, 10, true, 7);
    template <template <typename> class M, typename... Args>
    M<T>& add(Args&&... args) {
        std::unique_ptr<Module<T>> p(
            new M<T>(std::forward<Args>(args)...));
        modules_.push_back(std::move(p));
        return *static_cast<M<T>*>(modules_.back().get());
    }

    // Take ownership of an already-built module.
    Module<T>& push(std::unique_ptr<Module<T>> m) {
        Module<T>& r = *m;
        modules_.push_back(std::move(m));
        return r;
    }

    std::size_t size() const { return modules_.size(); }
    Module<T>& operator[](std::size_t i) { return *modules_[i]; }
    const Module<T>& operator[](std::size_t i) const { return *modules_[i]; }

    tensor<quaternion<T>> forward(const tensor<quaternion<T>>& x) override {
        tensor<quaternion<T>> y = x;
        for (auto& m : modules_) y = m->forward(y);
        return y;
    }

    tensor<quaternion<T>> backward(const tensor<quaternion<T>>& dy) override {
        tensor<quaternion<T>> g = dy;
        for (auto it = modules_.rbegin(); it != modules_.rend(); ++it) {
            g = (*it)->backward(g);
        }
        return g;
    }

    std::vector<tensor<quaternion<T>>*> parameters() override {
        std::vector<tensor<quaternion<T>>*> out;
        for (auto& m : modules_) {
            const std::vector<tensor<quaternion<T>>*> p = m->parameters();
            out.insert(out.end(), p.begin(), p.end());
        }
        return out;
    }

    std::vector<tensor<quaternion<T>>*> gradients() override {
        std::vector<tensor<quaternion<T>>*> out;
        for (auto& m : modules_) {
            const std::vector<tensor<quaternion<T>>*> g = m->gradients();
            out.insert(out.end(), g.begin(), g.end());
        }
        return out;
    }

    void zero_grad() override {
        for (auto& m : modules_) m->zero_grad();
    }

    void apply_gradients(T lr) override {
        for (auto& m : modules_) m->apply_gradients(lr);
    }

private:
    std::vector<std::unique_ptr<Module<T>>> modules_;
};

}  // namespace nn
}  // namespace qnn

#endif  // QNN_NN_SEQUENTIAL_HPP