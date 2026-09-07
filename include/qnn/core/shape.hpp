#ifndef QNN_CORE_SHAPE_HPP
#define QNN_CORE_SHAPE_HPP

#include <cstddef>
#include <initializer_list>
#include <ostream>
#include <vector>

namespace qnn {

class shape {
public:
    shape() = default;
    shape(std::initializer_list<std::size_t> dims) : dims_(dims) {}

    std::size_t rank() const { return dims_.size(); }
    std::size_t dim(std::size_t i) const { return dims_[i]; }

    std::size_t size() const {
        std::size_t n = 1;
        for (std::size_t d : dims_) n *= d;
        return n;
    }

    bool operator==(const shape& o) const { return dims_ == o.dims_; }
    bool operator!=(const shape& o) const { return !(*this == o); }

    const std::vector<std::size_t>& dims() const { return dims_; }

private:
    std::vector<std::size_t> dims_;
};

inline std::ostream& operator<<(std::ostream& os, const shape& s) {
    os << '[';
    for (std::size_t i = 0; i < s.rank(); ++i) {
        if (i) os << ", ";
        os << s.dim(i);
    }
    return os << ']';
}

}  // namespace qnn

#endif  // QNN_CORE_SHAPE_HPP