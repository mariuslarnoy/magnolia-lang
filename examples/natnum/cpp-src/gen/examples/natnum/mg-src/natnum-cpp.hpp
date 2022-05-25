#pragma once

#include "base.hpp"
#include <cassert>


namespace examples {
namespace natnum {
namespace mg_src {
namespace natnum_cpp {
struct NaturalNumbers {
private:
    static nat __nat;
public:
    typedef nat::Nat Nat;
    struct _add {
        inline NaturalNumbers::Nat operator()(const NaturalNumbers::Nat& a, const NaturalNumbers::Nat& b) {
            return __nat.add(a, b);
        };
    };

    static NaturalNumbers::_add add;
    struct _mul {
        inline NaturalNumbers::Nat operator()(const NaturalNumbers::Nat& a, const NaturalNumbers::Nat& b) {
            return __nat.mul(a, b);
        };
    };

    static NaturalNumbers::_mul mul;
    struct _one {
        inline NaturalNumbers::Nat operator()() {
            return __nat.one();
        };
    };

    static NaturalNumbers::_one one;
    struct _zero {
        inline NaturalNumbers::Nat operator()() {
            return __nat.zero();
        };
    };

    static NaturalNumbers::_zero zero;
};
} // examples
} // natnum
} // mg_src
} // natnum_cpp