#include <string>

struct nat {
    typedef size_t Nat;

    inline Nat zero() { return 0; }
    inline Nat one() { return 1; }

    inline Nat add(Nat a, Nat b) { return a + b; }
    inline Nat mul(Nat a, Nat b) { return a * b; }
};