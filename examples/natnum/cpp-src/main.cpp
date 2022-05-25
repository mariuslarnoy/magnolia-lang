#include <iostream>
#include <gen/examples/natnum/mg-src/natnum-cpp.hpp>

using examples::natnum::mg_src::natnum_cpp::NaturalNumbers;

int main(int argc, char** argv) {

    NaturalNumbers N;

    std::cout << "zero(): " << N.zero() << std::endl;

    std::cout << "add(one(), one()) = " << N.add(N.one(), N.one()) << std::endl;

}