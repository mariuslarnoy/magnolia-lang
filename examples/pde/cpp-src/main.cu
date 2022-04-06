#include <vector>
#include <stdio.h>
#include <iostream>
#include "gen/examples/pde/mg-src/pde-cpp.cuh"
#include "base.cuh"

//typedef array_ops::Shape Shape;

static const double s_dt = 0.00082212448155679772495;
static const double s_nu = 1.0;
static const double s_dx = 1.0;

/*
template<>
Array forall_ops<Array,Axis,Float,Index,Nat,Offset,PDEProgram::_snippet_ix>::forall_ix_snippet_cuda(const Array &u, const Array &v,
const Array &u0, const Array &u1, const Array &u2, const Float &c0,
const Float &c1, const Float &c2, const Float &c3, const Float &c4) {

    
  }
*/

int main() {

  typedef array_ops ArrayOps;

  typedef array_ops::Array Array;
  typedef array_ops::Index Index;
  typedef array_ops::Axis Axis;
  typedef array_ops::Float Float;
  typedef array_ops::Nat Nat;
  typedef array_ops::Offset Offset;
  
  examples::pde::mg_src::pde_cpp::PDEProgram pde = examples::pde::mg_src::pde_cpp::PDEProgram();

    size_t side = SIDE; //256;
    size_t array_size = side*side*side;
    size_t steps = 20;
    //Shape shape = Shape(std::vector<size_t>({ side, side, side }));
    Array u0, u1, u2;
    
    dumpsine(u0);
    dumpsine(u1);
    dumpsine(u2);

    //std::cout << "before loop" << std::endl;
    for (auto i = 0; i < steps; ++i) {
        //std::cout << "step nr: " << i << std::endl;
        pde.step(u0,u1,u2,s_nu,s_dx,s_dt);
        
        std::cout << u0[0] << " "
                  << u1[0] << " "
                  << u2[0] << std::endl;
        
    }
    //std::cout << "after loop" << std::endl;

    //for (auto i = 0; i < SIDE*SIDE*SIDE; ++i) {
    //  std::cout << u0[i] << std::endl;
    //}
    //return 0;
    /*
    
    It's like: step { snippet { parallelize ix computations here } }
    */
}