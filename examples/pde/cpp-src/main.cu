#include <vector>
#include <stdio.h>
#include <iostream>
#include "gen/examples/pde/mg-src/pde-cpp.cuh"
#include "base.cuh"

//typedef array_ops::Shape Shape;
typedef array_ops ArrayOps;

typedef array_ops::Array Array;
typedef array_ops::Index Index;
typedef array_ops::Axis Axis;
typedef array_ops::Float Float;
typedef array_ops::Nat Nat;
typedef array_ops::Offset Offset;

typedef examples::pde::mg_src::pde_cpp::PDEProgram PDEProgram;

static const double s_dt = 0.00082212448155679772495;
static const double s_nu = 1.0;
static const double s_dx = 1.0;

__global__ void forall_kernel(Array *u, Array *v, Array *u0, Array *u1, Array *u2, float c0, float c1, float c2, float c3, float c4) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < SIDE * SIDE * SIDE) {

        u[i] = PDEProgram::snippet_ix(u, v, u0, u1, u2, c0, c1, c2, c3, c4);
    }
}

__host__ __device__ Array forall_ops<Array,Axis,Float,Index,Nat,Offset,PDEProgram::_snippet_ix>::forall_ix_snippet_cuda(const Array &u, const Array &v,
const Array &u0, const Array &u1, const Array &u2, const Float &c0,
const Float &c1, const Float &c2, const Float &c3, const Float &c4) {

    Array *u_d,*v_d,*d0, *d1, *d2;

    cudaMalloc(&u_d,SIDE*SIDE*SIDE*sizeof(Array));
    cudaMalloc(&v_d,SIDE*SIDE*SIDE*sizeof(Array));
    cudaMalloc(&d0,SIDE*SIDE*SIDE*sizeof(Array));
    cudaMalloc(&d1,SIDE*SIDE*SIDE*sizeof(Array));
    cudaMalloc(&d2,SIDE*SIDE*SIDE*sizeof(Array));

    cudaMemcpy(&u_d,&u,SIDE*SIDE*SIDE*sizeof(Array),cudaMemcpyHostToDevice);
    cudaMemcpy(&v_d,&v,SIDE*SIDE*SIDE*sizeof(Array),cudaMemcpyHostToDevice);
    cudaMemcpy(&d0,&u0,SIDE*SIDE*SIDE*sizeof(Array),cudaMemcpyHostToDevice);
    cudaMemcpy(&d1,&u1,SIDE*SIDE*SIDE*sizeof(Array),cudaMemcpyHostToDevice);
    cudaMemcpy(&d2,&u2,SIDE*SIDE*SIDE*sizeof(Array),cudaMemcpyHostToDevice);
    
    forall_kernel<<<1,1>>>(u_d, v_d, d0, d1, d2, c0, c1, c2, c3, c4);
    
    cudaDeviceSynchronize();

    //cudaMemcpy(&u,&u_d,SIDE*SIDE*SIDE*sizeof(Array),cudaMemcpyDeviceToHost);
    //cudaMemcpy(&v,&v_d,SIDE*sizeof(Array),cudaMemcpyDeviceToHost);
    //cudaMemcpy(&u0,&d0,SIDE*sizeof(Array),cudaMemcpyDeviceToHost);
    //cudaMemcpy(&u1,&d1,SIDE*sizeof(Array),cudaMemcpyDeviceToHost);
    //cudaMemcpy(&u2,&d2,SIDE*sizeof(Array),cudaMemcpyDeviceToHost);

    return u0;
  }

int main() {

    
    size_t side = SIDE; //256;
    size_t array_size = side*side*side;
    size_t steps = 20;
    //Shape shape = Shape(std::vector<size_t>({ side, side, side }));
    Array u0, u1, u2;
    
    dumpsine(u0);
    dumpsine(u1);
    dumpsine(u2);

    for (auto i = 0; i < steps; ++i) {
        PDEProgram::step(u0,u1,u2,s_nu,s_dx,s_dt);
    }

    return 0;
    /*
    
    It's like: step { snippet { parallelize ix computations here } }
    */
}