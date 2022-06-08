#include <vector>
#include <stdio.h>
#include <iostream>
//#include "gen/examples/pde/mg-src/pde-cpp.cuh"
#include "base.cuh"

static const double s_dt = 0.00082212448155679772495;
static const double s_nu = 1.0;
static const double s_dx = 1.0;
static const size_t steps = 10;

  typedef array_ops ArrayOps;

  typedef array_ops::Array Array;
  typedef array_ops::Index Index;
  typedef array_ops::Axis Axis;
  typedef array_ops::Float Float;
  typedef array_ops::Nat Nat;
  typedef array_ops::Offset Offset;

  typedef forall_ops<Array, Axis, Float, Index, Nat, Offset> ForallOps;

  __global__ void global_step(Array *u0, Array *u1, Array *u2,
                            Float s_nu, Float s_dx, Float s_dt) {
	if(threadIdx.x == 0) {
          ForallOps forall_ops;
	  forall_ops.step(*u0,*u1,*u2,s_nu,s_dx,s_dt);
	}
}

__global__ void ix_snippet_global(array_ops::Array res, const array_ops::Array u, const array_ops::Array v, const array_ops::Array u0, const array_ops::Array u1, const array_ops::Array u2,
  const array_ops::Float c0,
    const array_ops::Float c1,
      const array_ops::Float c2,
        const array_ops::Float c3,
          const array_ops::Float c4) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < SIDE*SIDE*SIDE) {
    res[i] = snippet_cuda(u, v, u0, u1, u2, c0, c1, c2, c3, c4, i);
  }
}

void allocateDeviceMemory(Float* &u0_host_content, 
                          Float* &u1_host_content,    
                          Float* &u2_host_content,
		                      Float* &u0_dev_content, 
                          Float* &u1_dev_content, Float* &u2_dev_content,
		          Array* &u0_dev, Array* &u1_dev, Array* &u2_dev) {

      cudaMalloc((void**)&u0_dev_content, sizeof(Float) * SIDE * SIDE * SIDE);
      cudaMalloc((void**)&u1_dev_content, sizeof(Float) * SIDE * SIDE * SIDE);
      cudaMalloc((void**)&u2_dev_content, sizeof(Float) * SIDE * SIDE * SIDE);

      cudaMalloc((void**)&u0_dev, sizeof(*u0_dev));
      cudaMalloc((void**)&u1_dev, sizeof(*u1_dev));
      cudaMalloc((void**)&u2_dev, sizeof(*u2_dev));

}

void copyDeviceMemory(Float* &u0_host_content, 
                      Float* &u1_host_content,    
                      Float* &u2_host_content,
                          Float* &u0_dev_content, 
                      Float* &u1_dev_content, Float* &u2_dev_content,
              Array* &u0_dev, Array* &u1_dev, Array* &u2_dev) {

      cudaMemcpy(u0_dev_content, u0_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);
      cudaMemcpy(u1_dev_content, u1_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);
      cudaMemcpy(u2_dev_content, u2_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);

      // Binding pointers with _dev
      cudaMemcpy(&(u0_dev->content), &u0_dev_content, sizeof(u0_dev->content), cudaMemcpyHostToDevice);
      cudaMemcpy(&(u1_dev->content), &u1_dev_content, sizeof(u1_dev->content), cudaMemcpyHostToDevice);
      cudaMemcpy(&(u2_dev->content), &u2_dev_content, sizeof(u2_dev->content), cudaMemcpyHostToDevice);

}

int main(void) {
    
    array_ops ArrayOps;

    Float c0 = ArrayOps.div(ArrayOps.div(1.0, 2.0), s_dx);
    Float c1 = ArrayOps.div(ArrayOps.div(1.0, s_dx), s_dx);
    Float c2 = ArrayOps.div(ArrayOps.div(2.0, s_dx), s_dx);
    Float c3 = s_nu;
    Float c4 = ArrayOps.div(s_dt, 2.0);

    size_t side = SIDE; //256;
    size_t array_size = side*side*side;
    std::cout << "Dims: " << side << "*" << side << "*" << side << ", steps: " << steps << std::endl;
    
    size_t mf, ma;
    cudaMemGetInfo(&mf,&ma);
    std::cout << "free: " << mf << " total: " << ma << std::endl;
  
    Array u0, u1, u2;
    
    dumpsine(u0);
    dumpsine(u1);
    dumpsine(u2);
    
    // Allocate host data 
    Float *u0_host_content, *u1_host_content, *u2_host_content;
    u0_host_content = u0.content;
    u1_host_content = u1.content;
    u2_host_content = u2.content;
    
    // Allocate device data
    Float *u0_dev_content, *u1_dev_content, *u2_dev_content;

    // Allocate device side helper structs
    Array *u0_dev, *u1_dev, *u2_dev;

    for (auto i = 0; i < steps; ++i) {

      allocateDeviceMemory(u0_host_content, u1_host_content, u2_host_content,
		           u0_dev_content, u1_dev_content, u2_dev_content,
			   u0_dev, u1_dev, u2_dev);

    } 

    
    cudaMemGetInfo(&mf,&ma);
    std::cout << "free: " << mf << " total: " << ma << std::endl;

    cudaDeviceReset();
    exit(0);
}

