#include <vector>
#include <stdio.h>
#include <iostream>
//#include "gen/examples/pde/mg-src/pde-cpp.cuh"
#include "base.cuh"

//typedef array_ops::Shape Shape;

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

int main(void) {

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
	
      cudaMalloc((void**)&u0_dev_content, sizeof(Float) * SIDE * SIDE * SIDE);
      cudaMalloc((void**)&u1_dev_content, sizeof(Float) * SIDE * SIDE * SIDE);
      cudaMalloc((void**)&u2_dev_content, sizeof(Float) * SIDE * SIDE * SIDE);

      cudaMalloc((void**)&u0_dev, sizeof(*u0_dev));
      cudaMalloc((void**)&u1_dev, sizeof(*u1_dev));
      cudaMalloc((void**)&u2_dev, sizeof(*u2_dev));

      // Copy data from host to device
      cudaMemcpy(u0_dev_content, u0_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);
      cudaMemcpy(u1_dev_content, u1_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);
      cudaMemcpy(u2_dev_content, u2_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);

      // Binding pointers with _dev
      cudaMemcpy(&(u0_dev->content), &u0_dev_content, sizeof(u0_dev->content), cudaMemcpyHostToDevice);
      cudaMemcpy(&(u1_dev->content), &u1_dev_content, sizeof(u1_dev->content), cudaMemcpyHostToDevice);
      cudaMemcpy(&(u2_dev->content), &u2_dev_content, sizeof(u2_dev->content), cudaMemcpyHostToDevice);
    
      // Launch parent kernel
      global_step<<<1,1>>>(u0_dev,u1_dev,u2_dev,s_nu,s_dx,s_dt);
      cudaDeviceSynchronize();
 	
      
      // Copy back to CPU
      Float *u0_test_content;
      u0_test_content = new Float[array_size];

      cudaMemcpy(u0_test_content, u0_dev_content, sizeof(*u0_test_content), cudaMemcpyDeviceToHost);

      Array test = Array();
      memcpy(test.content, u0_test_content, sizeof(*u0_test_content)*array_size);
      
      std::cout << "test[0]: " << test[0] << std::endl;
      cudaDeviceReset();
    } 


    cudaMemGetInfo(&mf,&ma);
    std::cout << "free: " << mf << " total: " << ma << std::endl;

    cudaDeviceReset();
    exit(0);
}

