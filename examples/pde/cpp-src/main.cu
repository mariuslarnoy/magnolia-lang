#include <vector>
#include <stdio.h>
#include <iostream>
#include "gen/examples/pde/mg-src/pde-cpp.cuh"
#include "base.cuh"

//typedef array_ops::Shape Shape;

static const double s_dt = 0.00082212448155679772495;
static const double s_nu = 1.0;
static const double s_dx = 1.0;
static const size_t steps = 10;

template<class _PDEProgram>
__global__ void global_step(array_ops::Array *u0, array_ops::Array *u1, array_ops::Array *u2,
                            const array_ops::Float s_nu, const array_ops::Float s_dx, const array_ops::Float s_dt, _PDEProgram pde) {
	if(threadIdx.x == 0) {
	  printf("%f %f %f \n", u0[0], u1[0], u2[0]);
	  //pde.step(*u0,*u1,*u2,s_nu,s_dx,s_dt);
	  
	}
}

int main(void) {
  
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
    std::cout << "Dims: " << side << "*" << side << "*" << side << ", steps: " << steps << std::endl;
    
    size_t mf, ma;
    cudaMemGetInfo(&mf,&ma);
    std::cout << "free: " << mf << " total: " << ma << std::endl;
    
    Array u0, u1, u2;
    
    dumpsine(u0);
    dumpsine(u1);
    dumpsine(u2);
    
    for (auto i = 0; i < steps; ++i) {
      
      // Allocate host data
      Float *u0_host_content, *u1_host_content, *u2_host_content;
      
      u0_host_content = u0.content;
      u1_host_content = u1.content;
      u2_host_content = u2.content;

      // Allocate device data
      Float *u0_dev_content, *u1_dev_content, *u2_dev_content;
      
      cudaMalloc((void**)&u0_dev_content, sizeof(Float) * array_size);
      cudaMalloc((void**)&u1_dev_content, sizeof(Float) * array_size);
      cudaMalloc((void**)&u2_dev_content, sizeof(Float) * array_size);
      

      // Allocate device side helper structs
      Array *u0_dev, *u1_dev, *u2_dev;
        
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
      global_step<<<1,1>>>(u0_dev,u1_dev,u2_dev,s_nu,s_dx,s_dt, pde);
      cudaDeviceSynchronize();

      // Copy u0, u1, u2 back to CPU
      cudaMemcpy(u0_host_content, u0_dev_content, sizeof(*u0_host_content) * array_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(u1_host_content, u1_dev_content, sizeof(*u1_host_content) * array_size, cudaMemcpyDeviceToHost);      
      cudaMemcpy(u2_host_content, u2_dev_content, sizeof(*u2_host_content) * array_size, cudaMemcpyDeviceToHost);

      // Reset device memory
      cudaDeviceReset();
      cudaMemGetInfo(&mf,&ma);
      std::cout << "free: " << mf << " total: " << ma << std::endl;

    }
    
    /* 
    Array v0, v1, v2;

    zeros(v0);
    zeros(v1);
    zeros(v2);
    
    memcpy(v0.content, u0.content, SIDE*SIDE*SIDE*sizeof(Float));
    memcpy(v1.content, u1.content, SIDE*SIDE*SIDE*sizeof(Float));
    memcpy(v2.content, u2.content, SIDE*SIDE*SIDE*sizeof(Float));
    
    // Allocate host data 
    Float *v0_host_content, *v1_host_content, *v2_host_content, *u0_host_content, *u1_host_content, *u2_host_content;
      	
    v0_host_content = v0.content;
    v1_host_content = v1.content;
    v2_host_content = v2.content;
    u0_host_content = u0.content;
    u1_host_content = u1.content;
    u2_host_content = u2.content;
    
    // Allocate device data
    Float *v0_dev_content, *v1_dev_content, *v2_dev_content, *u0_dev_content, *u1_dev_content, *u2_dev_content;

    cudaMalloc((void**)&v0_dev_content, sizeof(Float) * SIDE * SIDE * SIDE);
    cudaMalloc((void**)&v1_dev_content, sizeof(Float) * SIDE * SIDE * SIDE);
    cudaMalloc((void**)&v2_dev_content, sizeof(Float) * SIDE * SIDE * SIDE);
    cudaMalloc((void**)&u0_dev_content, sizeof(Float) * SIDE * SIDE * SIDE);
    cudaMalloc((void**)&u1_dev_content, sizeof(Float) * SIDE * SIDE * SIDE);
    cudaMalloc((void**)&u2_dev_content, sizeof(Float) * SIDE * SIDE * SIDE);
      
    // Allocate device side helper structs
    Array  *v0_dev, *v1_dev, *v2_dev, *u0_dev, *u1_dev, *u2_dev;
	
    cudaMalloc((void**)&v0_dev, sizeof(*v0_dev));
    cudaMalloc((void**)&v1_dev, sizeof(*v1_dev));
    cudaMalloc((void**)&v2_dev, sizeof(*v2_dev));
    cudaMalloc((void**)&u0_dev, sizeof(*u0_dev));
    cudaMalloc((void**)&u1_dev, sizeof(*u1_dev));
    cudaMalloc((void**)&u2_dev, sizeof(*u2_dev));

    // Copy data from host to device
    cudaMemcpy(v0_dev_content, v0_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);
    cudaMemcpy(v1_dev_content, v1_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);
    cudaMemcpy(v2_dev_content, v2_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);
    cudaMemcpy(u0_dev_content, u0_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);
    cudaMemcpy(u1_dev_content, u1_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);
    cudaMemcpy(u2_dev_content, u2_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);
      
    // Binding pointers with _dev
    cudaMemcpy(&(v0_dev->content), &v0_dev_content, sizeof(u0_dev->content), cudaMemcpyHostToDevice);
    cudaMemcpy(&(v1_dev->content), &v1_dev_content, sizeof(u0_dev->content), cudaMemcpyHostToDevice);
    cudaMemcpy(&(v2_dev->content), &v2_dev_content, sizeof(u0_dev->content), cudaMemcpyHostToDevice);
    cudaMemcpy(&(u0_dev->content), &u0_dev_content, sizeof(u0_dev->content), cudaMemcpyHostToDevice);
    cudaMemcpy(&(u1_dev->content), &u1_dev_content, sizeof(u1_dev->content), cudaMemcpyHostToDevice);
    cudaMemcpy(&(u2_dev->content), &u2_dev_content, sizeof(u2_dev->content), cudaMemcpyHostToDevice);
      
    // Launch parent kernel
    global_step<<<1,1>>>(*v0_dev,*v1_dev,*v2_dev,*u0_dev,*u1_dev,*u2_dev,s_nu,s_dx,s_dt, pde);
    cudaDeviceSynchronize();


    cudaMemGetInfo(&mf,&ma);
    std::cout << "free: " << mf << " total: " << ma << std::endl;
*/
    cudaDeviceReset();
    exit(0);
}

