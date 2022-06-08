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


__global__ void ix_snippet_global(array_ops::Array *u, const array_ops::Array *v, const array_ops::Array *u0, const array_ops::Array *u1, const array_ops::Array *u2,
  const array_ops::Float c0,
    const array_ops::Float c1,
      const array_ops::Float c2,
        const array_ops::Float c3,
          const array_ops::Float c4) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.x;
  int i = y*SIDE+x;
  if (i < SIDE*SIDE*SIDE) {
    u->content[i] = snippet_cuda(*u, *v, *u0, *u1, *u2, c0, c1, c2, c3, c4, i);
  }
  
}

void allocateDeviceMemory(Float* &u0_host_content, 
                          Float* &u1_host_content,    
                          Float* &u2_host_content,
		                      Float* &u0_dev_content, 
                          Float* &u1_dev_content, 
                          Float* &u2_dev_content,
                          Float* &v0_host_content,
                          Float* &v1_host_content,
                          Float* &v2_host_content,
                          Float* &v0_dev_content,
                          Float* &v1_dev_content,
                          Float* &v2_dev_content,
		                      Array* &u0_dev, Array* &u1_dev, Array* &u2_dev,
                          Array* &v0_dev, Array* &v1_dev, Array* &v2_dev) {

      cudaMalloc((void**)&u0_dev_content, sizeof(Float) * SIDE * SIDE * SIDE);
      cudaMalloc((void**)&u1_dev_content, sizeof(Float) * SIDE * SIDE * SIDE);
      cudaMalloc((void**)&u2_dev_content, sizeof(Float) * SIDE * SIDE * SIDE);
      cudaMalloc((void**)&v0_dev_content, sizeof(Float) * SIDE * SIDE * SIDE);
      cudaMalloc((void**)&v1_dev_content, sizeof(Float) * SIDE * SIDE * SIDE);
      cudaMalloc((void**)&v2_dev_content, sizeof(Float) * SIDE * SIDE * SIDE);

      cudaMalloc((void**)&u0_dev, sizeof(*u0_dev));
      cudaMalloc((void**)&u1_dev, sizeof(*u1_dev));
      cudaMalloc((void**)&u2_dev, sizeof(*u2_dev));
      cudaMalloc((void**)&v0_dev, sizeof(*v0_dev));
      cudaMalloc((void**)&v1_dev, sizeof(*v1_dev));
      cudaMalloc((void**)&v2_dev, sizeof(*v2_dev));

}

void copyDeviceMemory(Float* &u0_host_content, 
                      Float* &u1_host_content,    
                      Float* &u2_host_content,
                      Float* &u0_dev_content, 
                      Float* &u1_dev_content, 
                      Float* &u2_dev_content,
                      Float* &v0_host_content,
                      Float* &v1_host_content,
                      Float* &v2_host_content,
                      Float* &v0_dev_content,
                      Float* &v1_dev_content,
                      Float* &v2_dev_content,
                      Array* &u0_dev, Array* &u1_dev, Array* &u2_dev,
                      Array* &v0_dev, Array* &v1_dev, Array* &v2_dev) {

      cudaMemcpy(u0_dev_content, u0_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);
      cudaMemcpy(u1_dev_content, u1_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);
      cudaMemcpy(u2_dev_content, u2_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);
      cudaMemcpy(v0_dev_content, v0_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);
      cudaMemcpy(v1_dev_content, v1_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);
      cudaMemcpy(v2_dev_content, v2_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);

      // Binding pointers with _dev
      cudaMemcpy(&(u0_dev->content), &u0_dev_content, sizeof(u0_dev->content), cudaMemcpyHostToDevice);
      cudaMemcpy(&(u1_dev->content), &u1_dev_content, sizeof(u1_dev->content), cudaMemcpyHostToDevice);
      cudaMemcpy(&(u2_dev->content), &u2_dev_content, sizeof(u2_dev->content), cudaMemcpyHostToDevice);
      cudaMemcpy(&(v0_dev->content), &v0_dev_content, sizeof(v0_dev->content), cudaMemcpyHostToDevice);
      cudaMemcpy(&(v1_dev->content), &v1_dev_content, sizeof(v1_dev->content), cudaMemcpyHostToDevice);
      cudaMemcpy(&(v2_dev->content), &v2_dev_content, sizeof(v2_dev->content), cudaMemcpyHostToDevice);
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
  

    //kernel dims
    dim3 block_shape = dim3(BLOCK_SIZE, BLOCK_DIM);
    dim3 thread_shape = dim3(THREAD_SIZE);

    Array u0, u1, u2;
    
    dumpsine(u0);
    dumpsine(u1);
    dumpsine(u2);
    
    // Allocate host data 
    Float *u0_host_content, *u1_host_content, *u2_host_content;
    Float *v0_host_content, *v1_host_content, *v2_host_content;

    u0_host_content = u0.content;
    u1_host_content = u1.content;
    u2_host_content = u2.content;
    v0_host_content = u0_host_content;
    v1_host_content = u1_host_content;
    v2_host_content = u2_host_content;
    
    // Allocate device data
    Float *u0_dev_content, *u1_dev_content, *u2_dev_content;
    Float *v0_dev_content, *v1_dev_content, *v2_dev_content;

    // Allocate device side helper structs
    Array *u0_dev, *u1_dev, *u2_dev;
    Array *v0_dev, *v1_dev, *v2_dev;
    
    allocateDeviceMemory(u0_host_content, u1_host_content, u2_host_content,
                         u0_dev_content, u1_dev_content, u2_dev_content,
                         v0_host_content, v1_host_content, v2_host_content,
                         v0_dev_content, v1_dev_content, v2_dev_content,    
                         u0_dev, u1_dev, u2_dev,
                         v0_dev, v1_dev, v2_dev);

    copyDeviceMemory(u0_host_content, u1_host_content, u2_host_content,
                     u0_dev_content, u1_dev_content, u2_dev_content,
                     v0_host_content, v1_host_content, v2_host_content,
                     v0_dev_content, v1_dev_content, v2_dev_content,
                     u0_dev, u1_dev, u2_dev,
                     v0_dev, v1_dev, v2_dev);

    // STEP
    for (auto i = 0; i < steps; ++i) {

      ix_snippet_global<<<block_shape,thread_shape>>>(v0_dev, u0_dev, u0_dev, u1_dev, u2_dev, c0, c1, c2, c3, c4);
      ix_snippet_global<<<block_shape,thread_shape>>>(v1_dev, u1_dev, u0_dev, u1_dev, u2_dev, c0, c1, c2, c3, c4);
      ix_snippet_global<<<block_shape,thread_shape>>>(v2_dev, u2_dev, u0_dev, u1_dev, u2_dev, c0, c1, c2, c3, c4);
      ix_snippet_global<<<block_shape,thread_shape>>>(u0_dev, v0_dev, u0_dev, u1_dev, u2_dev, c0, c1, c2, c3, c4);
      ix_snippet_global<<<block_shape,thread_shape>>>(u1_dev, v1_dev, u0_dev, u1_dev, u2_dev, c0, c1, c2, c3, c4);
      ix_snippet_global<<<block_shape,thread_shape>>>(u2_dev, v2_dev, u0_dev, u1_dev, u2_dev, c0, c1, c2, c3, c4);
                   
      cudaMemcpy(u0_host_content, u0_dev_content, sizeof(Float) * array_size, cudaMemcpyDeviceToHost);
      std::cout << "u0: " << u0_host_content[0] << std::endl;
    
    } 
    cudaMemGetInfo(&mf,&ma);
    std::cout << "free: " << mf << " total: " << ma << std::endl;

    cudaDeviceReset();
    exit(0);
}

