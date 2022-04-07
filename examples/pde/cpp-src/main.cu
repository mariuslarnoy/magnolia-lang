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
template<class _PDEProgram>
__global__ void global_step(array_ops::Array &v0, array_ops::Array &v1, array_ops::Array& v2,
                            array_ops::Array& u0, array_ops::Array &u1, array_ops::Array &u2,
                            array_ops::Float s_nu, array_ops::Float s_dx, array_ops::Float s_dt, _PDEProgram pde) {
    v0 = u0;
    v1 = u1;
    v2 = u2;
    pde.step(v0,v1,v2,u0,u1,u2,s_nu,s_dx,s_dt);
    printf("%f %f %f\n", u0[0], u1[0], u2[0]);
}

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
    size_t steps = 10;
    //Shape shape = Shape(std::vector<size_t>({ side, side, side }));
    Array u0, u1, u2;
    
    dumpsine(u0);
    dumpsine(u1);
    dumpsine(u2);
    
    Array v0 = Array();
    Array v1 = Array();
    Array v2 = Array();
    
    memcpy(v0.content, u0.content, SIDE*SIDE*SIDE*sizeof(Float));
    memcpy(v1.content, u1.content, SIDE*SIDE*SIDE*sizeof(Float));
    memcpy(v2.content, u2.content, SIDE*SIDE*SIDE*sizeof(Float));

    Float *v0_host_content, *v1_host_content, *v2_host_content, *u0_host_content, *u1_host_content, *u2_host_content;
    Float *v0_dev_content, *v1_dev_content, *v2_dev_content, *u0_dev_content, *u1_dev_content, *u2_dev_content;

      v0_host_content = v0.content;
      v1_host_content = v1.content;
      v2_host_content = v2.content;
      u0_host_content = u0.content;
      u1_host_content = u1.content;
      u2_host_content = u2.content;

      cudaMalloc((void**)&v0_dev_content, sizeof(Float) * SIDE * SIDE * SIDE);
      cudaMalloc((void**)&v1_dev_content, sizeof(Float) * SIDE * SIDE * SIDE);
      cudaMalloc((void**)&v2_dev_content, sizeof(Float) * SIDE * SIDE * SIDE);
      cudaMalloc((void**)&u0_dev_content, sizeof(Float) * SIDE * SIDE * SIDE);
      cudaMalloc((void**)&u1_dev_content, sizeof(Float) * SIDE * SIDE * SIDE);
      cudaMalloc((void**)&u2_dev_content, sizeof(Float) * SIDE * SIDE * SIDE);

      Array  *v0_dev, *v1_dev, *v2_dev, *u0_dev, *u1_dev, *u2_dev;
	
      cudaMalloc((void**)&v0_dev, sizeof(*v0_dev));
      cudaMalloc((void**)&v1_dev, sizeof(*v1_dev));
      cudaMalloc((void**)&v2_dev, sizeof(*v2_dev));
      cudaMalloc((void**)&u0_dev, sizeof(*u0_dev));
      cudaMalloc((void**)&u1_dev, sizeof(*u1_dev));
      cudaMalloc((void**)&u2_dev, sizeof(*u2_dev));

      cudaMemcpy(v0_dev_content, v0_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);
      cudaMemcpy(v1_dev_content, v1_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);
      cudaMemcpy(v2_dev_content, v2_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);
      cudaMemcpy(u0_dev_content, u0_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);
      cudaMemcpy(u1_dev_content, u1_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);
      cudaMemcpy(u2_dev_content, u2_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);

      cudaMemcpy(&(v0_dev->content), &v0_dev_content, sizeof(u0_dev->content), cudaMemcpyHostToDevice);
      cudaMemcpy(&(v1_dev->content), &v1_dev_content, sizeof(u0_dev->content), cudaMemcpyHostToDevice);
      cudaMemcpy(&(v2_dev->content), &v2_dev_content, sizeof(u0_dev->content), cudaMemcpyHostToDevice);
      cudaMemcpy(&(u0_dev->content), &u0_dev_content, sizeof(u0_dev->content), cudaMemcpyHostToDevice);
      cudaMemcpy(&(u1_dev->content), &u1_dev_content, sizeof(u1_dev->content), cudaMemcpyHostToDevice);
      cudaMemcpy(&(u2_dev->content), &u2_dev_content, sizeof(u2_dev->content), cudaMemcpyHostToDevice);

      //std::cout << v0_dev << std::endl;
      for (auto i = 0; i< steps; i++) {
        global_step<<<1,1>>>(*v0_dev,*v1_dev,*v2_dev,*u0_dev,*u1_dev,*u2_dev,s_nu,s_dx,s_dt, pde);

        cudaDeviceSynchronize();
        cudaDeviceReset();
      }
    


        
      cudaMemcpy(v0_dev_content, v0_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);
      cudaMemcpy(v1_dev_content, v1_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);
      cudaMemcpy(v2_dev_content, v2_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);

      

      Array u0_res = Array();
      memcpy(u0_res.content, u0_host_content, sizeof(*u0_host_content) * SIDE * SIDE * SIDE);
      
      Array u1_res = Array();
      memcpy(u1_res.content, u1_host_content, sizeof(*u1_host_content) * SIDE * SIDE * SIDE);
      
      Array u2_res = Array();
      memcpy(u2_res.content, u2_host_content, sizeof(*u2_host_content) * SIDE * SIDE * SIDE);

      cudaFree(v0_dev_content);
      cudaFree(v1_dev_content);
      cudaFree(v2_dev_content);
      cudaFree(u0_dev_content);
      cudaFree(u1_dev_content);
      cudaFree(u2_dev_content);
      
      cudaFree(v0_dev);
      cudaFree(v1_dev);
      cudaFree(v2_dev);
      cudaFree(u0_dev);
      cudaFree(u1_dev);
      cudaFree(u2_dev);

      cudaDeviceReset();
      exit(0);
}

