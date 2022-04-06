#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <utility>
#include <stdio.h>

#include <cuda_runtime.h>

//#include <omp.h>

#define SIDE 512
#define NTILES 4
#define NB_CORES 2


struct array_ops {
  typedef float Float;
  struct Offset {
    int value = 1;
  };
  struct Axis {
    size_t value = 1;
  };
  typedef size_t Index;
  struct Nat {
    size_t value = 1;
  };

  struct Array {
    Float * content;
    __host__ __device__ Array() {
      this -> content = new Float[SIDE * SIDE * SIDE];
    }

    __host__ __device__ Array(const Array & other) {
      this -> content = new Float[SIDE * SIDE * SIDE];
      memcpy(this -> content, other.content,
        SIDE * SIDE * SIDE * sizeof(Float));
    }

    __host__ __device__ Array(Array && other) {
      this -> content = std::move(other.content);
    }

    __host__ __device__ Array & operator = (const Array & other) {
      this -> content = new Float[SIDE * SIDE * SIDE];
      memcpy(this -> content, other.content,
        SIDE * SIDE * SIDE * sizeof(Float));
      return *this;
    }

    __host__ __device__ Array & operator = (Array && other) {
      this -> content = std::move(other.content);
      return *this;
    }

    __host__ __device__ inline Float operator[](const Index & ix) const {
      return this -> content[ix];
    }

    __host__ __device__ inline Float & operator[](const Index & ix) {
      return this -> content[ix];
    }
  };

  __host__ __device__ inline Float psi(const Index & ix,
    const Array & array) {
    return array[ix];
  }

  /* Float ops */
  __host__ __device__ inline Float unary_sub(const Float & f) {
    return -f;
  }
  __host__ __device__ inline Float binary_add(const Float & lhs,
    const Float & rhs) {
    return lhs + rhs;
  }
  __host__ __device__ inline Float binary_sub(const Float & lhs,
    const Float & rhs) {
    return lhs - rhs;
  }
  __host__ __device__ inline Float mul(const Float & lhs,
    const Float & rhs) {
    return lhs * rhs;
  }
  __host__ __device__ inline Float div(const Float & num,
    const Float & den) {
    return num / den;
  }
  __host__ __device__ inline Float one_float() {
    return 1;
  }
  __host__ __device__ inline Float two_float() {
    return 2;
  }
  __host__ __device__ inline Float three_float() {
    return 3;
  }

  /* Scalar-Array ops */
  __host__ __device__ inline Array binary_add(const Float & lhs,
    const Array & rhs) {
    Array out;
    for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
      out[i] = lhs + rhs[i];
    }
    return out;
  }
  __host__ __device__ inline Array binary_sub(const Float & lhs,
    const Array & rhs) {
    Array out;
    for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
      out[i] = lhs - rhs[i];
    }
    return out;
  }
  __host__ __device__ inline Array mul(const Float & lhs,
    const Array & rhs) {
    Array out;
    for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
      out[i] = lhs * rhs[i];
    }
    return out;
  }
  __host__ __device__ inline Array div(const Float & lhs,
    const Array & rhs) {
    Array out;
    for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
      out[i] = lhs / rhs[i];
    }
    return out;
  }

  /* Array-Array ops */
  __host__ __device__ inline Array binary_add(const Array & lhs,
    const Array & rhs) {
    Array out;
    for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
      out[i] = lhs[i] + rhs[i];
    }
    return out;
  }
  __host__ __device__ inline Array binary_sub(const Array & lhs,
    const Array & rhs) {
    Array out;
    for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
      out[i] = lhs[i] - rhs[i];
    }
    return out;
  }
  __host__ __device__ inline Array mul(const Array & lhs,
    const Array & rhs) {
    Array out;
    for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
      out[i] = lhs[i] * rhs[i];
    }
    return out;
  }
  __host__ __device__ inline Array div(const Array & lhs,
    const Array & rhs) {
    Array out;
    for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
      out[i] = lhs[i] / rhs[i];
    }
    return out;
  }

  [[noreturn]] __host__ __device__ inline Array rotate(const Array & array,
    const Axis & axis,
      const Offset & o) {
    printf("rotate not implemented\n");
    //std::unreachable(); // Always optimize with DNF, do not rotate
  }

  __host__ __device__ inline Index rotate_ix(const Index & ix,
    const Axis & axis,
      const Offset & offset) {
    if (axis.value == 0) {
      return (ix + (offset.value * SIDE * SIDE)) % (SIDE * SIDE * SIDE);
    } else if (axis.value == 1) {
      size_t ix_subarray_base = ix / (SIDE * SIDE);
      size_t ix_in_subarray = (ix + offset.value * SIDE) % (SIDE * SIDE);
      return ix_subarray_base + ix_in_subarray;
    } else if (axis.value == 2) {
      size_t ix_subarray_base = ix / SIDE;
      size_t ix_in_subarray = (ix + offset.value) % SIDE;
      return ix_subarray_base + ix_in_subarray;
    }

    //throw "failed at rotating index";
    //std::unreachable();
    return 0;
  }

  __host__ __device__ inline Axis zero_axis() {
    auto a = Axis();
    a.value = 0;
    return a;
  }
  __host__ __device__ inline Axis one_axis() {
    auto a = Axis();
    a.value = 1;
    return a;
  }
  __host__ __device__ inline Axis two_axis() {
    auto a = Axis();
    a.value = 2;
    return a;
  }

  __host__ __device__ inline Offset one_offset() {
    return Offset();
  }
  __host__ __device__ inline Offset unary_sub(const Offset & offset) {
    auto o = offset;
    o.value = -offset.value;
    return o;
  }
};

__host__ __device__ inline void dumpsine(array_ops::Array & result) {
  double step = 0.01;
  double PI = 3.14159265358979323846;
  double amplitude = 10.0;
  double phase = 0.0125;
  double t = 0.0;

  for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
    result[i] = amplitude * sin(PI * t + phase);
    t += step;
  }
}

template<class _snippet_ix>
  __global__ void ix_snippet_global(array_ops::Array *res, const array_ops::Array *u, const array_ops::Array *v, const array_ops::Array *u0, const array_ops::Array *u1, const array_ops::Array *u2,
    const array_ops::Float c0,
      const array_ops::Float c1,
        const array_ops::Float c2,
          const array_ops::Float c3,
            const array_ops::Float c4, _snippet_ix snippet_ix) {
    
    

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < SIDE*SIDE*SIDE) {
      res->content[i] = snippet_ix(*u, *v, *u0, *u1, *u2, c0, c1, c2, c3, c4, i);
    }

}

template < typename _Array, typename _Axis, typename _Float, typename _Index,
  typename _Nat, typename _Offset, class _snippet_ix >
  struct forall_ops {

    forall_ops < _Array, _Axis, _Float, _Index,
      _Nat, _Offset, _snippet_ix > () {};

    typedef _Array Array;
    typedef _Axis Axis;
    typedef _Float Float;
    typedef _Index Index;
    typedef _Nat Nat;
    typedef _Offset Offset;

    _snippet_ix snippet_ix;

    __host__ inline Array forall_ix_snippet_cuda(const Array & u,
      const Array & v,
        const Array & u0,
          const Array & u1,
            const Array & u2,
              const Float & c0,
                const Float & c1,
                  const Float & c2,
                    const Float & c3,
                      const Float & c4) {
      
      Float *u_host_content, *v_host_content, *u0_host_content, *u1_host_content, *u2_host_content;
      
      Float *u_dev_content, *v_dev_content, *u0_dev_content, *u1_dev_content, *u2_dev_content;

      Float *res_host_content, *res_dev_content;

      u_host_content = u.content;
      v_host_content = v.content;
      u0_host_content = u0.content;
      u1_host_content = u1.content;
      u2_host_content = u2.content;

      res_host_content = new Float[SIDE * SIDE * SIDE];

      cudaMalloc((void**)&u_dev_content, sizeof(Float) * SIDE * SIDE * SIDE);
      cudaMalloc((void**)&v_dev_content, sizeof(Float) * SIDE * SIDE * SIDE);
      cudaMalloc((void**)&u0_dev_content, sizeof(Float) * SIDE * SIDE * SIDE);
      cudaMalloc((void**)&u1_dev_content, sizeof(Float) * SIDE * SIDE * SIDE);
      cudaMalloc((void**)&u2_dev_content, sizeof(Float) * SIDE * SIDE * SIDE);
      cudaMalloc((void**)&res_dev_content, sizeof(Float) * SIDE * SIDE * SIDE);
      
      Array *res_dev, *u_dev, *v_dev, *u0_dev, *u1_dev, *u2_dev;

      cudaMalloc((void**)&res_dev, sizeof(*res_dev));
      cudaMalloc((void**)&u_dev, sizeof(*u_dev));
      cudaMalloc((void**)&v_dev, sizeof(*v_dev));
      cudaMalloc((void**)&u0_dev, sizeof(*u0_dev));
      cudaMalloc((void**)&u1_dev, sizeof(*u1_dev));
      cudaMalloc((void**)&u2_dev, sizeof(*u2_dev));
      cudaMalloc((void**)&res_dev, sizeof(*res_dev));

      cudaMemcpy(res_dev_content,res_host_content,sizeof(*res_dev)*SIDE*SIDE*SIDE,cudaMemcpyHostToDevice);
      cudaMemcpy(u_dev_content, u_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);
      cudaMemcpy(v_dev_content, v_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);
      cudaMemcpy(u0_dev_content, u0_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);
      cudaMemcpy(u1_dev_content, u1_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);
      cudaMemcpy(u2_dev_content, u2_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);
      cudaMemcpy(res_dev_content, u_host_content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);

      cudaMemcpy(&(res_dev->content), &res_dev_content, sizeof(res_dev->content), cudaMemcpyHostToDevice);
      cudaMemcpy(&(u_dev->content), &u_dev_content, sizeof(u_dev->content), cudaMemcpyHostToDevice);
      cudaMemcpy(&(v_dev->content), &v_dev_content, sizeof(v_dev->content), cudaMemcpyHostToDevice);
      cudaMemcpy(&(u0_dev->content), &u0_dev_content, sizeof(u0_dev->content), cudaMemcpyHostToDevice);
      cudaMemcpy(&(u1_dev->content), &u1_dev_content, sizeof(u1_dev->content), cudaMemcpyHostToDevice);
      cudaMemcpy(&(u2_dev->content), &u2_dev_content, sizeof(u2_dev->content), cudaMemcpyHostToDevice);

      

      ix_snippet_global<<<16,512>>>(res_dev, u_dev, v_dev, u0_dev, u1_dev, u2_dev, c0, c1, c2, c3, c4, snippet_ix);

      cudaDeviceSynchronize();

      cudaMemcpy(res_host_content, res_dev_content, sizeof(*res_host_content), cudaMemcpyDeviceToHost);

      Array res = Array();
      memcpy(res.content, res_host_content, sizeof(*res_host_content) * SIDE * SIDE * SIDE);
      
      return res;
    }

    __host__ __device__ inline Array forall_ix_snippet(const Array & u,
      const Array & v,
        const Array & u0,
          const Array & u1,
            const Array & u2,
              const Float & c0,
                const Float & c1,
                  const Float & c2,
                    const Float & c3,
                      const Float & c4) {
      Array result;
      //printf("in forall_ix_snippet\n");
      for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
        result[i] = snippet_ix(u, v, u0, u1, u2, c0, c1, c2, c3, c4, i);
      }

      return result;
    }  
  };