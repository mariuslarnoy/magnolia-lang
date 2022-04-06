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

#define SIDE 30
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
      //printf("Array()\n");
      this -> content = new Float[SIDE * SIDE * SIDE];
    }

    __host__ __device__ Array(const Array & other) {
      //printf("Array(const Array& other)\n");
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
  __global__ void ix_snippet_global(array_ops::Array res, const array_ops::Array u, const array_ops::Array v, const array_ops::Array u0, const array_ops::Array u1, const array_ops::Array u2,
    const array_ops::Float c0,
      const array_ops::Float c1,
        const array_ops::Float c2,
          const array_ops::Float c3,
            const array_ops::Float c4, _snippet_ix snippet_ix) {
    
    printf("core\n");

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < SIDE*SIDE*SIDE) {
      res[i] = snippet_ix(u, v, u0, u1, u2, c0, c1, c2, c3, c4, i);
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

    __host__ void allocateDeviceArray(Array * hostArray, Array& deviceArray) {

      // allocate elems
      cudaMalloc(&deviceArray.content, sizeof(Float) * SIDE * SIDE * SIDE);

      // copy elems
      cudaMemcpy(deviceArray.content, hostArray->content, sizeof(Float) * SIDE * SIDE * SIDE, cudaMemcpyHostToDevice);

    }

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
      
      Array res = Array();
      Array res_d, u_d, v_d, u0_d, u1_d, u2_d;
    
      Array u_h = Array(u);
      Array v_h = Array(v);
      Array u0_h = Array(u0);
      Array u1_h = Array(u1);
      Array u2_h = Array(u2);

      Array* res_ptr = &res;
      Array* u_ptr = &u_h;
      Array* v_ptr = &v_h;
      Array* u0_ptr = &u0_h;
      Array* u1_ptr = &u1_h;
      Array* u2_ptr = &u2_h;

      allocateDeviceArray(res_ptr, res_d);
      allocateDeviceArray(u_ptr, u_d);
      allocateDeviceArray(v_ptr, v_d);
      allocateDeviceArray(u0_ptr, u0_d);
      allocateDeviceArray(u1_ptr, u1_d);
      allocateDeviceArray(u2_ptr, u2_d);    
      

      ix_snippet_global<<<1,2>>>(res_d,u_d,v_d,u0_d,u1_d,u2_d,c0,c1, c2, c3, c4, snippet_ix);
      
      cudaMemcpy(res_ptr->content,res_d.content,sizeof(Float)*SIDE*SIDE*SIDE,cudaMemcpyDeviceToHost);
      
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