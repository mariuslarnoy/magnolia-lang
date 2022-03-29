#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <utility>

#include <omp.h>

#define SIDE 30
#define NTILES 4
#define NB_CORES 2


struct array_ops {
  typedef float Float;
  struct Offset { int value = 1; };
  struct Axis { size_t value = 1; };
  typedef size_t Index;
  struct Nat { size_t value = 1; };

  struct Array {
    std::unique_ptr<Float[]> content;
    Array() {
      this->content = std::unique_ptr<Float[]>(new Float[SIDE * SIDE * SIDE]);
    }

    Array(const Array &other) {
      this->content = std::unique_ptr<Float[]>(new Float[SIDE * SIDE * SIDE]);
      memcpy(this->content.get(), other.content.get(),
             SIDE * SIDE * SIDE * sizeof(Float));
    }

    Array(Array &&other) {
        this->content = std::move(other.content);
    }

    Array &operator=(const Array &other) {
      this->content = std::unique_ptr<Float[]>(new Float[SIDE * SIDE * SIDE]);
      memcpy(this->content.get(), other.content.get(),
             SIDE * SIDE * SIDE * sizeof(Float));
      return *this;
    }

    __host__ __device__ Array &operator=(Array &&other) {
        this->content = std::move(other.content);
        return *this;
    }

    __host__ __device__ inline Float operator[](const Index &ix) const {
      return this->content[ix];
    }

    __host__ __device__ inline Float &operator[](const Index &ix) {
      return this->content[ix];
    }
  };

  inline Float psi(const Index &ix, const Array &array) { return array[ix]; }

  /* Float ops */
  inline Float unary_sub(const Float &f) { return -f; }
  inline Float binary_add(const Float &lhs, const Float &rhs) {
    return lhs + rhs;
  }
  inline Float binary_sub(const Float &lhs, const Float &rhs) {
    return lhs - rhs;
  }
  inline Float mul(const Float &lhs, const Float &rhs) {
    return lhs * rhs;
  }
  inline Float div(const Float &num, const Float &den) {
    return num / den;
  }
  __host__ __device__ inline Float one_float() { return 1; }
  inline Float two_float() { return 2; }
  inline Float three_float() { return 3; }

  /* Scalar-Array ops */
  inline Array binary_add(const Float &lhs, const Array &rhs) {
    Array out;
    for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
      out[i] = lhs + rhs[i];
    }
    return out;
  }
  inline Array binary_sub(const Float &lhs, const Array &rhs) {
    Array out;
    for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
      out[i] = lhs - rhs[i];
    }
    return out;
  }
  inline Array mul(const Float &lhs, const Array &rhs) {
    Array out;
    for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
      out[i] = lhs * rhs[i];
    }
    return out;
  }
  inline Array div(const Float &lhs, const Array &rhs) {
    Array out;
    for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
      out[i] = lhs / rhs[i];
    }
    return out;
  }

  /* Array-Array ops */
  inline Array binary_add(const Array &lhs, const Array &rhs) {
    Array out;
    for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
      out[i] = lhs[i] + rhs[i];
    }
    return out;
  }
  inline Array binary_sub(const Array &lhs, const Array &rhs) {
    Array out;
    for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
      out[i] = lhs[i] - rhs[i];
    }
    return out;
  }
  inline Array mul(const Array &lhs, const Array &rhs) {
    Array out;
    for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
      out[i] = lhs[i] * rhs[i];
    }
    return out;
  }
  inline Array div(const Array &lhs, const Array &rhs) {
    Array out;
    for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
      out[i] = lhs[i] / rhs[i];
    }
    return out;
  }

  [[noreturn]] inline Array rotate(const Array &array, const Axis &axis, const Offset &o) {
    throw "rotate not implemented";
    //std::unreachable(); // Always optimize with DNF, do not rotate
  }

  inline Index rotate_ix(const Index &ix, const Axis &axis, const Offset &offset) {
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

    throw "failed at rotating index";
    //std::unreachable();
    return 0;
  }

  inline Axis zero_axis() {auto a = Axis(); a.value = 0; return a; }
  inline Axis one_axis() { auto a = Axis(); a.value = 1; return a; }
  inline Axis two_axis() { auto a = Axis(); a.value = 2; return a; }

  inline Offset one_offset() { return Offset(); }
  inline Offset unary_sub(const Offset &offset) { auto o = offset; o.value = -offset.value; return o; }
};

template <typename _Array, typename _Axis, typename _Float, typename _Index,
          typename _Nat, typename _Offset, class _snippet_ix>
struct forall_ops {
  typedef _Array Array;
  typedef _Axis Axis;
  typedef _Float Float;
  typedef _Index Index;
  typedef _Nat Nat;
  typedef _Offset Offset;

  _snippet_ix snippet_ix;

  //inline Nat nbCores() { auto n = Nat(); n.value = NB_CORES; return n; }

  inline Array forall_ix_snippet(const Array &u, const Array &v,
      const Array &u0, const Array &u1, const Array &u2, const Float &c0,
      const Float &c1, const Float &c2, const Float &c3, const Float &c4) {
    Array result;
    std::cout << "in forall_ix_snippet" << std::endl;
    for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
      result[i] = snippet_ix(u, v, u0, u1, u2, c0, c1, c2, c3, c4, i);
    }

    std::cout << result[SIDE * SIDE * SIDE - 1] << " "
              << result[0] << std::endl;

    return result;
  }

 /*
  inline Array forall_ix_snippet_threaded(const Array &u, const Array &v,
      const Array &u0, const Array &u1, const Array &u2, const Float &c0,
      const Float &c1, const Float &c2, const Float &c3, const Float &c4,
      const Nat &nbThreads) {
    Array result;
    omp_set_num_threads(nbThreads.value);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
      result[i] = snippet_ix(u, v, u0, u1, u2, c0, c1, c2, c3, c4, i);
    }
    return result;
  }

  inline Array forall_ix_snippet_tiled(const Array &u, const Array &v,
      const Array &u0, const Array &u1, const Array &u2, const Float &c0,
      const Float &c1, const Float &c2, const Float &c3, const Float &c4) {
    Array result;

    #pragma omp parallel for schedule(static) collapse(3)
    for (size_t ti = 0; ti < SIDE; ti += SIDE/NTILES) {
      for (size_t tj = 0; tj < SIDE; tj += SIDE/NTILES) {
        for (size_t tk = 0; tk < SIDE; tk += SIDE/NTILES) {
          for (size_t i = ti; i < ti + SIDE/NTILES; ++i) {
            for (size_t j = tj; j < tj + SIDE/NTILES; ++j) {
              for (size_t k = tk; k < tk + SIDE/NTILES; ++k) {
                size_t ix = i * SIDE * SIDE + j * SIDE + k;
                //assert (ix < SIDE * SIDE * SIDE);
                result[ix] = snippet_ix(u, v, u0, u1, u2, c0, c1, c2, c3, c4, ix);
              }
            }
          }
        }
      }
    }

    return result;
  }*/
  template <class __snippet_ix>
  struct _forall_kernel {

    __snippet_ix snippet_ix;
    __host__ __device__ void operator()(Array *res, Array *u, Array *v, Array *u0, Array *u1, Array *u2, float c0, float c1, float c2, float c3, float c4) {

        int i = blockIdx.x*blockDim.x+threadIdx.x;
        printf("hello");
        if (i < SIDE * SIDE * SIDE) {
            printf("%f", snippet_ix(*u, *v, *u0, *u1, *u2, c0, c1, c2, c3, c4,i));
        }
      }
  };

  inline Array forall_ix_snippet_cuda(const Array &u, const Array &v,
    const Array &u0, const Array &u1, const Array &u2, const Float &c0,
    const Float &c1, const Float &c2, const Float &c3, const Float &c4) {
      printf("sdfsdf");
      _forall_kernel<_snippet_ix> forall_kernel;

      array_ops::Array *u_d,*v_d,*d0, *d1, *d2;
      array_ops::Array *res_h, *res_d;

      cudaMalloc(&res_d,sizeof(array_ops::Array));
      cudaMalloc(&u_d,sizeof(array_ops::Array));
      cudaMalloc(&v_d,sizeof(array_ops::Array));
      cudaMalloc(&d0,sizeof(array_ops::Array));
      cudaMalloc(&d1,sizeof(array_ops::Array));
      cudaMalloc(&d2,sizeof(array_ops::Array));
      printf("hello9");
      cudaMemcpy(u_d,&u,sizeof(array_ops::Array),cudaMemcpyHostToDevice);
      Float *u_ptr;
      cudaMalloc(&u_ptr,sizeof(array_ops::Float)*SIDE*SIDE*SIDE);
      cudaMemcpy(u_ptr,u.content.get(),sizeof(array_ops::Float)*SIDE*SIDE*SIDE,cudaMemcpyHostToDevice);
      u_d->content = std::unique_ptr<Float[]>(u_ptr);
      printf("hello1");
      cudaMemcpy(v_d,&v,sizeof(array_ops::Array),cudaMemcpyHostToDevice);
      Float *v_ptr;
      cudaMalloc(&v_ptr,sizeof(array_ops::Float)*SIDE*SIDE*SIDE);
      cudaMemcpy(v_ptr,v.content.get(),sizeof(array_ops::Float)*SIDE*SIDE*SIDE,cudaMemcpyHostToDevice);
      v_d->content = std::unique_ptr<Float[]>(v_ptr);
      printf("hello2");
      cudaMemcpy(d0,&u0,sizeof(array_ops::Array),cudaMemcpyHostToDevice);
      Float *d0_ptr;
      cudaMalloc(&d0_ptr,sizeof(array_ops::Float)*SIDE*SIDE*SIDE);
      cudaMemcpy(d0_ptr,u0.content.get(),sizeof(array_ops::Float)*SIDE*SIDE*SIDE,cudaMemcpyHostToDevice);
      d0 ->content = std::unique_ptr<Float[]>(d0_ptr);
      printf("hello3");
      cudaMemcpy(d1,&u1,sizeof(array_ops::Array),cudaMemcpyHostToDevice);
      Float *d1_ptr;
      cudaMalloc(&d1_ptr,sizeof(array_ops::Float)*SIDE*SIDE*SIDE);
      cudaMemcpy(d1_ptr,u1.content.get(),sizeof(array_ops::Float)*SIDE*SIDE*SIDE,cudaMemcpyHostToDevice);
      d1 ->content = std::unique_ptr<Float[]>(d1_ptr);
      printf("hello4");
      cudaMemcpy(d2,&u2,sizeof(array_ops::Array),cudaMemcpyHostToDevice);
      Float *d2_ptr;
      cudaMalloc(&d2_ptr,sizeof(array_ops::Float)*SIDE*SIDE*SIDE);
      cudaMemcpy(d2_ptr,u2.content.get(),sizeof(array_ops::Float)*SIDE*SIDE*SIDE,cudaMemcpyHostToDevice);
      d2 ->content = std::unique_ptr<Float[]>(d2_ptr);

      Float *res_d_ptr;
      cudaMalloc(&res_d_ptr,sizeof(array_ops::Float)*SIDE*SIDE*SIDE);
      res_d ->content = std::unique_ptr<Float[]>(res_d_ptr);

      forall_ix_snippet_cuda_x<<<1,1>>>(res_d,u_d, v_d, d0, d1, d2, c0, c1, c2, c3, c4, forall_kernel);

      cudaDeviceSynchronize();
      cudaMemcpy(&res_h,&res_d,SIDE*SIDE*SIDE*sizeof(array_ops::Array),cudaMemcpyDeviceToHost);

      return *res_h;
    }
};

template<class _kernel>
__global__ inline void forall_ix_snippet_cuda_x(array_ops::Array *res,array_ops::Array *u,
  array_ops::Array *v, array_ops::Array *u0, array_ops::Array *u1, array_ops::Array *u2, const array_ops::Float &c0,
  const array_ops::Float &c1, const array_ops::Float &c2, const array_ops::Float &c3, const array_ops::Float &c4, _kernel kernel)
{

  kernel(res,u,v,u0,u1,u2,c0,c1,c2,c3,c4);



//kernel(u_d, v_d, d0, d1, d2, c0, c1, c2, c3, c4);



//cudaMemcpy(&u,&u_d,SIDE*SIDE*SIDE*sizeof(Array),cudaMemcpyDeviceToHost);
//cudaMemcpy(&v,&v_d,SIDE*sizeof(Array),cudaMemcpyDeviceToHost);
//cudaMemcpy(&u0,&d0,SIDE*sizeof(Array),cudaMemcpyDeviceToHost);
//cudaMemcpy(&u1,&d1,SIDE*sizeof(Array),cudaMemcpyDeviceToHost);
//cudaMemcpy(&u2,&d2,SIDE*sizeof(Array),cudaMemcpyDeviceToHost);

}



inline void dumpsine(array_ops::Array &result) {
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