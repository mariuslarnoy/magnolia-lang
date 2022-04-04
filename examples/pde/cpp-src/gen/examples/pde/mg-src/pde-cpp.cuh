#pragma once

#include "base.cuh"

#include <cassert>


namespace examples {
  namespace pde {
    namespace mg_src {
      namespace pde_cpp {

        struct PDEProgram {

          PDEProgram() {}

          struct _two {

            _two() {}

            template < typename T >
              inline __host__ __device__ T operator()() {
                T o;
                PDEProgram::two0(o);
                return o;
              };
          };

          struct _one {

            _one() {}

            template < typename T >
              inline __host__ __device__ T operator()() {
                T o;
                PDEProgram::one0(o);
                return o;
              };
          };

          private:
            array_ops __array_ops;
          public:
            typedef array_ops::Offset Offset;
          private:
            static inline __host__ __device__ void one0(PDEProgram::Offset & o) {
              array_ops __array_ops;
              o = __array_ops.one_offset();
            };
          public:
            typedef array_ops::Nat Nat;
          typedef array_ops::Index Index;
          typedef array_ops::Float Float;
          struct _three {
            inline __host__ __device__ PDEProgram::Float operator()() {
              array_ops __array_ops;

              return __array_ops.three_float();
            };
          };

          struct _unary_sub {

            _unary_sub() {}

            inline __host__ __device__ PDEProgram::Float operator()(const PDEProgram::Float & f) {
              array_ops __array_ops;

              return __array_ops.unary_sub(f);
            };
            inline __host__ __device__ PDEProgram::Offset operator()(const PDEProgram::Offset & o) {
              array_ops __array_ops;
              return __array_ops.unary_sub(o);
            };
          };

          private:
            static inline __host__ __device__ void one0(PDEProgram::Float & o) {
              array_ops __array_ops;
              o = __array_ops.one_float();
            };
          static inline __host__ __device__ void two0(PDEProgram::Float & o) {
            array_ops __array_ops;
            o = __array_ops.two_float();
          };
          public:
            typedef array_ops::Axis Axis;
          struct _rotate_ix {

            _rotate_ix() {}

            inline __host__ __device__ PDEProgram::Index operator()(const PDEProgram::Index & ix,
              const PDEProgram::Axis & axis,
                const PDEProgram::Offset & o) {
              array_ops __array_ops;
              return __array_ops.rotate_ix(ix, axis, o);
            };
          };

          struct _zero {

            _zero() {}

            inline __host__ __device__ PDEProgram::Axis operator()() {
              array_ops __array_ops;
              return __array_ops.zero_axis();
            };
          };

          private:
            static inline __host__ __device__ void one0(PDEProgram::Axis & o) {
              array_ops __array_ops;
              o = __array_ops.one_axis();
            };
          static inline __host__ __device__ void two0(PDEProgram::Axis & o) {
            array_ops __array_ops;
            o = __array_ops.two_axis();
          };
          public:
            typedef array_ops::Array Array;
          struct _binary_add {

            _binary_add() {}

            inline __host__ __device__ PDEProgram::Float operator()(const PDEProgram::Float & lhs,
              const PDEProgram::Float & rhs) {
              array_ops __array_ops;
              return __array_ops.binary_add(lhs, rhs);
            };
            inline __host__ __device__ PDEProgram::Array operator()(const PDEProgram::Float & lhs,
              const PDEProgram::Array & rhs) {
              array_ops __array_ops;
              return __array_ops.binary_add(lhs, rhs);
            };
            inline __host__ __device__ PDEProgram::Array operator()(const PDEProgram::Array & lhs,
              const PDEProgram::Array & rhs) {
              array_ops __array_ops;
              return __array_ops.binary_add(lhs, rhs);
            };
          };

          struct _binary_sub {

            _binary_sub() {}

            inline __host__ __device__ PDEProgram::Float operator()(const PDEProgram::Float & lhs,
              const PDEProgram::Float & rhs) {
              array_ops __array_ops;
              return __array_ops.binary_sub(lhs, rhs);
            };
            inline __host__ __device__ PDEProgram::Array operator()(const PDEProgram::Float & lhs,
              const PDEProgram::Array & rhs) {
              array_ops __array_ops;
              return __array_ops.binary_sub(lhs, rhs);
            };
            inline __host__ __device__ PDEProgram::Array operator()(const PDEProgram::Array & lhs,
              const PDEProgram::Array & rhs) {
              array_ops __array_ops;
              return __array_ops.binary_sub(lhs, rhs);
            };
          };

          struct _div {

            _div() {}

            inline __host__ __device__ PDEProgram::Float operator()(const PDEProgram::Float & num,
              const PDEProgram::Float & den) {
              array_ops __array_ops;
              return __array_ops.div(num, den);
            };
            inline __host__ __device__ PDEProgram::Array operator()(const PDEProgram::Float & num,
              const PDEProgram::Array & den) {
              array_ops __array_ops;
              return __array_ops.div(num, den);
            };
          };


          struct _mul {

            _mul() {}

            inline __host__ __device__ PDEProgram::Float operator()(const PDEProgram::Float & lhs,
              const PDEProgram::Float & rhs) {
              array_ops __array_ops;
              return __array_ops.mul(lhs, rhs);
            };
            inline __host__ __device__ PDEProgram::Array operator()(const PDEProgram::Float & lhs,
              const PDEProgram::Array & rhs) {
              array_ops __array_ops;
              return __array_ops.mul(lhs, rhs);
            };
            inline __host__ __device__ PDEProgram::Array operator()(const PDEProgram::Array & lhs,
              const PDEProgram::Array & rhs) {
              array_ops __array_ops;
              return __array_ops.mul(lhs, rhs);
            };
          };

          struct _psi {

            _psi() {}

            inline __host__ __device__ PDEProgram::Float operator()(const PDEProgram::Index & ix,
              const PDEProgram::Array & array) {
              array_ops __array_ops;
              return __array_ops.psi(ix, array);
            };
          };

          struct _rotate {

            _rotate() {}

            inline __host__ __device__ PDEProgram::Array operator()(const PDEProgram::Array & a,
              const PDEProgram::Axis & axis,
                const PDEProgram::Offset & o) {
              array_ops __array_ops;
              return __array_ops.rotate(a, axis, o);
            };
          };

          _zero zero = _zero();
          _one one = _one();
          _two two = _two();
          _three three = _three();          
          _unary_sub unary_sub = _unary_sub();
          _rotate_ix rotate_ix = _rotate_ix();
          _binary_add binary_add = _binary_add();
          _binary_sub binary_sub = _binary_sub();
          _div div = _div();
          _mul mul = _mul();
          _psi psi = _psi();
          _rotate rotate = _rotate();
          
          struct _snippet_ix {
            
            _snippet_ix() {}

            _zero zero = _zero();
            _one one = _one();
            _two two = _two();
            _three three = _three();          
            _unary_sub unary_sub = _unary_sub();
            _rotate_ix rotate_ix = _rotate_ix();
            _binary_add binary_add = _binary_add();
            _binary_sub binary_sub = _binary_sub();
            _div div = _div();
            _mul mul = _mul();
            _psi psi = _psi();
            _rotate rotate = _rotate();


            inline __host__ __device__ PDEProgram::Float operator()(const PDEProgram::Array & u,
              const PDEProgram::Array & v,
                const PDEProgram::Array & u0,
                  const PDEProgram::Array & u1,
                    const PDEProgram::Array & u2,
                      const PDEProgram::Float & c0,
                        const PDEProgram::Float & c1,
                          const PDEProgram::Float & c2,
                            const PDEProgram::Float & c3,
                              const PDEProgram::Float & c4,
                                const PDEProgram::Index & ix) {
              PDEProgram::Axis zero_ax = zero.operator()();
              PDEProgram::Offset one_of = one.operator()<Offset>();
              PDEProgram::Axis two_ax = two.operator()< Axis > ();
              PDEProgram::Float result = binary_add.operator()(psi.operator()(ix, u),
               mul.operator()(c4,binary_sub.operator()(mul.operator()(c3,
                 binary_sub.operator()(mul.operator()(c1,
                 binary_add.operator()(binary_add.operator()
                    (binary_add.operator()(binary_add.operator()
                        (binary_add.operator()(psi.operator()
                            (rotate_ix.operator()(ix, zero_ax, 
                                unary_sub.operator()(one_of)), v),
                                psi.operator()(rotate_ix.operator()(ix, 
                                zero_ax, one_of), v)), psi.operator()
                                (rotate_ix.operator()(ix, 
                                one.operator() < Axis > (), 
                                unary_sub.operator()(one_of)), v)),
                                psi.operator()(rotate_ix.operator()
                                (ix, one.operator() < Axis > (), one_of), v)),
                                psi.operator()(rotate_ix.operator()(ix, two_ax, unary_sub.operator()(one_of)), v)), psi.operator()(rotate_ix.operator()(ix, two_ax, one_of), v))), mul.operator()(mul.operator()(three(), c2), psi.operator()(ix, u0)))), mul.operator()(c0, binary_add.operator()(binary_add.operator()(mul.operator()(binary_sub.operator()(psi.operator()(rotate_ix.operator()(ix, zero_ax, one_of), v), psi.operator()(rotate_ix.operator()(ix, zero_ax, unary_sub.operator()(one_of)), v)), psi.operator()(ix, u0)), mul.operator()(binary_sub.operator()(psi.operator()(rotate_ix.operator()(ix, one.operator() < Axis > (), one_of), v), psi.operator()(rotate_ix.operator()(ix, one.operator() < Axis > (), unary_sub.operator()(one_of)), v)), psi.operator()(ix, u1))), mul.operator()(binary_sub.operator()(psi.operator()(rotate_ix.operator()(ix, two_ax, one_of), v), psi.operator()(rotate_ix.operator()(ix, two_ax, unary_sub.operator()(one_of)), v)), psi.operator()(ix, u2)))))));
              return result;
            };
          };
          forall_ops < PDEProgram::Array, PDEProgram::Axis, PDEProgram::Float, PDEProgram::Index, PDEProgram::Nat, PDEProgram::Offset, PDEProgram::_snippet_ix > __forall_ops = forall_ops<PDEProgram::Array, PDEProgram::Axis, PDEProgram::Float, PDEProgram::Index, PDEProgram::Nat, PDEProgram::Offset, PDEProgram::_snippet_ix>();


          struct _forall_ix_snippet_cuda {

            _forall_ix_snippet_cuda() {}

            forall_ops < PDEProgram::Array, PDEProgram::Axis, PDEProgram::Float, PDEProgram::Index, PDEProgram::Nat, PDEProgram::Offset, PDEProgram::_snippet_ix > __forall_ops = forall_ops<PDEProgram::Array, PDEProgram::Axis, PDEProgram::Float, PDEProgram::Index, PDEProgram::Nat, PDEProgram::Offset, PDEProgram::_snippet_ix>();
            
            inline __host__ PDEProgram::Array operator()(const PDEProgram::Array & u,
              const PDEProgram::Array & v,
                const PDEProgram::Array & u0,
                  const PDEProgram::Array & u1,
                    const PDEProgram::Array & u2,
                      const PDEProgram::Float & c0,
                        const PDEProgram::Float & c1,
                          const PDEProgram::Float & c2,
                            const PDEProgram::Float & c3,
                              const PDEProgram::Float & c4) {
              printf("forall_ix_snippet_cuda\n");

              return __forall_ops.forall_ix_snippet_cuda(u, v, u0, u1, u2, c0, c1, c2, c3, c4);
            };
          };

          struct _forall_ix_snippet {

            _forall_ix_snippet() {}

            forall_ops < PDEProgram::Array, PDEProgram::Axis, PDEProgram::Float, PDEProgram::Index, PDEProgram::Nat, PDEProgram::Offset, PDEProgram::_snippet_ix > __forall_ops = forall_ops<PDEProgram::Array, PDEProgram::Axis, PDEProgram::Float, PDEProgram::Index, PDEProgram::Nat, PDEProgram::Offset, PDEProgram::_snippet_ix>();

            inline __host__ __device__ PDEProgram::Array operator()(const PDEProgram::Array & u,
              const PDEProgram::Array & v,
                const PDEProgram::Array & u0,
                  const PDEProgram::Array & u1,
                    const PDEProgram::Array & u2,
                      const PDEProgram::Float & c0,
                        const PDEProgram::Float & c1,
                          const PDEProgram::Float & c2,
                            const PDEProgram::Float & c3,
                              const PDEProgram::Float & c4) {
              return __forall_ops.forall_ix_snippet(u, v, u0, u1, u2, c0, c1, c2, c3, c4);
            };
          };
          struct _snippet {

            _snippet() {}

            _forall_ix_snippet_cuda forall_ix_snippet_cuda = _forall_ix_snippet_cuda();

            inline __host__ void operator()(PDEProgram::Array & u,
              const PDEProgram::Array & v,
                const PDEProgram::Array & u0,
                  const PDEProgram::Array & u1,
                    const PDEProgram::Array & u2,
                      const PDEProgram::Float & c0,
                        const PDEProgram::Float & c1,
                          const PDEProgram::Float & c2,
                            const PDEProgram::Float & c3,
                              const PDEProgram::Float & c4) {
              u = forall_ix_snippet_cuda.operator()(u, v, u0, u1, u2, c0, c1, c2, c3, c4);
            };
          };
          struct _step {

            _step() {}


            _snippet_ix snippet_ix = _snippet_ix();
            _snippet snippet = _snippet();

            _zero zero = _zero();
            _one one = _one();
            _two two = _two();
            _three three = _three();          
            _unary_sub unary_sub = _unary_sub();
            _rotate_ix rotate_ix = _rotate_ix();
            _binary_add binary_add = _binary_add();
            _binary_sub binary_sub = _binary_sub();
            _div div = _div();
            _forall_ix_snippet forall_ix_snippet = _forall_ix_snippet();
            _mul mul = _mul();
            _psi psi = _psi();
            _rotate rotate = _rotate();

            inline __host__ __device__ void operator()(PDEProgram::Array & u0, PDEProgram::Array & u1, PDEProgram::Array & u2,
              const PDEProgram::Float & nu,
                const PDEProgram::Float & dx,
                  const PDEProgram::Float & dt) {
              PDEProgram::Float one_f = one.operator() < Float > ();
              PDEProgram::Float _2 = two.operator() < Float > ();
              PDEProgram::Float c0 = div.operator()(div.operator()(one_f, _2), dx);
              PDEProgram::Float c1 = div.operator()(div.operator()(one_f, dx), dx);
              PDEProgram::Float c2 = div.operator()(div.operator()(_2, dx), dx);
              PDEProgram::Float c3 = nu;
              PDEProgram::Float c4 = div.operator()(dt, _2);
              PDEProgram::Array v0 = u0;
              PDEProgram::Array v1 = u1;
              PDEProgram::Array v2 = u2;
              snippet.operator()(v0, u0, u0, u1, u2, c0, c1, c2, c3, c4);
              snippet.operator()(v1, u1, u0, u1, u2, c0, c1, c2, c3, c4);
              snippet.operator()(v2, u2, u0, u1, u2, c0, c1, c2, c3, c4);
              snippet.operator()(u0, v0, u0, u1, u2, c0, c1, c2, c3, c4);
              snippet.operator()(u1, v1, u0, u1, u2, c0, c1, c2, c3, c4);
              snippet.operator()(u2, v2, u0, u1, u2, c0, c1, c2, c3, c4);
            };
          };

          _step step = _step();
        };
      } // examples
    } // pde
  } // mg_src
} // pde_cpp