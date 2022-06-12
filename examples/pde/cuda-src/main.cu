#include <vector>

#include <omp.h>

#include "gen/examples/pde/mg-src/pde-cuda.cuh"
#include "base.cuh"

//typedef array_ops::Shape Shape;
typedef array_ops<float>::Array Array;
typedef array_ops<float>::Index Index;
typedef examples::pde::mg_src::pde_cuda::BasePDEProgram BasePDEProgram;
typedef examples::pde::mg_src::pde_cuda::PDEProgramDNF PDEProgramDNF;

int main() {
    size_t steps = 50;
    Array u0, u1, u2;
    dumpsine(u0);
    dumpsine(u1);
    dumpsine(u2);


    BasePDEProgram pde;
    PDEProgramDNF pde_dnf;
    double begin = omp_get_wtime();

    for (size_t i = 0; i < steps; ++i) {
        pde_dnf.step(u0, u1, u2);//, S_NU, S_DX, S_DT);
        std::cout << u0[PAD0 * PADDED_S1 * PADDED_S2 + PAD1 * PADDED_S2 + PAD2] << " "
                  << u1[PAD0 * PADDED_S1 * PADDED_S2 + PAD1 * PADDED_S2 + PAD2] << " "
                  << u2[PAD0 * PADDED_S1 * PADDED_S2 + PAD1 * PADDED_S2 + PAD2] << std::endl;
    }

    double end = omp_get_wtime();

    std::cout << end - begin << "[s] elapsed with sizes ("
              << S0 << ", "
              << S1 << ", "
              << S2 << ") with padding ("
              << PAD0 << ", "
              << PAD1 << ", "
              << PAD2 << ") on "
              << NB_CORES << " threads for "
              << steps << " steps" << std::endl;
}
