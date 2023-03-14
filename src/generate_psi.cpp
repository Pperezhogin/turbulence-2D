#include <cassert>
#include <string>
#include "generate_psi.h"

int main(int argc, char** argv) 
{
    omp_set_num_threads(OPENMP_CORES);

	MPI_Init(&argc, &argv);

    grid.set(
    (Real) 0.0, (Real) 0.0, domain_length, domain_width,
    domain_nx, domain_ny, domain_gcx, domain_gcy, 1);

    unsigned seed = time(NULL);
    MPI_Bcast(&seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    seed += grid.mpi_com.rank;
	srand(seed);
	for (int i = 0; i < grid.mpi_com.size; i++) {
		int rnk = grid.mpi_com.rank;
		if (rnk == i)
			printf("my id = %i, my seed = %i \n", rnk, seed);
		MPI_Barrier(grid.mpi_com.comm);
	}

    for (int idx = 1; idx < 51; idx++) {
        Real Psi[grid.size];
        power_exp_spectra(Psi, (Real)10.0, grid);
        mul(Psi, sqrt((Real)0.5), grid.size);
        write_binary_przgn("init_fields/psi-.nsx", Psi, grid, idx);
        if (grid.mpi_com.rank == 0) printf("idx = %i \n", idx);
    }

    MPI_Barrier(grid.mpi_com.comm);

    MPI_Finalize();

    return 0;
}