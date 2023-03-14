#include <cassert>
#include <string>
#include "test1.h"

#define epsilon (Real)1.e-10

Real function_launcher(Real* wc, Real* w, const uniGrid2d< Real >& grid, 
            void func(Real* wc, Real* w, const uniGrid2d< Real >& grid), std::string str)
{
    Real err;
    null(wc, grid.size);
    func(wc, w, grid);
    err = error_inside(wc, w, grid); // compare with 1. inside
    if(grid.mpi_com.rank==0) printf("error %s  = %E \n", str.c_str(), err);

    return err;
}

int main(int argc, char** argv) 
{
    omp_set_num_threads(OPENMP_CORES);

	MPI_Init(&argc, &argv);

    grid.set(
    (Real) 0.0, (Real) 0.0, domain_length, domain_width,
    domain_nx, domain_ny, domain_gcx, domain_gcy, 1);

    srand (grid.mpi_com.rank);

    if (grid.mpi_com.rank == 0) printf("mpi communicator size = %i \n", grid.mpi_com.size);

    Real *w, *wc;
    Real err;
    allocate(&w, &wc, grid.size);

    set_ones_inside(w, grid);

    {
        null(wc, grid.size);

        top_hat(wc, w, (Real)2.0, grid);

        err = error_inside(wc, w, grid); // compare with 1. inside

        if(grid.mpi_com.rank==0) printf("error top_hat = %E \n", err);

        assert(err < epsilon);

        apply_filter_iter(wc, wc, 1, grid);

        err = error_inside(wc, w, grid); // compare with 1. inside

        if(grid.mpi_com.rank==0) printf("error top_hat_iter = %E \n", err);

        apply_filter(wc, wc, (Real)2.0, grid);

        err = error_inside(wc, w, grid); // compare with 1. inside

        if(grid.mpi_com.rank==0) printf("error apply_filter = %E \n", err);
    }

    assert(function_launcher(wc, w, grid, u_to_v, "u_to_v") < epsilon);
    assert(function_launcher(wc, w, grid, v_to_u, "v_to_u") < epsilon);
    assert(function_launcher(wc, w, grid, w_to_u, "w_to_u") < epsilon);
    assert(function_launcher(wc, w, grid, w_to_v, "w_to_v") < epsilon);
    assert(function_launcher(wc, w, grid, u_to_w, "u_to_w") < epsilon);
    assert(function_launcher(wc, w, grid, v_to_w, "v_to_w") < epsilon);
    assert(function_launcher(wc, w, grid, u_to_p, "u_to_p") < epsilon);
    assert(function_launcher(wc, w, grid, v_to_p, "v_to_p") < epsilon);
    assert(function_launcher(wc, w, grid, p_to_u, "p_to_u") < epsilon);
    assert(function_launcher(wc, w, grid, p_to_v, "p_to_v") < epsilon);
    assert(function_launcher(wc, w, grid, w_to_p, "w_to_p") < epsilon);
    assert(function_launcher(wc, w, grid, p_to_w, "p_to_w") < epsilon);

    Real psi[grid.size], u[grid.size], v[grid.size], sx[grid.size], sy[grid.size], 
    px[grid.size], py[grid.size];

    get_random_field(psi, grid);
    grid.mpi_com.exchange_halo(psi,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);
    velocity_stag(u, v, psi, grid);
    grid.mpi_com.exchange_halo(u, v,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);
    vorticity(w, u, v, grid);

    // here check fft solver
    fft_mpi_poisson2d_data< Real > fft_data;
    fft_data.init(grid);
    poisson_fft(psi, w, fft_data, grid);      

    nabla(sx, sy, w, grid);
    nabla(px, py, psi, grid);

    Real w2[grid.size], w2_p[grid.size];
    Real ps[grid.size];
    Real ave_1, ave_2;

    mul(w2, w, w, grid.size);
    w_to_p(w2_p, w2, grid);
    ave_1 =   average_xy(w2_p, grid);
    scal_prod(ps, px, py, sx, sy, grid);
    ave_2 = - average_xy(ps, grid);
    err = fabs(ave_1 - ave_2) / fabs(ave_1);
    if(grid.mpi_com.rank==0) printf("error in gradient and poisson  = %E \n", err);

    Real wim[grid.size], Esub[grid.size];
    for (int i = 0; i < grid.size; i++)
        wim[i] = 0;
    get_random_field(Esub, grid);
    upwind_advection_p(wim, u, v, Esub, grid);
    Real mean_adv = average_xy(wim, grid);
    if (grid.mpi_com.rank == 0) printf("mean value of upwind = %E \n", mean_adv);

    MPI_Finalize();

    return 0;
}