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

void test_tensor_differentiation(Real* w, Real* u, Real* v, const uniGrid2d< Real >& grid){
    Real Sxx[grid.size], Sxy[grid.size], Syy[grid.size];
    Real Sx[grid.size], Sy[grid.size], sx[grid.size], sy[grid.size];
    Real Dx[grid.size], Dy[grid.size], dx[grid.size], dy[grid.size];
    Real S[grid.size], s[grid.size];

    // S_ij
    strain_tensor(Sxx, Sxy, Syy, u, v, grid);
    mul(Sxx, 2.0, grid.size); mul(Syy, 2.0, grid.size); mul(Sxy, 2.0, grid.size);

    // \partial x_j w
    curl_tensor(Sx, Sy, Sxx, Sxy, Syy, grid);

    // laplacian(u_i)
    // Change sign because it is present in divergence
    divergence_tensor(Dx, Dy, Sxx, Sxy, Syy, grid);
    mul(Dx, -1.0, grid.size); mul(Dy, -1.0, grid.size);

    // laplacian(w)
    compute_divergence_vector(S, Sx, Sy, grid);

    // Compute reference fields

    //  \partial x_j w
    nabla(sx, sy, w, grid);

    // laplacian(u_i)
    laplacian(dx, u, grid);
    laplacian(dy, v, grid);

    // laplacian(w)
    laplacian(s, w, grid);

    Real err;
    err = error_relative(Sx, sx, grid) + error_relative(Sy, sy, grid);
    if(grid.mpi_com.rank==0) printf("Relative Linf error in curl_tensor = %E \n", err);

    err = error_relative(Dx, dx, grid) + error_relative(Dy, dy, grid);
    if(grid.mpi_com.rank==0) printf("Relative Linf error in divergence_tensor = %E \n", err);

    err = error_relative(S, s, grid);
    if(grid.mpi_com.rank==0) printf("Relative Linf Error in compute_divergence_vector = %E \n", err);
}

void test_Smagorinsky_model(Real* w, Real* u, Real* v, const uniGrid2d< Real >& grid){
    Real Mx[grid.size], My[grid.size];
    Real Mxx[grid.size], Mxy[grid.size], Myy[grid.size];
    Real sx[grid.size], sy[grid.size];
    Real mx[grid.size], my[grid.size];
    Real M[grid.size], m[grid.size];

    // - |S| nabla omega
    nabla(sx, sy, w, grid);
    lap_smagorinsky_model(mx, my, sx, sy, u, v, 1.0, grid);
    compute_divergence_vector(m, mx, my, grid);

    lap_UV_smagorinsky_model(Mxx, Mxy, Myy, u, v, 1.0, grid);
    curl_tensor(Mx, My, Mxx, Mxy, Myy, grid);
    compute_divergence_vector(M, Mx, My, grid);

    Real err;
    err = error_relative(Mx, mx, grid) + error_relative(My, my, grid);
    if(grid.mpi_com.rank==0) printf("Smagorinsky: Relative Linf error in vorticity flux = %E \n", err);

    err = error_relative(M, m, grid);
    if(grid.mpi_com.rank==0) printf("Smagorinsky: Relative Linf error in vorticity flux divergence = %E \n", err);
}

void test_Leonard_stress(Real* w, Real* u, Real* v, const uniGrid2d< Real >& grid, int scheme = 0){
    Real lx[grid.size], ly[grid.size];
    Real Lxx[grid.size], Lxy[grid.size], Lyy[grid.size];
    Real Lx[grid.size], Ly[grid.size];
    Real L[grid.size], l[grid.size];
    
    compute_leonard_vector(lx, ly, w, u, v, sqrt(6.0), 0.0, 1.0, grid, scheme);
    compute_divergence_vector(l, lx, ly, grid);

    compute_leonard_tensor(Lxx, Lxy, Lyy, u, v, sqrt(6.0), grid);
    curl_tensor(Lx, Ly, Lxx, Lxy, Lyy, grid);
    compute_divergence_vector(L, Lx, Ly, grid);

    Real err, corr;
    err = error_relative(Lx, lx, grid) + error_relative(Ly, ly, grid);
    corr = (correlation(Lx, lx, grid) + correlation(Ly, ly, grid)) * 0.5;
    if(grid.mpi_com.rank==0) printf("Leonard %i: In vorticity flux Linf error = %.3f, correlation =  %.3f \n", scheme, err, corr);

    err = error_relative(L, l, grid);
    corr = correlation(L, l, grid);
    if(grid.mpi_com.rank==0) printf("Leonard %i: In vorticity flux divergence Linf error = %.3f, correlation =  %.3f \n", scheme, err, corr);
}

void test_momentum_vorticity_transport(Real *w, Real* u, Real* v, bool trace_free, int scheme, const uniGrid2d< Real >& grid) {
    Real u_center[grid.size], v_center[grid.size], u_corner[grid.size], v_corner[grid.size];
    Real uu[grid.size], vv[grid.size], uv[grid.size];
    Real uu_side[grid.size], vv_side[grid.size];
    Real UW[grid.size], VW[grid.size];
    Real u_v[grid.size], v_u[grid.size], w_u[grid.size], w_v[grid.size], uw[grid.size], vw[grid.size];
    Real L[grid.size], l[grid.size];

    // Momentum flux
    u_to_w(u_corner, u, grid);
    v_to_w(v_corner, v, grid);
    mul(uv, u_corner, v_corner, grid.size);

    if (scheme == 0) {
        u_to_p(u_center, u, grid);
        v_to_p(v_center, v, grid);
        
        mul(uu, u_center, u_center, grid.size);
        mul(vv, v_center, v_center, grid.size);
    }

    if (scheme == 1) {
        mul(uu_side, u, u, grid.size);
        mul(vv_side, v, v, grid.size);

        u_to_p(uu, uu_side, grid);
        v_to_p(vv, vv_side, grid);
    }

    if (trace_free) {
        Real half_trace;
        for (int i = 0; i < grid.size; i++) {
            half_trace = 0.5 * (uu[i] + vv[i]);
            uu[i] -= half_trace;
            vv[i] -= half_trace;
        }
    }

    curl_tensor(UW, VW, uu, uv, vv, grid);
    compute_divergence_vector(L, UW, VW, grid);

    // Vorticity flux
    u_to_v(u_v, u, grid);
    v_to_u(v_u, v, grid);

    w_to_u(w_u, w, grid);
    w_to_v(w_v, w, grid);
    
    mul(uw, u_v, w_v, grid.size);
    mul(vw, v_u, w_u, grid.size);
    compute_divergence_vector(l, uw, vw, grid);

    Real err, corr;
    err = error_relative(UW, uw, grid) + error_relative(VW, vw, grid);
    corr = (correlation(UW, uw, grid) + correlation(VW, vw, grid)) * 0.5;
    if(grid.mpi_com.rank==0) printf("Transport %i %i: In vorticity flux Linf error = %.3f, correlation =  %.3f \n", int(trace_free), scheme, err, corr);

    err = error_relative(L, l, grid);
    corr = correlation(L, l, grid);
    if(grid.mpi_com.rank==0) printf("Transport %i %i: In vorticity flux divergence Linf error =  %.3f, correlation =  %.3f \n", int(trace_free), scheme, err, corr);
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

    // ------------ test momentum components of dynamic models ------------- //
    get_random_field(psi, grid);
    grid.mpi_com.exchange_halo(psi,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);
    velocity_stag(u, v, psi, grid);
    grid.mpi_com.exchange_halo(u, v,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);
    vorticity(w, u, v, grid);
    
    if (grid.mpi_com.rank == 0) printf("\nTest on random noise: \n");
    test_tensor_differentiation(w, u, v, grid);
    test_Smagorinsky_model(w, u, v, grid);
    test_Leonard_stress(w, u, v, grid, 0);
    test_Leonard_stress(w, u, v, grid, 1);
    test_momentum_vorticity_transport(w, u, v, true, 0, grid);
    test_momentum_vorticity_transport(w, u, v, false, 0, grid);
    // test_momentum_vorticity_transport(w, u, v, true, 1, grid);
    // test_momentum_vorticity_transport(w, u, v, false, 1, grid);

    /*
    Real uf[grid.size], vf[grid.size], wf[grid.size];
    apply_filter(wf, w, 2.0, grid);
    apply_filter(uf, u, 2.0, grid);
    apply_filter(vf, v, 2.0, grid);

    if (grid.mpi_com.rank == 0) printf("\nTest on filtered fields: \n");
    test_tensor_differentiation(w, u, v, grid);
    test_Smagorinsky_model(wf, uf, vf, grid);
    test_Leonard_stress(wf, uf, vf, grid);
    test_momentum_vorticity_transport(w, u, v, true, 0, grid);
    test_momentum_vorticity_transport(w, u, v, false, 0, grid);
    test_momentum_vorticity_transport(w, u, v, true, 1, grid);
    test_momentum_vorticity_transport(w, u, v, false, 1, grid);
    */
    MPI_Finalize();

    return 0;
}