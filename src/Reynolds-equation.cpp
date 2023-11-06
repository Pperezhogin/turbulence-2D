#include "Reynolds-equation.h"

#define small_eps (T)1.e-12

template< typename T >
void SGS_KE_struct<T>::init(T* w, T* u, T* v, int _nu2_method, int _initial_cond, T _dt, const uniGrid2d< T >&grid)
{
    nu2_method = _nu2_method;
    dt = _dt;

    filter_width = sqrt((T)6.0) * grid.dx;

    allocate(&sgs_ke, &sfs_ke, grid.size);

    allocate(&nu_eddy, grid.size);
    allocate(&sqrt_nu_eddy, grid.size);
    allocate(&nu2, grid.size);

    allocate(&Ediss, &Eback, grid.size);

    allocate(&wim_diss, &wim_back, grid.size);

    switch (_initial_cond) {
        case set_zero_energy:
            assign(sgs_ke, (T)0.0, grid.size);
            break;
       case set_SFS_energy:
            estimate_sgs_ke(sgs_ke, w, u, v, filter_width, grid);
            break;
    }
}

template<typename T>
void SGS_KE_struct<T>::clear() 
{
    deallocate(sgs_ke, sfs_ke);
    deallocate(nu_eddy);
    deallocate(sqrt_nu_eddy);
    deallocate(nu2);
    deallocate(Ediss, Eback);
    deallocate(wim_diss, wim_back);
}

template<typename T>
void SGS_KE_struct<T>::update_KE(T* w, T* u, T* v, const uniGrid2d< T >&grid) 
{
    T eim[grid.size], sgs_ken[grid.size];
    T S[grid.size], sxx[grid.size], sxy[grid.size], syy[grid.size];
    T fx[grid.size], fy[grid.size];
    T Ediss_x[grid.size], Ediss_y[grid.size];
    T Ediss_x_p[grid.size], Ediss_y_p[grid.size];
    T nu2_w[grid.size];
    T Eback_p[grid.size], Eback_w[grid.size];

    assign(eim, (T)0.0, grid.size);
    upwind_advection_p(eim, u, v, sgs_ke, grid);

    // ----------------------- check transport --------------- //
    T min0, max0, min1, max1;
    min0 = min_xy(sgs_ke, grid);
    max0 = max_xy(sgs_ke, grid);

    assign(sgs_ken, (T)1.0, sgs_ke, dt, eim, grid.size);
    min1 = min_xy(sgs_ken, grid);
    max1 = max_xy(sgs_ken, grid);

    if (min1 < min0) {
        printf("minimum value is corrupted\n");
    }
    if (max1 > max0) {
        printf("maximum value is corrupted\n");
    }

    // ----------------------- Ediss ------------------------- //
    strain_tensor(sxx, sxy, syy, u, v, grid);
    compute_S(S, sxx, sxy, syy, grid);

    T C0 = (T)0.06 * sqr(sqr(grid.dx));
    for (int idx = 0; idx < grid.size; idx++) {
        nu_eddy[idx] = C0 * S[idx];
        sqrt_nu_eddy[idx] = sqrt(nu_eddy[idx]);
    }

    lap_UV_model(fx, fy, u, v, sqrt_nu_eddy, grid);

    mul(Ediss_x, fx, fx, grid.size); 
    mul(Ediss_y, fy, fy, grid.size);

    u_to_p(Ediss_x_p, Ediss_x, grid);
    v_to_p(Ediss_y_p, Ediss_y, grid);

    assign(Ediss, (T)1.0, Ediss_x_p, (T)1.0, Ediss_y_p, grid.size);

    m_Ediss_eq = average_xy(Ediss, grid);
    // ------------------------------------------------------ //

    // ------------------- negative viscosity ----------------//
    estimate_sgs_ke(sfs_ke, w, u, v, filter_width, grid);
    T ke_diff;
    switch (nu2_method) {
        case nu2_null:
            assign(nu2, (T)0.0, grid.size);
            break;
        case nu2_jansen:
            Cback = 0.11;
            for (int idx = 0; idx < grid.size; idx ++) {
                nu2[idx] = - Cback * grid.dx * sqrt(max(sgs_ke[idx], (T)0.0));
            }
            break;
        case nu2_est:
            Cback = 1.0;
            for (int idx = 0; idx < grid.size; idx ++) {
                nu2[idx] = - Cback * grid.dx * sqrt(max(sgs_ke[idx] - sfs_ke[idx], (T)0.0));
            }
            //apply_filter(nu2, nu2, sqrt((T)6.0), grid);
            break;
        case nu2_est_pm:
            Cback = 1.0;
            for (int idx = 0; idx < grid.size; idx ++) {
                ke_diff = sgs_ke[idx] - sfs_ke[idx];
                nu2[idx] = - sign(ke_diff) * Cback * grid.dx * sqrt(fabs(ke_diff));
            }
            //apply_filter(nu2, nu2, sqrt((T)6.0), grid);
            break;       
        default:
            assert(1 == 2 && "negative viscosity estimation si wrong");
            break;
    }

    m_nu2 = average_xy(nu2, grid);

    p_to_w(nu2_w, nu2, grid);

    for (int idx = 0; idx < grid.size; idx++) {
        Eback[idx]   = (T)2.0 * nu2[idx] * (sqr(sxx[idx]) + sqr(syy[idx]));
        Eback_w[idx] = (T)4.0 * nu2_w[idx] * sqr(sxy[idx]);
    }
    w_to_p(Eback_p, Eback_w, grid);
    update(Eback, (T)1.0, Eback_p, grid.size);
    m_Eback_eq = average_xy(Eback, grid);

    update(eim, (T)1.0, Ediss, (T)1.0, Eback, grid.size);

    m_SGS_KE = average_xy(sgs_ke, grid);
    m_SFS_KE = average_xy(sfs_ke, grid);

    T negative_ke[grid.size];
    for (int idx = 0; idx < grid.size; idx++) {
        negative_ke[idx] = - min(sgs_ke[idx], (T)0.0);
    }
    neg_SGS_KE = average_xy(negative_ke, grid);
    min_SGS_KE = min_xy(sgs_ke, grid);

    update(sgs_ke, dt, eim, grid.size);

    /*
    for (int idx = 0; idx < grid.size; idx++)
    {
        sgs_ke[idx] = max(sgs_ke[idx], (T)0.0);
    }
    */
}

template<typename T>
void SGS_KE_struct<T>::apply(T* wim, T* u, T* v, const uniGrid2d< T >&grid) 
{
    T fx[grid.size], fy[grid.size];
    T ffx[grid.size], ffy[grid.size];

    // -- dissipation -- //
    lap_UV_model(fx, fy, u, v, sqrt_nu_eddy, grid);
    lap_UV_model(ffx, ffy, fx, fy, sqrt_nu_eddy, grid);

    assign(ffx, -(T)1.0, ffx, grid.size);
    assign(ffy, -(T)1.0, ffy, grid.size);

    m_Ediss_ten = - integrate_xy(ffx, ffy, u, v, grid) / (grid.mpi_length * grid.mpi_width);

    velocity_to_vorticity(wim_diss, ffx, ffy, grid);

    lap_UV_model(fx, fy, u, v, nu2, grid);

    m_Eback_ten = - integrate_xy(fx, fy, u, v, grid) / (grid.mpi_length * grid.mpi_width);
    
    velocity_to_vorticity(wim_back, fx, fy, grid);

    update(wim, (T)1.0, wim_diss, (T)1.0, wim_back, grid.size);
}


template struct SGS_KE_struct< float >;
template struct SGS_KE_struct< double >;
