#pragma once

#include "unigrid2d.h"
#include "vecmath.h"
#include "dynamic-model-supplementary.h"
#include <math.h>
#include <stdio.h>

using namespace nse;
#define small_eps std::numeric_limits<T>::min()

template < typename T >
struct Lagrangian_eq_struct
{
    T* fx; // Zonal acceleration
    T* fy; // Meridional acceleration

    T* fxp; // Zonal acceleration in past time moment
    T* fyp; // Meridional acceleration in past time moment

    public:
        void init(const uniGrid2d< T >&grid);
        void clear();
        void init_with_ZB(T* w, T* u, T* v, const T filter_width, const uniGrid2d< T >&grid);
        void RK_init(const uniGrid2d< T >&grid);
        void RK_step(T* w, T* u, T* v, T dt, const uniGrid2d< T >&grid);
        void apply(T* wim, const uniGrid2d< T >&grid);
};

template< typename T >
struct Reynolds_eq_struct
{   
    // Components of the SGS stress tensor
    T* tau_xy; // The off-diagonal component
    T* tau_dd; // The diagonal deviatoric component
    T* tau_tr; // The trace component (exactly, half-trace)

    // SGS stress in past time moment
    // Used to implement RK3 time-integration scheme
    T* tau_xyp;
    T* tau_ddp;
    T* tau_trp;

    // Diagnostics
    T SGS_KE;
    T SGS_KE_prod; // Production of SGS KE in Reynolds equation
    T KE_loss;     // Loss of resolved KE; ideally, SGS_KE_prod = KE_loss
    
    public:
        void init(const uniGrid2d< T >&grid);
        void clear();
        void init_with_ZB(T* w, T* u, T* v, const T filter_width, const uniGrid2d< T >&grid);
        void diagnostics(T* Psi, T* w, T* u, T* v, const uniGrid2d< T >&grid);
        void RK_init(const uniGrid2d< T >&grid);
        void RK_step(T* w, T* u, T* v, T dt, const uniGrid2d< T >&grid);
        void apply(T* wim, const uniGrid2d< T >&grid);
};

template < typename T >
void RHS_Production(T* rhs_xy, T* rhs_dd, T* rhs_tr,
                    T* tau_xy, T* tau_dd, T* tau_tr, 
                    T* w, T* D, T* D_hat, 
                    const uniGrid2d< T >& grid)
{
    /*
    Compute the RHS of the Reynolds stress equation related to the
    Production term, i.e. the only directly computable term.
    The production term is given by:
    -(tau_{ki} partial_k u_j + tau_{kj} partial_k u_i)
    
    Transforming the gradients of the resolved flow to:
    D     = partial_y u + partial_x v (corner)
    D_hat = partial_x u - partial_y v (center)
    w     = partial_x v - partial_y u (corner)

    And transforming the SGS stress tensor to:
    tau_{xy}                -> tau_xy (corner)
    (tau_{xx} - tau_{yy})/2 -> tau_dd (center)
    (tau_{xx} + tau_{yy})/2 -> tau_tr (center)

    The evolution equations becomem, where the only RHS is the 
    Production term:
    d tau_xy / dt = - D * tau_tr - w     * tau_dd (corner)
    d tau_dd / dt =   w * tau_xy - D_hat * tau_tr (center)
    d tau_tr / dt = - D * tau_xy - D_hat * tau_dd (center)

    Note that the sign convention for SGS tensor is:
    d u_i / dt = - partial_j tau_{ij}
    */

    // Interpolated to the corner
    T tau_dd_w[grid.size], tau_tr_w[grid.size];
    p_to_w(tau_dd_w, tau_dd, grid);
    p_to_w(tau_tr_w, tau_tr, grid);

    // Interpolate to the center
    T tau_xy_p[grid.size];
    T D_p[grid.size], w_p[grid.size];

    w_to_p(tau_xy_p, tau_xy, grid);
    w_to_p(D_p, D, grid);
    w_to_p(w_p, w, grid);

    // Better numerical scheme accounting for the staggered grid
    T D_tau_xy[grid.size], D_tau_xy_p[grid.size];
    mul(D_tau_xy, D, tau_xy, grid.size);
    w_to_p(D_tau_xy_p, D_tau_xy, grid);

    T w_tau_xy[grid.size], w_tau_xy_p[grid.size];
    mul(w_tau_xy, w, tau_xy, grid.size);
    w_to_p(w_tau_xy_p, w_tau_xy, grid);


    // Compute the RHS
    for (int i = 0; i < grid.size; i++)
    {
        rhs_xy[i] += - D[i] * max(tau_tr_w[i], small_eps) - w[i] * tau_dd_w[i];
        rhs_dd[i] +=   w_tau_xy_p[i] - D_hat[i] * max(tau_tr[i], small_eps);
        rhs_tr[i] += - D_tau_xy_p[i] - D_hat[i] * tau_dd[i];
    }
}

template < typename T >
void Velocity_gradients(T* D, T* D_hat, 
                        T* u, T* v, 
                        const uniGrid2d< T >& grid)
{
    /*
    D     = partial_y u + partial_x v (corner)
    D_hat = partial_x u - partial_y v (center)

    This function transforms the strain-rate tensor components
    to our components of the strain-rate (D and D_hat)
    */

    T sxx[grid.size], syy[grid.size], sxy[grid.size];
    strain_tensor(sxx, sxy, syy, u, v, grid);
    for (int i = 0; i < grid.size; i++)
    {
        D_hat[i] = sxx[i] - syy[i];
        D[i] = (T)2.0 * sxy[i];
    }
}

template < typename T >
void ZB20_model(T* tau_xy, T* tau_dd, T* tau_tr, 
                T* w, T* D, T* D_hat, 
                const T filter_width, const uniGrid2d< T >& grid)
{
    /*
    This function computes the SGS stress tensor components
    using the Zanna Bolton (2020) model
    */

    T D_p[grid.size], w_p[grid.size];
    w_to_p(D_p, D, grid);
    w_to_p(w_p, w, grid);

    T D_hat_w[grid.size];
    p_to_w(D_hat_w, D_hat, grid);

    // Different numerical scheme
    T Dw[grid.size], Dw_p[grid.size];
    mul(Dw, D, w, grid.size);
    w_to_p(Dw_p, Dw, grid);

    T C;
    C = grid.dx * grid.dy * filter_width * filter_width / 24.0;
    for (int i=0; i < grid.size; i++)
    {
        // Compute the SGS tensor components
        tau_xy[i] =   C * D_hat_w[i] * w[i];
        tau_dd[i] = - C * D_p[i] * w_p[i]; //Energy-non-conserving-scheme
        //tau_dd[i] = - C * Dw_p[i]; // Energy-conserving-scheme
        tau_tr[i] =   max(C * (T)0.5 * (w_p[i] * w_p[i] + D_hat[i] * D_hat[i] + D_p[i] * D_p[i]), small_eps);
    }
}

template < typename T >
void ZB20_model_uv(T* fx, T* fy,
                T* w, T* D, T* D_hat, 
                const T filter_width, const uniGrid2d< T >& grid)
{
    T tau_xy[grid.size], tau_dd[grid.size], tau_tr[grid.size];
    ZB20_model(tau_xy, tau_dd, tau_tr, w, D, D_hat, filter_width, grid);

    T Txx[grid.size], Tyy[grid.size], Txy[grid.size];
    for (int i = 0; i < grid.size; i++)
    {
        Txx[i] = (tau_tr[i] + tau_dd[i]);
        Tyy[i] = (tau_tr[i] - tau_dd[i]);
        Txy[i] =  tau_xy[i];
    }
    divergence_tensor(fx, fy, Txx, Txy, Tyy, grid); // Minus inside
}

template < typename T >
void Relaxation_to_ZB(T* rhs_xy, T* rhs_dd, T* rhs_tr,
                      T* tau_xy, T* tau_dd, T* tau_tr, 
                      T* w, T* D, T* D_hat, 
                      const T filter_width, const uniGrid2d< T >& grid)
{
    /*
    The solution to Reynolds equation has exponentially
    growing modes. I.e., it does not admit a meaningful equailibrium.

    Here we impose a meaningful equilibrium with the relaxation 
    towards the Zanna-Bolton (2020) model.
    */
 
    T ZB_xy[grid.size], ZB_dd[grid.size], ZB_tr[grid.size];
    ZB20_model(ZB_xy, ZB_dd, ZB_tr, w, D, D_hat, filter_width, grid);

    T S[grid.size], S_w[grid.size];
    T D_p[grid.size];

    w_to_p(D_p, D, grid);

    for (int i = 0; i < grid.size; i++)
    {
        S[i] = sqrt(D_p[i] * D_p[i] + D_hat[i] * D_hat[i]);
    }
    p_to_w(S_w, S, grid);

    // As relaxation in energy equation is an uncontrollable source of energy,
    // We keep for a while only relaxation for two other terms
    for (int i = 0; i < grid.size; i++)
    {
            rhs_xy[i] += (ZB_xy[i] - tau_xy[i]) * S_w[i];
            rhs_dd[i] += (ZB_dd[i] - tau_dd[i]) * S[i];
            rhs_tr[i] += (ZB_tr[i] - tau_tr[i]) * S[i];
    }
}

template < typename T >
void Relaxation_to_ZB_uv(T* rhs_x, T* rhs_y,
                         T* fx, T* fy,
                         T* w, T* D, T* D_hat, 
                         const T filter_width, const uniGrid2d< T >& grid)
{

    T ZB_x[grid.size], ZB_y[grid.size];
    ZB20_model_uv(ZB_x, ZB_y, w, D, D_hat, filter_width, grid);

    T S[grid.size], S_u[grid.size], S_v[grid.size];
    T D_p[grid.size];

    w_to_p(D_p, D, grid);

    for (int i = 0; i < grid.size; i++)
    {
        S[i] = sqrt(D_p[i] * D_p[i] + D_hat[i] * D_hat[i]);
    }
    p_to_u(S_u, S, grid);
    p_to_v(S_v, S, grid);

    // As relaxation in energy equation is an uncontrollable source of energy,
    // We keep for a while only relaxation for two other terms
    for (int i = 0; i < grid.size; i++)
    {
            rhs_x[i] += (ZB_x[i] - fx[i]) * S_u[i];
            rhs_y[i] += (ZB_y[i] - fy[i]) * S_v[i];
    }
}