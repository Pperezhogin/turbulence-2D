#pragma once

#include "unigrid2d.h"
#include "vecmath.h"
#include "dynamic-model-supplementary.h"
#include <math.h>
#include <stdio.h>

using namespace nse;

enum {set_zero_energy, set_ZB_energy};

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
    
    public:
        void init(const uniGrid2d< T >&grid);
        void clear();
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

    // Compute the RHS
    for (int i = 0; i < grid.size; i++)
    {
        rhs_xy[i] = -   D[i] * tau_tr_w[i] -     w[i] * tau_dd_w[i];
        rhs_dd[i] =   w_p[i] * tau_xy_p[i] - D_hat[i] * tau_tr[i];
        rhs_tr[i] = - D_p[i] * tau_xy_p[i] - D_hat[i] * tau_dd[i];
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