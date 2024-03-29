#include "Reynolds-equation.h"

#define small_eps std::numeric_limits<T>::min()

template < typename T >
void Lagrangian_eq_struct<T>::init(const uniGrid2d< T >&grid)
{
    allocate(&fx, grid.size);
    allocate(&fy, grid.size);

    allocate(&fxp, grid.size);
    allocate(&fyp, grid.size);

    // Init SGS tensor to small non-zero value
    assign(fx, (T)0.0, grid.size);
    assign(fy, (T)0.0, grid.size);
}

template < typename T >
void Lagrangian_eq_struct<T>::clear() 
{
    deallocate(fx);
    deallocate(fy);

    deallocate(fxp);
    deallocate(fyp);
}

template < typename T >
void Lagrangian_eq_struct<T>::init_with_ZB(T* w, T* u, T* v, const T filter_width, const uniGrid2d< T >&grid)
{
    // Velocity gradients
    T D[grid.size], D_hat[grid.size];
    Velocity_gradients(D, D_hat, u, v, grid);

    ZB20_model_uv(fx, fy,
               w, D, D_hat, 
               filter_width, grid);   
}

template < typename T >
void Lagrangian_eq_struct<T>::RK_init(const uniGrid2d< T >&grid)
{
    memcpy(fxp, fx, grid.size * sizeof(T));
    memcpy(fyp, fy, grid.size * sizeof(T));
}

template < typename T >
void Lagrangian_eq_struct<T>::RK_step(T* w, T* u, T* v, T dt, const uniGrid2d< T >&grid)
{
    T D[grid.size], D_hat[grid.size];
    T rhs_x[grid.size], rhs_y[grid.size];

    // Compute instantaneous velocity gradients
    Velocity_gradients(D, D_hat, u, v, grid);

    assign(rhs_x, (T)0.0, grid.size);
    assign(rhs_y, (T)0.0, grid.size);

    Relaxation_to_ZB_uv(rhs_x, rhs_y,
                        fx, fy,
                        w, D, D_hat, 
                        (T)sqrt(6.0), grid);
    
    upwind_advection_u(rhs_x, u, v, fx, grid);
    upwind_advection_v(rhs_y, u, v, fy, grid);

    assign(fx, (T)1.0, fxp, dt, rhs_x, grid.size);
    assign(fy, (T)1.0, fyp, dt, rhs_y, grid.size);

    // ZB20_model_uv(fx, fy,
    //            w, D, D_hat, 
    //            (T)sqrt(6.0), grid);   
}

template<typename T>
void Lagrangian_eq_struct<T>::apply(T* wim, const uniGrid2d< T >&grid)
{
    /*
    Update the vorticity tendency with the current SGS tensor.
    */
    
    // Vorticity forcing
    T f[grid.size];
    velocity_to_vorticity(f, fx, fy, grid);
    update(wim, (T)1.0, f, grid.size);
}

template< typename T >
void Reynolds_eq_struct<T>::init(const uniGrid2d< T >&grid)
{
    allocate(&tau_xy, grid.size);
    allocate(&tau_dd, grid.size);
    allocate(&tau_tr, grid.size);

    allocate(&tau_xyp, grid.size);
    allocate(&tau_ddp, grid.size);
    allocate(&tau_trp, grid.size);

    // Init SGS tensor to small non-zero value
    assign(tau_xy, (T)0.0, grid.size);
    assign(tau_dd, (T)0.0, grid.size);
    assign(tau_tr, small_eps, grid.size);
}

template<typename T>
void Reynolds_eq_struct<T>::clear() 
{
    deallocate(tau_xy);
    deallocate(tau_dd);
    deallocate(tau_tr);

    deallocate(tau_xyp);
    deallocate(tau_ddp);
    deallocate(tau_trp);
}

template<typename T>
void Reynolds_eq_struct<T>::init_with_ZB(T* w, T* u, T* v, const T filter_width, const uniGrid2d< T >&grid)
{
    // Velocity gradients
    T D[grid.size], D_hat[grid.size];
    Velocity_gradients(D, D_hat, u, v, grid);

    ZB20_model(tau_xy, tau_dd, tau_tr, 
               w, D, D_hat, 
               filter_width, grid);
}

template<typename T>
void Reynolds_eq_struct<T>::diagnostics(T* Psi, T* w, T* u, T* v, const uniGrid2d< T >&grid)
{
    SGS_KE = average_xy(tau_tr, grid);
    
    // SGE KE production diagnostics
    T D[grid.size], D_hat[grid.size];
    T rhs_xy[grid.size], rhs_dd[grid.size], rhs_tr[grid.size];
    Velocity_gradients(D, D_hat, u, v, grid);
    // Start forming the RHS
    RHS_Production(rhs_xy, rhs_dd, rhs_tr,
                   tau_xy, tau_dd, tau_tr,
                   w, D, D_hat,           
                   grid);

    SGS_KE_prod = average_xy(rhs_tr, grid);

    T wim[grid.size], Prod[grid.size];
    assign(wim, (T)0.0, grid.size);
    apply(wim, grid);
    mul(Prod, wim, Psi, grid.size);

    // KE loss is minus the KE tendency
    KE_loss = average_xy(Prod, grid);
}

template<typename T>
void Reynolds_eq_struct<T>::RK_init(const uniGrid2d< T >&grid)
{
    /*
    This function initializes RK3 method.
    In RK3 the SGS tensor without index p
    will be used to compute RHS
    and the SGS tensor with index p
    to make a time step
    */
    memcpy(tau_xyp, tau_xy, grid.size * sizeof(T));
    memcpy(tau_ddp, tau_dd, grid.size * sizeof(T));
    memcpy(tau_trp, tau_tr, grid.size * sizeof(T));
}

template<typename T>
void Reynolds_eq_struct<T>::RK_step(T* w, T* u, T* v, T dt, const uniGrid2d< T >&grid)
{
    /*
    Here w, u, v and SGS tensor correspond to intermediate values 
    computed during RK3 iteration
    This function executes a single Euler-type time step
    which is used as a building block in RK3 method
    */

    // Velocity gradients
    T D[grid.size], D_hat[grid.size];
    // RHS for update of Reynolds equations
    T rhs_xy[grid.size], rhs_dd[grid.size], rhs_tr[grid.size];

    // Compute instantaneous velocity gradients
    Velocity_gradients(D, D_hat, u, v, grid);

    assign(rhs_xy, (T)0.0, grid.size);
    assign(rhs_dd, (T)0.0, grid.size);
    assign(rhs_tr, (T)0.0, grid.size);

    // Start forming the RHS
    // RHS_Production(rhs_xy, rhs_dd, rhs_tr,
    //                tau_xy, tau_dd, tau_tr, // Here we use current SGS tensor
    //                w, D, D_hat,            // Here we use current gradients 
    //                grid);

    Relaxation_to_ZB(rhs_xy, rhs_dd, rhs_tr,
                     tau_xy, tau_dd, tau_tr, 
                     w, D, D_hat, 
                    (T)sqrt(6.0), grid);
    
    upwind_advection_w(rhs_xy, u, v, tau_xy, grid);
    upwind_advection_p(rhs_dd, u, v, tau_dd, grid);
    upwind_advection_p(rhs_tr, u, v, tau_tr, grid);

    assign(tau_xy, (T)1.0, tau_xyp, dt, rhs_xy, grid.size);
    assign(tau_dd, (T)1.0, tau_ddp, dt, rhs_dd, grid.size);
    assign(tau_tr, (T)1.0, tau_trp, dt, rhs_tr, grid.size);

    // // Hard-coded ZB model, without relaxation
    // ZB20_model(tau_xy, tau_dd, tau_tr, 
    //            w, D, D_hat, 
    //            (T)sqrt(6.0), grid);
}

template<typename T>
void Reynolds_eq_struct<T>::apply(T* wim, const uniGrid2d< T >&grid)
{
    /*
    Update the vorticity tendency with the current SGS tensor.
    */
    
    // SGS stress tensor
    T Txx[grid.size], Tyy[grid.size], Txy[grid.size];
    for (int i = 0; i < grid.size; i++)
    {
        Txx[i] = (tau_tr[i] + tau_dd[i]);
        Tyy[i] = (tau_tr[i] - tau_dd[i]);
        Txy[i] =  tau_xy[i];
    }

    // Velocity forcing
    T fx[grid.size], fy[grid.size];
    divergence_tensor(fx, fy, Txx, Txy, Tyy, grid); // Minus inside

    // Vorticity forcing
    T f[grid.size];
    velocity_to_vorticity(f, fx, fy, grid);
    update(wim, (T)1.0, f, grid.size);
}

template struct Reynolds_eq_struct< float >;
template struct Reynolds_eq_struct< double >;

template struct Lagrangian_eq_struct< float >;
template struct Lagrangian_eq_struct< double >;
