#include "dynamic-model.h"

#define small_eps (T)1.e-12

template< typename T >
void dynamic_model< T >::allocate_memory(const uniGrid2d< T >&grid)
{
    allocate(&damping_coef, grid.size);

    allocate(&inv_T, grid.size);
    allocate(&LM, &MM, grid.size);

    allocate(&lx, &ly, grid.size);
    allocate(&Lx, &Ly, grid.size);
    allocate(&tx, &ty, grid.size);
    allocate(&txc, &tyc, grid.size);
    allocate(&mx, &my, grid.size);
    allocate(&lm, &mm, grid.size);
    allocate(&l, &m, grid.size); // Maulik & San 2017
    
    allocate(&ssmx, &ssmy, grid.size); // for consistent output
    allocate(&hx, &hy, grid.size);     // for germano error
    if (mixed_model) {
        allocate(&ssmxc, &ssmyc, grid.size);
        allocate(&Ssmx, &Ssmy, grid.size);
    }

    allocate(&bx, &by, grid.size);
    allocate(&Bx, &By, grid.size);
    allocate(&bxc, &byc, grid.size);
    allocate(&Esub, grid.size);
    allocate(&Ediss, &Eback, grid.size);
    allocate(&nub, grid.size);
    
    allocate(&uc, &vc, grid.size);
    allocate(&wc, grid.size);
    allocate(&sx, &sy, grid.size);
    allocate(&sxc, &syc, grid.size);
    
    allocate(&Cs2_local, grid.size);
    allocate(&Cs2_x, &Cs2_y, grid.size);
}

template< typename T >
void dynamic_model< T >::clear()
{
    deallocate(damping_coef);

    deallocate(inv_T);
    deallocate(LM, MM);

    deallocate(lx, ly);
    deallocate(Lx, Ly);
    deallocate(tx, ty);
    deallocate(txc, tyc);
    deallocate(mx, my);
    deallocate(lm, mm);
    deallocate(l, m);
    
    deallocate(ssmx, ssmy);
    deallocate(hx, hy);
    if (mixed_model) {
        deallocate(ssmxc, ssmyc);
        deallocate(Ssmx, Ssmy);
    }

    deallocate(bx, by);
    deallocate(Bx, By);
    deallocate(bxc, byc);
    deallocate(Esub);
    deallocate(Ediss, Eback);
    deallocate(nub);
    
    deallocate(uc, vc);
    deallocate(wc);
    deallocate(sx, sy);
    deallocate(sxc, syc);
    
    deallocate(Cs2_local);
    deallocate(Cs2_x, Cs2_y);
}

template< typename T >
void dynamic_model< T >::init(int _viscosity_model,
int _averaging_method, bool _mixed_model, int _mixed_type, bool _negvisc_backscatter, bool _reynolds_backscatter, bool _adm_model,
int _adm_order, T _tf_width, T _bf_width, 
int _filter_iterations, int _leonard_scheme, T _lagrangian_time, T dt, const uniGrid2d< T >&grid, bool alloc_memory)
{
    viscosity_model   = _viscosity_model;
    averaging_method  = _averaging_method;
    filter_iterations = _filter_iterations;
    lagrangian_time   = _lagrangian_time;
    mixed_model = _mixed_model;
    mixed_type = _mixed_type;
    negvisc_backscatter = _negvisc_backscatter;
    reynolds_backscatter = _reynolds_backscatter;
    adm_model = _adm_model;
    adm_order = _adm_order;
    leonard_scheme = _leonard_scheme;

    if (negvisc_backscatter) {
        if (viscosity_model != bilap_smag && viscosity_model != bilap) {
            printf("backscatter error! \n");
            exit(0);            
        }
    }
    
    Csim = (T)1.0;

    //cback = (T)0.4 * sqrt((T)2.0); //Jansen time
    cback = (T)0.1; // my time
    //cback = (T)0.02;
    
    tf_width = _tf_width;
    bf_width = _bf_width;
    tb_width = sqrt(sqr(tf_width) + sqr(bf_width));
    
    // maximum Smagorinsky or leith constant
    switch (viscosity_model)
    {
        case lap:
            max_C2 = (T)1.0 / ((T)8.0 * dt);
            break;
        case lap_UV:
            max_C2 = (T)1.0 / ((T)8.0 * dt);
            break;
        case bilap:
            max_C2 = (T)1.0 / ((T)64.0 * dt);
            break;
        default:
            max_C2 = (T)1.0;
            break;
    }

    t_E_mean = (T)0.0;
    ssm_E_mean = (T)0.0;

    if (alloc_memory)
        allocate_memory(grid);
}

// needs vorticity gradient to be precomputed
template< typename T >
void dynamic_model< T >::compute_mx_my(T* w, T* u, T* v, bool mul_C2, const uniGrid2d< T >&grid)
{    
    // tx, ty, Tx, Ty, mx, my
    model_vector(tx, ty, sx, sy, u, v, w, grid.dx * bf_width, viscosity_model, grid);
    if (mul_C2) {
        mul(tx, Cs2_x, tx, grid.size);
        mul(ty, Cs2_y, ty, grid.size);
    }
    apply_filter(txc, tx, tf_width, grid);
    apply_filter(tyc, ty, tf_width, grid);
    model_vector(mx, my, sxc, syc, uc, vc, wc, grid.dx * tb_width, viscosity_model, grid);
    if (mul_C2) {
        mul(mx, Cs2_x, mx, grid.size);
        mul(my, Cs2_y, my, grid.size);
    }
    update(mx, -(T)1.0, txc, grid.size);
    update(my, -(T)1.0, tyc, grid.size);
}

// needs filtered fields to be precomputed
template< typename T >
void dynamic_model< T >::compute_lx_ly(T* w, T* u, T* v, T* psi, const uniGrid2d< T >&grid)
{
    // compute leonard vector
    compute_leonard_vector(lx, ly, w, u, v, tf_width, (T)0.0, (T)1.0, grid, leonard_scheme); // computes leonard vector for Germano identity  

    memcpy(Lx, lx, sizeof(T)*grid.size);
    memcpy(Ly, ly, sizeof(T)*grid.size);
    
    // update leonard vector if mixed model used
    if (mixed_model) 
    {   
        if (mixed_type == mixed_ssm)
            compute_leonard_vector(ssmx, ssmy, w, u, v, bf_width, (T)0.0, Csim, grid);
        else if (mixed_type == mixed_ngm) {
            //compute_ngm_vector(ssmx, ssmy, u, v, sx, sy, grid.dx * bf_width, grid);
            compute_ngm_vector_conservative(ssmx, ssmy, w, u, v, bf_width, (T)0.0, grid);
        }
        else
            assert(1 == 2 && "mixed model type is wrong");
        
        apply_filter(ssmxc, ssmx, tf_width, grid);
        apply_filter(ssmyc, ssmy, tf_width, grid);
        
        if (mixed_type == mixed_ssm)
            compute_leonard_vector(Ssmx, Ssmy, wc, uc, vc, tf_width, bf_width, Csim, grid);
        else if (mixed_type == mixed_ngm) {
            //compute_ngm_vector(Ssmx, Ssmy, uc, vc, sxc, syc, grid.dx * tb_width, grid);
            compute_ngm_vector_conservative(Ssmx, Ssmy, wc, uc, vc, tf_width, bf_width, grid);
        }
        else
            assert(1 == 2 && "mixed model type is wrong");

        assign(hx, (T)1.0, Ssmx, -(T)1.0, ssmxc, grid.size);
        assign(hy, (T)1.0, Ssmy, -(T)1.0, ssmyc, grid.size);
        
        // subtract from Leonard vector ssm model
        update(Lx, -(T)1.0, hx, grid.size);
        update(Ly, -(T)1.0, hy, grid.size);
    }
    
    if (negvisc_backscatter)
    {
        lap_nu_model(bx, by, w, nub, (T)1.0, grid);
        apply_filter(bxc, bx, tf_width, grid);
        apply_filter(byc, by, tf_width, grid);
        T delta_filters = tb_width / bf_width;
        lap_nu_model(Bx, By, wc, nub, delta_filters, grid);
        assign(Bx, (T)1.0, Bx, -(T)1.0, bxc, grid.size);
        assign(By, (T)1.0, By, -(T)1.0, byc, grid.size);

        update(Lx, -(T)1.0, Bx, grid.size);
        update(Ly, -(T)1.0, By, grid.size);
    }

    if (adm_model)
    {
        compute_adm_vector(bx, by, w, u, v, adm_order, bf_width, (T)0.0, grid);
        apply_filter(bxc, bx, tf_width, grid);
        apply_filter(byc, by, tf_width, grid);
        compute_adm_vector(Bx, By, wc, uc, vc, adm_order, tf_width, bf_width, grid);
        assign(Bx, (T)1.0, Bx, -(T)1.0, bxc, grid.size);
        assign(By, (T)1.0, By, -(T)1.0, byc, grid.size);

        update(Lx, -(T)1.0, Bx, grid.size);
        update(Ly, -(T)1.0, By, grid.size);
    }
}

template< typename T >
void dynamic_model< T >::compute_bx_by(T* w, T* u, T* v, T* psi, const uniGrid2d< T >&grid)
{
    if (reynolds_backscatter && (averaging_method == dyn2 || averaging_method == dyn2_ZE || averaging_method == dyn2_Morinishi))
    {
        T _Csim_back = (T)1.0;
        backscatter_ssm(bx, by, w, u, v, (T*)NULL, bf_width, (T)0.0, (T)0.0, _Csim_back, grid);
        apply_filter(bxc, bx, tf_width, grid);
        apply_filter(byc, by, tf_width, grid);
        backscatter_ssm(Bx, By, wc, uc, vc, (T*)NULL, tf_width, bf_width, (T)0.0, _Csim_back, grid);
        assign(Bx, (T)1.0, Bx, -(T)1.0, bxc, grid.size);
        assign(By, (T)1.0, By, -(T)1.0, byc, grid.size);
    }
}


template< typename T >
void dynamic_model< T >::compute_lagrangian_time(T* u, T* v, const uniGrid2d< T >&grid)
{
    T sxx[grid.size], sxy[grid.size], syy[grid.size], S_p[grid.size];
    
    strain_tensor(sxx, sxy, syy, u, v, grid);
    compute_S(S_p, sxx, sxy, syy, grid);

    assign(inv_T, (T)1.0 / lagrangian_time, S_p, grid.size);
}

template< typename T >
void dynamic_model< T >::init_lagrangian_eq(T* w, T* u, T* v, T* psi, const uniGrid2d< T >&grid)
{
    if (averaging_method == lagrangian)
    {
        // find initial eddy viscosity
        averaging_method = averaging_global;
        update_viscosity(w, u, v, psi, (T)0.0, false, (T)0.0, grid);

        // init as Meneveau
        assign(MM, (T)1.0, mm, grid.size);
        assign(LM, Cs2_mean, mm, grid.size);

        averaging_method = lagrangian;
    }
}

template< typename T >
void dynamic_model< T >::update_viscosity(T* w, T* u, T* v, T* psi, T dt, bool set_Cs, T Cs_in, const uniGrid2d< T >&grid)
{
    if (negvisc_backscatter) {
        T nub_T[grid.size];
        for (int i = 0; i < grid.size; i++)
        {
            nub_T[i] = - cback * grid.dx * sqrt(max(Esub[i],(T)0.0));
        }
        p_to_w(nub, nub_T, grid);
    }

    apply_filter(wc, w, tf_width, grid);
    apply_filter(uc, u, tf_width, grid);
    apply_filter(vc, v, tf_width, grid);

    // gradient of vorticity
    nabla(sx , sy , w, grid);
    nabla(sxc, syc, wc, grid);

    bool mul_C2 = false;
    compute_lx_ly(w, u, v, psi, grid); // lx, ly
    compute_mx_my(w, u, v, mul_C2, grid); // mx, my
    compute_bx_by(w, u, v, psi, grid); // bx, by if dyn2 
    
    // One-parameter dynamic model
    if (averaging_method == averaging_global || averaging_method == clipping || averaging_method == lagrangian) {
        scal_prod(lm, Lx, Ly, mx, my, grid);
        scal_prod(mm, mx, my, mx, my, grid);
    }

    // breaces for lagrangian case are needed because 
    // switch does not imply a new scope
    switch (averaging_method) {
        case averaging_global:
            assign(LM, average_xy(lm, grid), grid.size);
            assign(MM, average_xy(mm, grid), grid.size);
            break;
        case clipping:
            apply_filter_iter(LM, lm, filter_iterations, grid);
            apply_filter_iter(MM, mm, filter_iterations, grid);
            break;
        case lagrangian: {
                T rLM[grid.size], rMM[grid.size]; // rhs for Lagrange model  
                compute_lagrangian_time(u, v, grid);
                
                // sources
                mul(rLM, inv_T, lm, grid.size);
                mul(rMM, inv_T, mm, grid.size);

                upwind_advection_p(rLM, u, v, LM, grid);
                upwind_advection_p(rMM, u, v, MM, grid);
                
                // explicit Euler part
                update(LM, dt, rLM, grid.size);
                update(MM, dt, rMM, grid.size);
                
                // implicit Euler part (decay)
                for (int i = 0; i < grid.size; i++) {
                    LM[i] = LM[i] / ((T)1.0 + dt * inv_T[i]);
                    MM[i] = MM[i] / ((T)1.0 + dt * inv_T[i]);
                }
                break;
            }
    }

    if (averaging_method == averaging_global || averaging_method == clipping || averaging_method == lagrangian) {
        safe_division(Cs2_local, LM, MM, max_C2, grid);
    }

    if (reynolds_backscatter && averaging_method == dyn2) {
        dyn2_Cr_Cs_MSE(Cs2_mean, Csim_back, Lx, Ly, mx, my, Bx, By, max_C2, grid);
        assign(Cs2_local, Cs2_mean, grid.size);
    }

    if (reynolds_backscatter && averaging_method == dyn2_Morinishi) {
        dyn2_Cr_Cs_Morinishi(Cs2_mean, Csim_back, Lx, Ly, mx, my, Bx, By, max_C2, grid);
        assign(Cs2_local, Cs2_mean, grid.size);
    }

    if (averaging_method == Maulik2017) {
        compute_divergence_vector(l, Lx, Ly, grid);
        compute_divergence_vector(m, mx, my, grid);

        mul(lm, l, m, grid.size);
        mul(mm, m, m, grid.size);
        assign(LM, average_xy(lm, grid), grid.size);
        assign(MM, average_xy(mm, grid), grid.size);

        safe_division(Cs2_local, LM, MM, max_C2, grid);
    }

    if (reynolds_backscatter && averaging_method == dyn2_ZE) {
        T _psic[grid.size];
        T _pxc[grid.size], _pyc[grid.size];
        apply_filter(_psic, psi, tf_width, grid);
        nabla(_pxc, _pyc, _psic, grid);
        dyn2_Cs_Cr_ZE(Cs2_mean, Csim_back, Lx, Ly, mx, my, Bx, By, sxc, syc, _pxc, _pyc, grid);
        assign(Cs2_local, Cs2_mean, grid.size);
    }

    T Cs2_in;
    if (set_Cs) {
        if (viscosity_model == lap_smag) {
            Cs2_in = Cs_in * Cs_in;
        }
        if (viscosity_model == bilap_smag) {
            Cs2_in = Cs_in * Cs_in * Cs_in * Cs_in;
        }
        assign(Cs2_local, Cs2_in, grid.size);
    }

    p_to_v(Cs2_x, Cs2_local, grid);
    p_to_u(Cs2_y, Cs2_local, grid);
    Cs2_mean = average_xy(Cs2_local, grid);

    // --------- update viscosity model vector -------- //
    // ---- needed for backscatter and statistics ----- //
    mul_C2 = true;
    compute_mx_my(w, u, v, mul_C2, grid);
    if (reynolds_backscatter && (averaging_method == dyn2 || averaging_method == dyn2_ZE || averaging_method == dyn2_Morinishi)) {
        mul(bx, Csim_back, grid.size);
        mul(by, Csim_back, grid.size);
        mul(Bx, Csim_back, grid.size);
        mul(By, Csim_back, grid.size);
    }
    
    if (reynolds_backscatter && (averaging_method != dyn2 && averaging_method != dyn2_ZE && averaging_method != dyn2_Morinishi)) {
        Csim_back = (T)1.0;
        backscatter_ssm(bx, by, w, u, v, (T*)NULL, bf_width, (T)0.0, -(T)1.0, Csim_back, grid);
        Csim_back = compute_Cback(ssmx, ssmy, tx, ty, bx, by, w, psi, bf_width, grid);
        mul(bx, Csim_back, grid.size);
        mul(by, Csim_back, grid.size);
        apply_filter(bxc, bx, tf_width, grid);
        apply_filter(byc, by, tf_width, grid);
        backscatter_ssm(Bx, By, wc, uc, vc, (T*)NULL, tf_width, bf_width, -(T)1.0, Csim_back, grid);
        assign(Bx, (T)1.0, Bx, -(T)1.0, bxc, grid.size);
        assign(By, (T)1.0, By, -(T)1.0, byc, grid.size);
    }

    if (negvisc_backscatter) {
        update_subgrid_KE(w, u, v, dt, grid);
    }
}

template< typename T >
void dynamic_model< T >::update_subgrid_KE(T*w, T* u, T* v, T dt, const uniGrid2d< T >&grid)
{
    Ediss_Eback_bilap_smag(t_E_mean_flux, b_E_mean_flux, Ediss, Eback, w, u, v, Cs2_local, nub, grid.dx * bf_width, grid);

    T rhs_E[grid.size], Esubn[grid.size];
    assign(rhs_E, (T)0.0, grid.size);
    upwind_advection_p(rhs_E, u, v, Esub, grid);
    T min0, max0, min1, max1;
    min0 = min_xy(Esub, grid);
    max0 = max_xy(Esub, grid);

    assign(Esubn, (T)1.0, Esub, dt, rhs_E, grid.size);
    min1 = min_xy(Esubn, grid);
    max1 = max_xy(Esubn, grid);

    if (min1 < min0) {
        printf("minimum value is corrupted\n");
    }
    if (max1 > max0) {
        printf("maximum value is corrupted\n");
    }

    update(rhs_E, (T)1.0, Ediss, (T)1.0, Eback, grid.size);
    update(Esub, dt, rhs_E, grid.size);
    //apply_filter_iter(Esub, Esub, 1, grid);

    //for (int i = 0; i < grid.size; i++)
    //    Esub[i] = max(Esub[i],(T)0.0);
}

// this function to be applied needs only Cs2_x, Cs2_y
template< typename T >
void dynamic_model< T >::apply(T* wim, T* w, T* u, T* v, const uniGrid2d< T >&grid) const
{
    T _sx[grid.size], _sy[grid.size];
    T _tx[grid.size], _ty[grid.size];
    T _ssmx[grid.size], _ssmy[grid.size];
    T _bx[grid.size], _by[grid.size];

    nabla(_sx, _sy, w, grid);

    model_vector(_tx, _ty, _sx, _sy, u, v, w, grid.dx * bf_width, viscosity_model, grid);
    mul(_tx, Cs2_x, _tx, grid.size);
    mul(_ty, Cs2_y, _ty, grid.size);

    if (mixed_model) {
        if (mixed_type == mixed_ssm)
            compute_leonard_vector(_ssmx, _ssmy, w, u, v, bf_width, (T)0.0, Csim, grid);
        else if (mixed_type == mixed_ngm) {
            //compute_ngm_vector(_ssmx, _ssmy, u, v, _sx, _sy, grid.dx * bf_width, grid);
            compute_ngm_vector_conservative(_ssmx, _ssmy, w, u, v, bf_width, (T)0.0, grid);
        }
        else
            assert(1 == 2 && "mixed model type is wrong");

        //regularize_backscatter(damping_coef, _ssmx, _ssmy, w, u, v, grid);

        update(_tx, (T)1.0, _ssmx, grid.size);
        update(_ty, (T)1.0, _ssmy, grid.size);
    }

    if (negvisc_backscatter) {
        lap_nu_model(_bx, _by, w, nub, (T)1.0, grid);
        update(_tx, (T)1.0, _bx, grid.size);
        update(_ty, (T)1.0, _by, grid.size);
    }

    if (reynolds_backscatter)
    {
        T _Csim_back = Csim_back;
        backscatter_ssm(_bx, _by, w, u, v, (T*)NULL, bf_width, (T)0.0, -(T)1.0, _Csim_back, grid); // Csim in
        update(_tx, (T)1.0, _bx, grid.size);
        update(_ty, (T)1.0, _by, grid.size);
    }

    if (adm_model)
    {
        compute_adm_vector(_bx, _by, w, u, v, adm_order, bf_width, (T)0.0, grid);
        update(_tx, (T)1.0, _bx, grid.size);
        update(_ty, (T)1.0, _by, grid.size);
    }
    
    // dw/dt = ... - div(tx,ty);
    divergence_vector(wim, _tx, _ty, grid);
}

// apply AD filter of order p and one time step decrement (1-r)
// operator:
// dw/dt = r/dt * (F F_p^-1 - I) w
template< typename T >
void dynamic_model< T >::AD_filter(T* wim, T* w, const T dt, const int p, const T r, const uniGrid2d< T >&grid) const
{
    T Dw[grid.size]; // defiltered vorticity
    T FDw[grid.size]; // filtered-defiltered vorticity

    AD_iteration(Dw, w, p, bf_width, (T)0.0, grid);
    apply_filter(FDw, Dw, bf_width, grid);

    for (int i = 0; i < grid.size; i++)
        wim[i] += r / dt * (FDw[i] - w[i]);
}

// model dissipation, LM, germano
template< typename T >
void dynamic_model< T >::statistics(T* psi, T* w, T* u, T* v, T dt, const uniGrid2d< T >& grid)
{
    T Cs2_S[grid.size];
    T Cs2_S_max;
    T Cs2_S_mean;
    T nu2, nu4;
    compute_S_uv(Cs2_S, u, v, grid);
    mul(Cs2_S, Cs2_S, Cs2_local, grid.size);
    Cs2_S_max  = max_xy(Cs2_S, grid);
    Cs2_S_mean = average_xy(Cs2_S, grid);

    switch (viscosity_model) {
        case lap_smag:
            nu2 = Cs2_S_max * sqr(grid.dx * bf_width);
            CFL_EVM_max = (T)8. * dt * nu2 / sqr(grid.dx);
            nu2 = Cs2_S_mean * sqr(grid.dx * bf_width);
            CFL_EVM_mean = (T)8. * dt * nu2 / sqr(grid.dx);
            break;
        case bilap_smag:
            nu4 = Cs2_S_max * sqr(sqr(grid.dx * bf_width));
            CFL_EVM_max = (T)64. * dt * nu4 / sqr(sqr(grid.dx)); 
            nu4 = Cs2_S_mean * sqr(sqr(grid.dx * bf_width));
            CFL_EVM_mean = (T)64. * dt * nu4 / sqr(sqr(grid.dx)); 
            break;
        default:
            CFL_EVM_mean = 0.;
            CFL_EVM_max = 0.;
            break;
    }

    // get constant
    switch (viscosity_model) {
        case lap_leith:
            Cs = pow(Cs2_mean, (T)1.0/(T)3.0);
            break;
        case bilap_leith:
            Cs = pow(Cs2_mean, (T)1.0/(T)5.0);
            break;
        case lap_smag:
            Cs = pow(Cs2_mean, (T)1.0/(T)2.0);
            break;
        case bilap_smag:
            Cs = pow(Cs2_mean, (T)1.0/(T)4.0);
            break;
        case lap:
            Cs = (T)0.0; // do not need to be defined
            break;
        case bilap:
            Cs = (T)0.0;
            break;
        case lap_w_smag:
            Cs = pow(Cs2_mean, (T)1.0/(T)2.0);
            break;
        case bilap_w_smag:
            Cs = pow(Cs2_mean, (T)1.0/(T)4.0);
            break;
        case lap_UV_smag:
            Cs = pow(Cs2_mean, (T)1.0/(T)2.0);
            break;
        case lap_UV:
            Cs = (T)0.0;
            break;
        default:
            assert(1 == 2 && "model is wrong");
            break;
    }
    // local palinstrophy dissipation by models (dissipation > 0)
    T t_P[grid.size], ssm_P[grid.size], b_P[grid.size];
    T _lap_sx[grid.size], _lap_sy[grid.size];
    laplacian(_lap_sx, sx, grid);
    laplacian(_lap_sy, sy, grid);
    scal_prod(t_P, tx, ty, _lap_sx, _lap_sy, grid);
    scal_prod(b_P, bx, by, _lap_sx, _lap_sy, grid);
    scal_prod(ssm_P, ssmx, ssmy, _lap_sx, _lap_sy, grid);

    t_P_mean   =  average_xy(t_P, grid);
    b_P_mean   =  average_xy(b_P, grid);
    ssm_P_mean =  average_xy(ssm_P, grid);
    model_P_mean = t_P_mean + b_P_mean + ssm_P_mean;

    // local enstrophy dissipation by models (dissipation > 0)
    T t_Z[grid.size], ssm_Z[grid.size], b_Z[grid.size];
    scal_prod(t_Z, tx, ty, sx, sy, grid);
    scal_prod(b_Z, bx, by, sx, sy, grid);
    scal_prod(ssm_Z, ssmx, ssmy, sx, sy, grid);

    t_Z_mean   = - average_xy(t_Z, grid);
    b_Z_mean   = - average_xy(b_Z, grid);
    ssm_Z_mean = - average_xy(ssm_Z, grid);
    model_Z_mean = t_Z_mean + b_Z_mean + ssm_Z_mean;

    // local energy dissipation by models (dissipation > 0)
    T t_E[grid.size], ssm_E[grid.size], b_E[grid.size];
    T _px[grid.size], _py[grid.size];
    nabla(_px, _py, psi, grid);
    scal_prod(t_E, tx, ty, _px, _py, grid);
    scal_prod(b_E, bx, by, _px, _py, grid);
    scal_prod(ssm_E, ssmx, ssmy, _px, _py, grid);

    t_E_mean   = average_xy(t_E, grid);
    b_E_mean   = average_xy(b_E, grid);
    ssm_E_mean = average_xy(ssm_E, grid);
    model_E_mean = t_E_mean + b_E_mean + ssm_E_mean;    

    backscatter_rate = - b_E_mean / (t_E_mean+ssm_E_mean);
    
    // LM statistics (for non-lagrangian models equals to lm)
    // and backscatter statistics
    int i, j, idx;
    T E_neg = (T)0.0;
    T back_points = (T)0.0;
    T pure_diss  = (T)0.0;
    T pure_back  = (T)0.0;
    
    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
            if (Esub[idx] < - small_eps) {
                E_neg += (T)1.0;
            }
            if (averaging_method == lagrangian || averaging_method == clipping) {
                pure_diss += max(LM[idx], (T)0.0);
                pure_back -= min(LM[idx], (T)0.0);    
                if (LM[idx] < small_eps)
                    back_points += (T)1.0;
            } else {
                pure_diss += max(lm[idx], (T)0.0);
                pure_back -= min(lm[idx], (T)0.0);
                if (lm[idx] < small_eps)
                    back_points += (T)1.0;
            }
        }
    }
    
    LM_back_p = mpi_allreduce(back_points, MPI_SUM) * grid.dx * grid.dy / (grid.mpi_length * grid.mpi_width);
    LM_diss_to_back = mpi_allreduce(pure_diss, MPI_SUM) / mpi_allreduce(pure_back, MPI_SUM);
    
    E_neg_p = mpi_allreduce(E_neg, MPI_SUM) * grid.dx * grid.dy / (grid.mpi_length * grid.mpi_width);
    Esub_mean = average_xy(Esub, grid);

    T w2[grid.size];
    mul(w2, w, w, grid.size);
    T enstrophy = average_xy(w2, grid) * (T)0.5;
    Esub_time = sqrt(Esub_mean) / (enstrophy * grid.dx * cback);

    mean_lag_time = (T)1.0 / average_xy(inv_T, grid);

    germano_error(w, u, v, grid);
}

template< typename T >
void dynamic_model< T >::germano_error(T* w, T* u, T* v, const uniGrid2d< T >& grid)
{
    T ex[grid.size], ey[grid.size];
    // ssm subtracted
    assign(ex, (T)1.0, Lx, -(T)1.0, mx, grid.size);
    assign(ey, (T)1.0, Ly, -(T)1.0, my, grid.size);

    // because Reynolds backscatter is outside the Dynamic model,
    // but we want to be sure that Germano identit error still reasonable
    if (reynolds_backscatter) {
        update(ex, -(T)1.0, Bx, grid.size);
        update(ey, -(T)1.0, By, grid.size);
    }
    MSE_germano = (integrate_xy(ex,ex,grid) + integrate_xy(ey,ey,grid)) / (integrate_xy(lx,lx,grid) + integrate_xy(ly,ly,grid));

    // correlation for full model
    T modelx[grid.size], modely[grid.size];
    assign(modelx, (T)1.0, mx, (T)1.0, hx, (T)1.0, Bx, grid.size);
    assign(modely, (T)1.0, my, (T)1.0, hy, (T)1.0, By, grid.size);

    // just leonard vector, no ssm
    C_germano = (integrate_xy(lx,modelx,grid) + integrate_xy(ly,modely,grid)) 
              / sqrt(integrate_xy(lx,lx,grid) + integrate_xy(ly,ly,grid))
              / sqrt(integrate_xy(modelx,modelx,grid) + integrate_xy(modely,modely,grid));
    
    model_diss_to_l_diss_germano = (integrate_xy(modelx,sx,grid) + integrate_xy(modely,sy,grid)) / (integrate_xy(lx,sx,grid) + integrate_xy(ly,sy,grid));
}

// ---------- simple models ----------- //
template< typename T >
void dynamic_model< T >::set_simple_model(const int _viscosity_model, const T Cs2, const bool _mixed_model, const int _mixed_type, const uniGrid2d< T >&grid)
{
    bf_width = sqrt((T)6.0);
    viscosity_model = _viscosity_model;
    assign(Cs2_local, Cs2, grid.size);
    
    p_to_v(Cs2_x, Cs2_local, grid);
    p_to_u(Cs2_y, Cs2_local, grid);
    Cs2_mean = average_xy(Cs2_local, grid);

    mixed_model = _mixed_model;
    mixed_type = _mixed_type;
    negvisc_backscatter = false;
}

template< typename T >
T DSM_Pawar(T* w, T* u, T* v, T test_width, T base_width, 
    bool clipping, int averaging_method, const uniGrid2d< T >& grid)
{
    /*
    Represents dynamic model for momentum flux
    with possible clipping of the Leonard stress

    A priori analysis on deep learning of subgrid-scale parameterizations for Kraichnan turbulence
    Pawar, San, 2020
    */
    T lxx[grid.size], lyy[grid.size], lxy[grid.size];
    T uf[grid.size], vf[grid.size];
    T mxx[grid.size], myy[grid.size], mxy[grid.size];
    T mxxf[grid.size], myyf[grid.size], mxyf[grid.size];
    T Mxx[grid.size], Myy[grid.size], Mxy[grid.size];
    T LM[grid.size], MM[grid.size];
    T lx[grid.size], ly[grid.size];

    if (averaging_method == dyn_momentum_flux || averaging_method == dyn_momentum_forcing) {
        // Trace-free Leonard stress
        compute_leonard_tensor(lxx, lxy, lyy, u, v, test_width, grid);
    } else if (averaging_method == dyn_vorticity_flux || averaging_method == dyn_vorticity_forcing){
        compute_leonard_vector(lx, ly, w, u, v, test_width, (T)0.0, (T)1.0, grid, Leonard_PV_Z_scheme);
    }

    T mix_length = base_width * grid.dx;
    lap_UV_smagorinsky_model(mxx, mxy, myy, u, v, mix_length, grid);
    
    // Filtering of the base model
    apply_filter(mxxf, mxx, test_width, grid);
    apply_filter(myyf, myy, test_width, grid);
    apply_filter(mxyf, mxy, test_width, grid);

    // Smagorinsky model on the test level
    apply_filter(uf, u, test_width, grid);
    apply_filter(vf, v, test_width, grid);

    T tb_width = sqrt(sqr(test_width) + sqr(base_width));
    mix_length = tb_width * grid.dx;
    lap_UV_smagorinsky_model(Mxx, Mxy, Myy, uf, vf, mix_length, grid);

    update(Mxx, -(T)1.0, mxxf, grid.size);
    update(Myy, -(T)1.0, myyf, grid.size);
    update(Mxy, -(T)1.0, mxyf, grid.size);

    if (averaging_method == dyn_momentum_flux) {
        scal_prod_tensors(LM, lxx, lxy, lyy, Mxx, Mxy, Myy, grid);
        scal_prod_tensors(MM, Mxx, Mxy, Myy, Mxx, Mxy, Myy, grid);
    }
    
    if (averaging_method == dyn_momentum_forcing) {
        // Note that according to 
        // Vector level identity for dynamic subgrid scale modeling in large eddy simulation
        // Trace needs to be removed (as we does)

        T mx[grid.size], my[grid.size];
        // Compute momentum forcing from the momentum flux
        divergence_tensor(lx, ly, lxx, lxy, lyy, grid);
        divergence_tensor(mx, my, Mxx, Mxy, Myy, grid);
        // We swap x-y dimensions because scal_prod
        // is written for vorticity fluxes which 
        // are located in v and u points, correspondingly
        scal_prod(LM, ly, lx, my, mx, grid);
        scal_prod(MM, my, mx, my, mx, grid);
    }

    if (averaging_method == dyn_vorticity_flux) {
        T mx[grid.size], my[grid.size];

        curl_tensor(mx, my, Mxx, Mxy, Myy, grid);

        scal_prod(LM, lx, ly, mx, my, grid);
        scal_prod(MM, mx, my, mx, my, grid);
    }

    if (averaging_method == dyn_vorticity_forcing) {
        T mx[grid.size], my[grid.size];
        T l[grid.size], m[grid.size];

        curl_tensor(mx, my, Mxx, Mxy, Myy, grid);

        compute_divergence_vector(l, lx, ly, grid);
        compute_divergence_vector(m, mx, my, grid);

        mul(LM, l, m, grid.size);
        mul(MM, m, m, grid.size);
    }

    T epsilon = std::numeric_limits<T>::min();
    T Cs2_mean;
    Cs2_mean = min(average_xy(LM, grid) / (average_xy(MM, grid) + epsilon), (T)1.0);
    
    if (Cs2_mean < (T)0.0){
        printf("Mean value of Cs2<0\n");
    }

    Cs2_mean = max(Cs2_mean, (T)0.0);

    // it is local clipping followed by the averaging
    if (clipping) {
        for (int i = 0; i < grid.size; i++) {
            LM[i] = max(LM[i], (T)0.0);
        }
        Cs2_mean = min(average_xy(LM, grid) / (average_xy(MM, grid) + epsilon), (T)1.0);
    }

    return Cs2_mean;
}

template struct dynamic_model< float >;
template struct dynamic_model< double >;

template float DSM_Pawar(float* w, float* u, float* v, float test_width, 
    float base_width, bool clipping, int averaging_method, const uniGrid2d< float >& grid);
template double DSM_Pawar(double* w, double* u, double* v, double test_width, 
    double base_width, bool clipping, int averaging_method, const uniGrid2d< double >& grid);