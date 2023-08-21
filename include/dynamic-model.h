#pragma once

#include "unigrid2d.h"
#include "vecmath.h"
#include "dynamic-model-supplementary.h"
#include <math.h>
#include <stdio.h>
#include "nse-out2d.h"

using namespace nse;

template< typename T >
struct dynamic_model 
{   
    int viscosity_model;
    int averaging_method;
    bool mixed_model;
    int mixed_type; 
    bool negvisc_backscatter;
    bool adm_model;
    int adm_order;
    T Csim;

    T* damping_coef; // regularization coefficient for backscatter
    
    T tf_width; // width of test filter to construct Germano identity, in units of mesh step
    T bf_width; // width of base filter.
    T tb_width; // width of product test * base filters
    int filter_iterations; // Number of filter iterations in lm and mm in clipping method
    T max_C2; // maximum smagorinsky constant, for stability
    int leonard_scheme;

    // Lagrangian model data
    T lagrangian_time; // lagrangian time in |S|^-1 units. // It is assumed that 
    // lmmm_time_smag  = 1.5 |S|^-1
    // lmmm_time_leith = 1.0 |S|^-1
    // lmmm_time_bilap_smag = 0.5 |S|^-1    
    T *inv_T; // inverse lagrangian time in DNS units
    T *LM, *MM; // lagrangian averaged variables (or =lm,mm if no lagrangian averaging used)
    T mean_lag_time; // in model units
    
    // Dynamic procedure data
    T *lx, *ly; // leonard vector, in v and u points, correspondingly
    T *Lx, *Ly; // leonard vector minus SSM model
    T *tx, *ty; // viscosity model vector at base level
    T *txc, *tyc; // filtered model
    T *mx, *my; // model vector divided by "Smagorinsky" constant, in v and u points, correspondingly
    T *lm, *mm; // num. and den. for estimation of viscosity, in p point
    T *l, *m; // num. and den. in Dynamic model in divergence formulation (Maulik2017)
    
    // Mixed model data
    T *ssmx, *ssmy; // ssm model on base level
    T *ssmxc, *ssmyc; // filtered ssm model on base level
    T *Ssmx, *Ssmy; // ssm model on test level
    T* hx, *hy; // difference between ssm models (multiplied by Csim)

    // Negvisc backscatter data
    T cback; //energy to viscosity constant
    T *bx, *by; // backscatter on base level
    T *bxc, *byc;
    T *Bx, *By; // backscatter on test level or difference between models
    T *Esub; // subgrid kinetic energy
    T *Ediss, *Eback; // KE exchange with subgrid scales, Ediss > 0, Eback < 0
    T *nub; // negative viscosity <=0, in w points
    T E_neg_p; // percent of points with negative energy
    T Esub_mean; // mean subgrid energy
    T Esub_time; // time of subgrid energy decay (due to backscatter)

    // ssm backscatter data
    bool reynolds_backscatter;
    T Csim_back; // similarity constant in Reynolds term
    T backscatter_rate;
    
    T *uc, *vc, *wc; // filtered dynamic variables
    T *sx, *sy, *sxc, *syc; // gradient of vorticity
    
    T *Cs2_local; // generalized viscosity, i.e. Cs^2, Cl^3, Cs^4
    T *Cs2_x, *Cs2_y; //interpolated viscosity
    T Cs2_mean; // mean Cs2_local
    T Cs; // Cs -> (nu_mean)^(1/2), Cl -> (nu_mean)^(1/3), Cs -> (nu_mean)^(1/4)
    
    // dissipation statistics
    T t_P_mean, ssm_P_mean, b_P_mean, model_P_mean; // domain-averaged palinstrophy dissipation
    T t_Z_mean, ssm_Z_mean, b_Z_mean, model_Z_mean; // domain-averaged enstrophy dissipation
    T t_E_mean, ssm_E_mean, b_E_mean, model_E_mean; // domain-averaged energy dissipation
    T b_E_mean_flux, t_E_mean_flux; // energy fluxes in galilean-invariant form

    // numerator statistics
    T LM_back_p; // percent of <=0 points in numerator (after all averaging done)
    T LM_diss_to_back; // no backscatter -> infinity; 50% backscatter -> 1    

    // viscosity stability constraint
    T CFL_EVM_max, CFL_EVM_mean;

    // Germano identity statistics
    T MSE_germano;
    T C_germano; // corellation coefficient for full model
    T model_diss_to_l_diss_germano; //dissipation of enstrophy by full model to leonard dissipation in germano identity
    
    public:
        void init(int _viscosity_model,
            int _averaging_method, bool _mixed_model, int _mixed_type, bool _negvisc_backscatter, bool _reynolds_backscatter, bool _adm_model,
            int _adm_order, T _tf_width, T _bf_width, 
            int _filter_iterations, int _leonard_scheme, T _lagrangian_time, T dt, const uniGrid2d< T >&grid, bool alloc_memory=true);
        void clear();
        void allocate_memory(const uniGrid2d< T >&grid);
        void init_lagrangian_eq(T* w, T* u, T* v, T* psi, const uniGrid2d< T >&grid);
        void update_viscosity(T* w, T* u, T* v, T* psi, T dt, bool set_Cs, T Cs_in, const uniGrid2d< T >&grid); // computes all tensors and update LM, MM
        void apply(T* wim, T* w, T* u, T* v, const uniGrid2d< T >&grid) const;
        void AD_filter(T* wim, T* w, const T dt, const int p, const T r, const uniGrid2d< T >&grid) const;
        void statistics(T* psi, T* w, T* u, T* v, T dt, const uniGrid2d< T >& grid) ;// update viscosity-model and computes statistics
        void set_simple_model(const int _viscosity_model, const T Cs2, const bool _mixed_model, const int _mixed_type, const uniGrid2d< T >&grid);

    private:
        void compute_mx_my(T* w, T* u, T* v, bool mul_C2, const uniGrid2d< T >&grid);
        void compute_lx_ly(T* w, T* u, T* v, T* psi, const uniGrid2d< T >&grid);
        void compute_bx_by(T* w, T* u, T* v, T* psi, const uniGrid2d< T >&grid);
        void compute_lagrangian_time(T* u, T* v, const uniGrid2d< T >&grid);
        void germano_error(T* w, T* u, T* v, const uniGrid2d< T >& grid);
        void update_subgrid_KE(T* w, T* u, T* v, T dt, const uniGrid2d< T >&grid);
};

template < typename T >
	T DSM_Pawar(T* w, T* u, T* v, T test_width, T base_width, 
    bool clipping, int averaging_method, const uniGrid2d< T >& grid);
