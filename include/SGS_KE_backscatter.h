#pragma once

#include "unigrid2d.h"
#include "vecmath.h"
#include "dynamic-model-supplementary.h"
#include <math.h>
#include <stdio.h>

using namespace nse;

enum {nu2_null, nu2_jansen, nu2_est, nu2_est_pm}; // methods 
enum {set_zero_energy, set_SFS_energy};

template< typename T >
struct SGS_KE_struct
{   
    int nu2_method;
    T dt;
    T Cback;

    T filter_width;

    // in p points
    T *sgs_ke; // prognostic KE
    T *sfs_ke; // diagnostic KE

    // in p points
    T *sqrt_nu_eddy;
    T *nu_eddy;
    T *nu2;

    // in p points, Ediss > 0, Eback <= 0 (in average)
    T *Ediss, *Eback;

    // vorticity tendencies
    T *wim_diss, *wim_back;

    T m_Ediss_eq, m_Ediss_ten; // energy dissipation in SGS KE equation and in resulting tendency
    T m_Eback_eq, m_Eback_ten; 
    T m_SGS_KE;
    T m_SFS_KE;
    T m_nu2;
    T neg_SGS_KE;
    T min_SGS_KE;
    
    public:
        void init(T* w, T* u, T* v, int _nu2_method, int _initial_cond, T _dt, const uniGrid2d< T >&grid);
        void clear();
        void update_KE(T* w, T* u, T* v, const uniGrid2d< T >&grid);
        void apply(T* wim, T* u, T* v, const uniGrid2d< T >&grid);
};