#pragma once

#include "unigrid2d.h"
#include "vecmath.h"
#include "fourier-methods.h"
#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <limits>

#define MESH_HAT_FILTER
//#define SPECTRAL_GAUSS_FILTER
//#define ADM_LAYTON
#define ADM_CHOW

using namespace nse;

enum {lap, bilap, lap_leith, lap_smag, bilap_smag, lap_w_smag, bilap_w_smag, bilap_leith}; // viscosity models
enum {averaging_global, clipping, lagrangian, dyn2, dyn2_ZE, dyn2_Morinishi, Maulik2017}; // averaging methods. dyn2 stands for MSE of 2 constants. Applicable only for reynolds
enum {mixed_ssm, mixed_ngm};

//////////////////////////////////////////////////////////////
//////////// -------- exchanges inside --------- /////////////
//////////////////////////////////////////////////////////////


// -- averaging methods -- //
// pointwise scalar product of vectors, result in p point
template < typename T >
void scal_prod(T* ls, T* lx, T* ly, T* sx, T* sy, const uniGrid2d< T >& grid)
{
    int i, j, idx;
    
    T lsx[grid.size], lsy[grid.size];

    mul(lsx, lx, sx, grid.size);
    mul(lsy, ly, sy, grid.size);
    
    grid.mpi_com.exchange_halo(lsx, lsy,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);
        
    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
            
            ls[idx] = (lsx[idx] + lsx[idx + 1] + lsy[idx] + lsy[idx + grid.ny]) * (T)0.5;
        }
    }
}

template< typename T >
T integrate_xy(const T* u, const uniGrid2d< T >& grid)
{
    T sum_local = (T)0.0;
    T s;

    int i, j, idx;

    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
                sum_local += u[idx];
        }
    }

    s = mpi_allreduce(sum_local, MPI_SUM);

    return s * grid.dx * grid.dy;
}

template< typename T >
T integrate_xy(const T* lx, const T* ly, const uniGrid2d< T >& grid)
{
    T sum_local = (T)0.0;
    T s;

    int i, j, idx;

    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
                sum_local += lx[idx] * ly[idx];
        }
    }

    s = mpi_allreduce(sum_local, MPI_SUM);

    return s * grid.dx * grid.dy;
}

template< typename T >
T integrate_xy(const T* lx, const T* ly, const T* sx, const T* sy, const uniGrid2d< T >& grid)
{
    T sum_local = (T)0.0;
    T s;

    int i, j, idx;

    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
                sum_local += lx[idx] * sx[idx] + ly[idx] * sy[idx];
        }
    }

    s = mpi_allreduce(sum_local, MPI_SUM);

    return s * grid.dx * grid.dy;
}

template< typename T >
T average_xy(const T* u, const uniGrid2d< T >& grid)
{
    return integrate_xy(u, grid) / (grid.mpi_length * grid.mpi_width);
}

template< typename T >
T average_xy(const T* lx, const T* ly, const uniGrid2d< T >& grid)
{
    return integrate_xy(lx, ly, grid) / (grid.mpi_length * grid.mpi_width);
}

template< typename T >
T max_xy(const T* u, const uniGrid2d< T >& grid)
{
    T max_local = -std::numeric_limits<T>::max();

    int i, j, idx;

    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
                max_local = max(max_local, u[idx]);
        }
    }

    return mpi_allreduce(max_local, MPI_MAX);
}

template< typename T >
T min_xy(const T* u, const uniGrid2d< T >& grid)
{
    T min_local = std::numeric_limits<T>::max();

    int i, j, idx;

    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
                min_local = min(min_local, u[idx]);
        }
    }

    return mpi_allreduce(min_local, MPI_MIN);
}

template < typename T >
void dyn2_Cr_Cs_MSE(T &Cs2, T &Cr, T* taux, T* tauy, T* tx, T* ty, T* bx, T* by, const T max_nu, const uniGrid2d< T >&grid) {
    T detA;
    T tt, bt, tb, bb;
    T taut, taub;
    T Cs4;
    T epsilon = std::numeric_limits<T>::min();

    tt = integrate_xy(tx, ty, tx, ty, grid);
    bt = integrate_xy(bx, by, tx, ty, grid);
    tb = integrate_xy(tx, ty, bx, by, grid);
    bb = integrate_xy(bx, by, bx, by, grid);
    taut = integrate_xy(taux, tauy, tx, ty, grid);
    taub = integrate_xy(taux, tauy, bx, by, grid);

    // Equal to var(t)*var(b)*(1-corr(t,b)) >=0
    // Consequnetly, we can add epsilon to avoid division by zero
    // I think there is absolutely no way that b and t are perfectly correlated
    detA = tt * bb - bt * tb + epsilon;

    Cs2 = min(max((+bb*taut - bt*taub) / detA, (T)0.0), max_nu);
    Cr  = min(max((-tb*taut + tt*taub) / detA, (T)0.0), (T)30.0);
}

template < typename T >
void dyn2_Cr_Cs_Morinishi(T &Cs2, T &Cr, T* taux, T* tauy, T* tx, T* ty, T* bx, T* by, const T max_nu, const uniGrid2d< T >&grid) {
    T detA;
    T tt, bt, tb, bb;
    T taut, taub;
    T Cs4;
    T epsilon = std::numeric_limits<T>::min();

    tt = integrate_xy(tx, ty, tx, ty, grid);
    tb = integrate_xy(tx, ty, bx, by, grid);
    bb = integrate_xy(bx, by, bx, by, grid);
    taut = integrate_xy(taux, tauy, tx, ty, grid);
    taub = integrate_xy(taux, tauy, bx, by, grid);

    // In Morinishi-Vasilyev method we first solve for eddy viscosity
    // It is simply equivalent to DMM model
    Cs2 = taut / (tt + epsilon);
    Cs2 = min(max(Cs2, (T)0.0), max_nu);

    // simple MSE method for remaining coefficient
    Cr = (taub - Cs2 * tb) / (bb + epsilon);
    Cr = min(max(Cr, (T)0.0), (T)30.0);
}

template < typename T >
void dyn2_Cs_Cr_ZE(T &Cs2, T &Cr, T* taux, T* tauy, T* tx, T* ty, T* bx, T* by, T* sx, T*sy, T* px, T* py, const uniGrid2d< T >&grid) {
    T detA;
    T ts, bs, tp, bp;
    T taus, taup;

    ts = integrate_xy(tx, ty, sx, sy, grid);
    bs = integrate_xy(bx, by, sx, sy, grid);
    tp = integrate_xy(tx, ty, px, py, grid);
    bp = integrate_xy(bx, by, px, py, grid);
    taus = integrate_xy(taux, tauy, sx, sy, grid);
    taup = integrate_xy(taux, tauy, px, py, grid);

    detA = ts * bp - bs * tp;

    Cs2 = (+bp*taus - bs*taup) / detA;
    Cr  = min(max((-tp*taus + ts*taup) / detA, (T)0.0), (T)30.0);
}

template < typename T >
void top_hat(T* wc, T* w, const T tf_width, const uniGrid2d< T >& grid)
{
    int i, j, idx;
    T c0, c1;
    T w1[grid.size];

    // coeffs of 1d filtration, 2d filtration is product of templates
    c1 = tf_width * tf_width / (T)24.0;
    c0 = (T)1.0 - (T)2.0 * c1;

    grid.mpi_com.exchange_halo(w,
    grid.nx, grid.ny, grid.gcx, grid.gcy,
    1, 1, 1, 1);

    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
            
            w1[idx] = c1 * (c1 * w[idx - grid.ny - 1] + c0 * w[idx - grid.ny] + c1 * w[idx - grid.ny + 1])
                    + c0 * (c1 * w[idx           - 1] + c0 * w[idx          ] + c1 * w[idx           + 1])  
                    + c1 * (c1 * w[idx + grid.ny - 1] + c0 * w[idx + grid.ny] + c1 * w[idx + grid.ny + 1]);
        }
    }

    memcpy(wc, w1, grid.size * sizeof(T));
}

// -- filters -- //
// filter width here relative to mesh step
template < typename T >
void apply_filter(T* wc, T* w, const T tf_width, const uniGrid2d< T >& grid)
{
    #ifdef MESH_HAT_FILTER
    top_hat(wc, w, tf_width, grid);
    #endif

    #ifdef SPECTRAL_GAUSS_FILTER
    gauss_filter(wc, w, tf_width * grid.dx, grid);
    #endif
}

// product of two simple filters
template < typename T >
void apply_filter(T* wc, T* w, const T test_width, const T base_width, const uniGrid2d< T >& grid)
{
    T w1[grid.size];
    if (base_width < (T)1.0) {
        apply_filter(w1, w, test_width, grid);
        memcpy(wc, w1, grid.size * sizeof(T));
    }
    else {
        apply_filter(w1, w, base_width, grid);
        apply_filter(wc, w1, test_width, grid);
    }
}

template < typename T >
void apply_filter_iter(T* wc, T* w, const int n, const uniGrid2d< T >& grid)
{
    int i;
    T *w1, *w2;
    allocate(&w1, &w2, grid.size);

    memcpy(w1, w, grid.size * sizeof(T));
    for (i = 0; i < n; i++) {
        apply_filter(w2, w1, (T)sqrt(6.0), grid);
        std::swap(w1, w2);
    }
    memcpy(wc, w1, grid.size * sizeof(T));
    deallocate(w1, w2);
}

// -- interpolations -- //
template < typename T >
void u_to_v(T* v, T* u, const uniGrid2d< T >& grid)
{
    int i, j, idx;

    grid.mpi_com.exchange_halo(u,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);

    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
            
            v[idx] = (u[idx] + u[idx + grid.ny] + u[idx + grid.ny - 1] + u[idx - 1]) * (T)0.25;
        }
    }
}

template < typename T >
void v_to_u(T* u, T* v, const uniGrid2d< T >& grid)
{
    int i, j, idx;

    grid.mpi_com.exchange_halo(v,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);

    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
            
            u[idx] = (v[idx] + v[idx - grid.ny] + v[idx - grid.ny + 1] + v[idx + 1]) * (T)0.25;
        }
    }
}

template < typename T >
void w_to_v(T* v, T* w, const uniGrid2d< T >& grid)
{
    int i, j, idx;

    grid.mpi_com.exchange_halo(w,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);

    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
            
            v[idx] = (w[idx] + w[idx + grid.ny]) * (T)0.5;
        }
    }
}

template < typename T >
void w_to_u(T* u, T* w, const uniGrid2d< T >& grid)
{
    int i, j, idx;

    grid.mpi_com.exchange_halo(w,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);

    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
            
            u[idx] = (w[idx] + w[idx + 1]) * (T)0.5;
        }
    }
}

template < typename T >
void w_to_p(T* p, T* w, const uniGrid2d< T >& grid)
{
    int i, j, idx;

    grid.mpi_com.exchange_halo(w,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);

    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
            p[idx] = (w[idx] + w[idx + 1] + w[idx + grid.ny] + w[idx + grid.ny + 1]) * (T)0.25;
        }
    }
}

template < typename T >
void u_to_w(T* w, T* u, const uniGrid2d< T >& grid)
{
    int i, j, idx;

    grid.mpi_com.exchange_halo(u,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);

    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
            
            w[idx] = (u[idx] + u[idx - 1]) * (T)0.5;
        }
    }
}

template < typename T >
void v_to_w(T* w, T* v, const uniGrid2d< T >& grid)
{
    int i, j, idx;

    grid.mpi_com.exchange_halo(v,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);

    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
            
            w[idx] = (v[idx] + v[idx - grid.ny]) * (T)0.5;
        }
    }
}

template < typename T >
void u_to_p(T* p, T* u, const uniGrid2d< T >& grid)
{
    int i, j, idx;

    grid.mpi_com.exchange_halo(u,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);

    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
            
            p[idx] = (u[idx] + u[idx + grid.ny]) * (T)0.5;
        }
    }
}

template < typename T >
void v_to_p(T* p, T* v, const uniGrid2d< T >& grid)
{
    int i, j, idx;

    grid.mpi_com.exchange_halo(v,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);

    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
            
            p[idx] = (v[idx] + v[idx + 1]) * (T)0.5;
        }
    }
}

template < typename T >
void p_to_u(T* u, T* p, const uniGrid2d< T >& grid)
{
    int i, j, idx;

    grid.mpi_com.exchange_halo(p,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);

    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
            
            u[idx] = (p[idx] + p[idx - grid.ny]) * (T)0.5;
        }
    }
}

template < typename T >
void p_to_v(T* v, T* p, const uniGrid2d< T >& grid)
{
    int i, j, idx;

    grid.mpi_com.exchange_halo(p,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);

    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
            
            v[idx] = (p[idx] + p[idx - 1]) * (T)0.5;
        }
    }
}

template < typename T >
void p_to_w(T* w, T* p, const uniGrid2d< T >& grid)
{
    int i, j, idx;

    grid.mpi_com.exchange_halo(p,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);

    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
            w[idx] = (p[idx] + p[idx - 1] + p[idx - grid.ny] + p[idx - grid.ny - 1]) * (T)0.25;
        }
    }
}

// -- differential operators -- //
// sx, sy -- in v and u points
template < typename T >
void nabla(T* sx, T* sy, T* w, const uniGrid2d< T >& grid)
{
    int i, j, idx;

    grid.mpi_com.exchange_halo(w,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);

    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
            
            sx[idx] = (w[idx+grid.ny] - w[idx]) * grid.dxi;
            sy[idx] = (w[idx+1      ] - w[idx]) * grid.dyi;
        }
    }
}

template < typename T >
void laplacian(T* lapw, T* w, const uniGrid2d< T >& grid)
{
    int i, j, idx;

    grid.mpi_com.exchange_halo(w,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);

    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
            
            lapw[idx] = (w[idx+grid.ny] - (T)2.0 * w[idx] + w[idx-grid.ny]) * grid.dx2i + 
                        (w[idx+1      ] - (T)2.0 * w[idx] + w[idx-1      ]) * grid.dy2i;
        }
    }
}

// lxx, lyy - in p points
// lxy - in w points
// lx, ly -- in v and u points correspondingly
template < typename T >
void tensor_to_vector(T* lx, T* ly, T* lxx, T* lxy, T* lyy, const uniGrid2d< T >& grid)
{
    int i, j, idx;

    grid.mpi_com.exchange_halo(lxx, lyy,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);
    grid.mpi_com.exchange_halo(lxy,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);

    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
            
            lx[idx] = (lxy[idx + grid.ny] - lxy[idx]) * grid.dxi - (lxx[idx] - lxx[idx - 1]) * grid.dyi;
            ly[idx] = (lyy[idx] - lyy[idx - grid.ny]) * grid.dxi - (lxy[idx + 1] - lxy[idx]) * grid.dyi;
        }
    }
}

// dw/dt = ... - div(tx,ty)
template < typename T >
void divergence_vector(T* wim, T* tx, T* ty, const uniGrid2d< T >& grid)
{
    int i, j, idx;

    grid.mpi_com.exchange_halo(tx, ty,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);

    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

            wim[idx] -=   (tx[idx] - tx[idx - grid.ny]) * grid.dxi
                        + (ty[idx] - ty[idx - 1      ]) * grid.dyi;
        }
    }
}

template < typename T >
void divergence_tensor(T* fx, T* fy, T* txx, T* txy, T* tyy, const uniGrid2d< T >& grid)
{
    int i, j, idx;

    grid.mpi_com.exchange_halo(txx, tyy,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);

    grid.mpi_com.exchange_halo(txy,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);

    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

            fx[idx] = - (txx[idx] - txx[idx - grid.ny]) * grid.dxi
                      - (txy[idx + 1] - txy[idx]) * grid.dyi;
            
            fy[idx] = - (tyy[idx] - tyy[idx - 1]) * grid.dyi
                      - (txy[idx + grid.ny] - txy[idx]) * grid.dxi;
        }
    }
}

// upwind tendency in p points
template < typename T >
void upwind_advection_p(T* wim, T* u, T* v, T* X, const uniGrid2d< T >& grid)
{
    int i, j, idx;
    T flux_u[grid.size], flux_v[grid.size]; // u and v fluxes in u and v points

    grid.mpi_com.exchange_halo(X,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);

    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

            flux_u[idx] = max(u[idx], (T)0.0) * X[idx - grid.ny]
                        + min(u[idx], (T)0.0) * X[idx];
            flux_v[idx] = max(v[idx], (T)0.0) * X[idx - 1]
                        + min(v[idx], (T)0.0) * X[idx];
        }
    }

    grid.mpi_com.exchange_halo(flux_u, flux_v,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);

    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

            wim[idx] -= (flux_u[idx + grid.ny] - flux_u[idx]) * grid.dxi
                      + (flux_v[idx +       1] - flux_v[idx]) * grid.dyi;
        }
    }
}

// abs value in p points
template < typename T >
void abs_vector(T* abs_p, T* sx, T* sy, const uniGrid2d< T >& grid)
{
    int i, j, idx;
    T sx2[grid.size], sy2[grid.size]; 

    grid.mpi_com.exchange_halo(sx, sy,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);
    
    mul(sx2, sx, sx, grid.size);
    mul(sy2, sy, sy, grid.size);

    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
            
            abs_p[idx] = sqrt((sx2[idx] + sx2[idx+1      ]) * (T)0.5 + 
                              (sy2[idx] + sy2[idx+grid.ny]) * (T)0.5);
        }
    }
}

// sxx, syy in p points, sxy in w points
template < typename T >
void strain_tensor(T* sxx, T* sxy, T* syy, T* u, T* v, const uniGrid2d< T >& grid)
{
    int i, j, idx;

    grid.mpi_com.exchange_halo(u, v,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);
        
    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
            
            sxx[idx] = (u[idx + grid.ny] - u[idx]) * grid.dxi;
            syy[idx] = (v[idx + 1      ] - v[idx]) * grid.dyi;
            sxy[idx] = (u[idx] - u[idx - 1      ]) * grid.dyih
                     + (v[idx] - v[idx - grid.ny]) * grid.dxih;
        }
    }
}

// S in p points
template < typename T >
void compute_S(T* S, T* sxx, T* sxy, T* syy, const uniGrid2d< T >& grid)
{
    int i, j, idx;
    
    T s2w[grid.size];
    
    grid.mpi_com.exchange_halo(sxy,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);
    
    mul(s2w, sxy, sxy, grid.size);
    
    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
            
            S[idx] = sqrt((T)2.0 * (sxx[idx] * sxx[idx] + syy[idx] * syy[idx] + (T)0.5 * (s2w[idx] + s2w[idx + grid.ny] + s2w[idx + grid.ny + 1] + s2w[idx + 1])));
        }
    }
}

// compute S in p poitns given velocity
template < typename T >
void compute_S_uv(T* S, T* u, T* v, const uniGrid2d< T >& grid)
{
    T sxx[grid.size], sxy[grid.size], syy[grid.size];

    strain_tensor(sxx, sxy, syy, u, v, grid);
    compute_S(S, sxx, sxy, syy, grid);
}

// in p points
// filter width in absolute units
template < typename T >
void estimate_sgs_ke(T* ke_estimate, T* w, T* u, T* v, T filter_width, const uniGrid2d< T >& grid)
{
    T sxx[grid.size], sxy[grid.size], syy[grid.size];
    T S[grid.size], w2[grid.size], w2_p[grid.size];

    strain_tensor(sxx, sxy, syy, u, v, grid);
    compute_S(S, sxx, sxy, syy, grid);

    mul(w2, w, w, grid.size);
    w_to_p(w2_p, w2, grid);

    T C0 = sqr(filter_width) / (T)48.0;
    for (int idx = 0; idx < grid.size; idx++) {
        ke_estimate[idx] = C0 * (sqr(S[idx]) + w2_p[idx]);
    }
}

// L = filter(uw) - filter(u)filter(w)
// lx in v point, ly in u point
// if base_width > 1.0, then two filters applied, otherwise only test filter
template < typename T >
void compute_leonard_vector(T* lx, T* ly, T* w, T* u, T* v, const T test_width, const T base_width, const T Csim, const uniGrid2d< T >& grid)
{
    T u_v[grid.size], v_u[grid.size], w_u[grid.size], w_v[grid.size], uw[grid.size], vw[grid.size];
    T wc[grid.size], uc[grid.size], vc[grid.size];
    
    // ----- first part of Leonard vector ---- //
    
    // u,v in lx, ly points
    u_to_v(u_v, u, grid);
    v_to_u(v_u, v, grid);

    // w in lx, ly points
    w_to_u(w_u, w, grid);
    w_to_v(w_v, w, grid);
    
    mul(uw, u_v, w_v, grid.size);
    mul(vw, v_u, w_u, grid.size);
    
    apply_filter(lx, uw, test_width, base_width, grid);
    apply_filter(ly, vw, test_width, base_width, grid);
    
    // ----- second part of Leonard vector ---- //
    
    apply_filter(wc, w, test_width, base_width, grid);
    apply_filter(uc, u, test_width, base_width, grid);
    apply_filter(vc, v, test_width, base_width, grid);
    
    u_to_v(u_v, uc, grid);
    v_to_u(v_u, vc, grid);

    w_to_u(w_u, wc, grid);
    w_to_v(w_v, wc, grid);
    
    mul(uw, u_v, w_v, grid.size);
    mul(vw, v_u, w_u, grid.size);
    
    // ---------- full Leonard vector -------- //
    
    update(lx, -(T)1.0, uw, grid.size);
    update(ly, -(T)1.0, vw, grid.size);

    mul(lx, Csim, grid.size);
    mul(ly, Csim, grid.size);
}

template < typename T >
void backscatter_ngm(T* lx, T* ly, T* w, T* u, T* v, const T test_width, const T base_width, const uniGrid2d< T >& grid)
{
    T uc[grid.size], vc[grid.size], wc[grid.size];
    T uf[grid.size], vf[grid.size], wf[grid.size];
    T sxf[grid.size], syf[grid.size];

    apply_filter(wc, w, test_width, base_width, grid);
    apply_filter(uc, u, test_width, base_width, grid);
    apply_filter(vc, v, test_width, base_width, grid);

    assign(uf, (T)1.0, u, -(T)1.0, uc, grid.size);
    assign(vf, (T)1.0, v, -(T)1.0, vc, grid.size);
    assign(wf, (T)1.0, w, -(T)1.0, wc, grid.size);

    nabla(sxf, syf, wf, grid);

    T filter_width = sqrt(sqr(test_width) + sqr(base_width)) * grid.dx;   
    compute_ngm_vector(lx, ly, uf, vf, sxf, syf, filter_width, grid);
    apply_filter(lx, lx, test_width, base_width, grid);
    apply_filter(ly, ly, test_width, base_width, grid);
}

template < typename T >
void compute_ngm_vector(T* lx, T* ly, T* u, T* v, T* sx, T* sy, T filter_width, const uniGrid2d< T >& grid)
{
    T sxx[grid.size], sxy[grid.size], syy[grid.size];
    T sxx_v[grid.size], sxy_u[grid.size], syy_u[grid.size], sxy_v[grid.size];
    T sx_u[grid.size], sy_v[grid.size];

    strain_tensor(sxx, sxy, syy, u, v, grid);
    p_to_v(sxx_v, sxx, grid);
    w_to_u(sxy_u, sxy, grid);
    p_to_u(syy_u, syy, grid);
    w_to_v(sxy_v, sxy, grid);

    v_to_u(sx_u, sx, grid);
    u_to_v(sy_v, sy, grid);

    T C0 = sqr(filter_width) / (T)12.0;

    for (int i = 0; i < grid.size; i++)
    {
        lx[i] = C0 * (sxx_v[i] * sx  [i] + sxy_v[i] * sy_v[i]);
        ly[i] = C0 * (sxy_u[i] * sx_u[i] + syy_u[i] * sy  [i]);
    }
}

template < typename T >
void compute_ngm_vector_conservative(T* lx, T* ly, T* w, T* u, T* v, const T test_width, const T base_width, const uniGrid2d< T >& grid)
{
    T sxx[grid.size], sxy[grid.size], syy[grid.size];
    T lxx[grid.size], lxy[grid.size], lyy[grid.size];
    T sxx_w[grid.size], sxyw[grid.size];
    T wc[grid.size];
 
    T filter_ratio = sqrt((T)2.0 / exp(1)); // regularization filter width ratio

    strain_tensor(sxx, sxy, syy, u, v, grid);

    //assign(wc, (T)1.0, w, grid.size);
    //apply_filter(wc, w, test_width, base_width, grid);
    //apply_filter(wc, w, test_width, (T)0.0, grid); //work better, but similarity is worse
    //apply_filter(wc, w, test_width * filter_ratio, base_width * filter_ratio, grid); // optimal filter width
    assign(wc, (T)1.0, w, grid.size);

    mul(sxyw, wc, sxy, grid.size);
    w_to_p(lyy, sxyw, grid);
    assign(lxx, -(T)1.0, lyy, grid.size);

    p_to_w(sxx_w, sxx, grid);
    mul(lxy, sxx_w, wc, grid.size);
    
    tensor_to_vector(lx, ly, lxx, lxy, lyy, grid);

    T filter_width = sqrt(sqr(test_width) + sqr(base_width));   
    T C0 = sqr(filter_width * grid.dx) / (T)12.0;

    mul(lx, C0, grid.size);
    mul(ly, C0, grid.size);
}

template < typename T >
void regularize_backscatter(T* damping_coef, T* lx, T* ly, T* w, T* u, T* v, const uniGrid2d< T >& grid)
{
    T sxx[grid.size], sxy[grid.size], syy[grid.size];
    T sxx_v[grid.size], sxy_u[grid.size], syy_u[grid.size], sxy_v[grid.size];
    T sx[grid.size], sy[grid.size];
    T sx_u[grid.size], sy_v[grid.size];
    T lap_lx[grid.size], lap_ly[grid.size]; // laplace(lx,ly)
    T Sw_x[grid.size], Sw_y[grid.size]; // S * nabla omega
    T ls[grid.size]; // nabla omega * l
    T wSw[grid.size]; // nabla omega * S nabla omega
    T w_lapl[grid.size]; //nabla omega * laplace(lx,ly)

    strain_tensor(sxx, sxy, syy, u, v, grid);
    p_to_v(sxx_v, sxx, grid);
    w_to_u(sxy_u, sxy, grid);
    p_to_u(syy_u, syy, grid);
    w_to_v(sxy_v, sxy, grid);

    nabla(sx, sy, w, grid);
    v_to_u(sx_u, sx, grid);
    u_to_v(sy_v, sy, grid);

    for (int i = 0; i < grid.size; i++)
    {
        Sw_x[i] = (sxx_v[i] * sx  [i] + sxy_v[i] * sy_v[i]);
        Sw_y[i] = (sxy_u[i] * sx_u[i] + syy_u[i] * sy  [i]);
    }

    scal_prod(ls, sx, sy, lx, ly, grid);
    scal_prod(wSw, sx, sy, Sw_x, Sw_y, grid);
    laplacian(lap_lx, lx, grid);
    laplacian(lap_ly, ly, grid);
    scal_prod(w_lapl, sx, sy, lap_lx, lap_ly, grid);

    assign(damping_coef, (T)1.0, grid.size);
    bool enstrophy_backscatter;
    bool gradient_fall_resolved;
    bool gradient_rise_subgrid;
    bool gradient_rise_total;
    for (int i = 0; i < grid.size; i++)
    {
        enstrophy_backscatter = ls[i] > (T)0.0;
        gradient_fall_resolved = wSw[i] > (T)0.0;
        gradient_rise_subgrid = w_lapl[i] < (T)0.0;
        gradient_rise_total = (- wSw[i] - w_lapl[i]) > (T)0.0;

        if (enstrophy_backscatter && gradient_fall_resolved && gradient_rise_subgrid && gradient_rise_total)
        {
            damping_coef[i] = - wSw[i] / w_lapl[i];
            if (damping_coef[i] > (T)1.0) {
                printf("Warning! backscatter regularization is incorrect\n");
            }
        }
    }

    // in diagnostic regime
    //T damp_x[grid.size], damp_y[grid.size];
    //p_to_v(damp_x, damping_coef, grid);
    //p_to_u(damp_y, damping_coef, grid);
    
    //mul(lx, lx, damp_x, grid.size);
    //mul(ly, ly, damp_y, grid.size);
}

template < typename T >
void backscatter_ssm(T* lx, T* ly, T* w, T* u, T* v, T* psi, const T test_width, const T base_width, const T Eback, T& Csim, const uniGrid2d< T >& grid)
{
    T epsilon = std::numeric_limits<T>::min();

    T uc[grid.size], vc[grid.size], wc[grid.size];
    T uf[grid.size], vf[grid.size], wf[grid.size];

    apply_filter(wc, w, test_width, base_width, grid);
    apply_filter(uc, u, test_width, base_width, grid);
    apply_filter(vc, v, test_width, base_width, grid);

    assign(uf, (T)1.0, u, -(T)1.0, uc, grid.size);
    assign(vf, (T)1.0, v, -(T)1.0, vc, grid.size);
    assign(wf, (T)1.0, w, -(T)1.0, wc, grid.size);

    compute_leonard_vector(lx, ly, wf, uf, vf, test_width, base_width, (T)1.0, grid);
    //apply_filter(lx, lx, test_width, base_width, grid);
    //apply_filter(ly, ly, test_width, base_width, grid);
    if (Csim > -epsilon) {
        mul(lx, Csim, grid.size);
        mul(ly, Csim, grid.size);
    }
    else {
        T E_back_ssm;
        T b_E[grid.size];
        T px[grid.size], py[grid.size];
        nabla(px, py, psi, grid);
        scal_prod(b_E, lx, ly, px, py, grid);
        E_back_ssm = - average_xy(b_E, grid); // > 0 means energy generation
        if (E_back_ssm > epsilon && Eback > epsilon)
            Csim = min(Eback / E_back_ssm, (T)30.0);
        else
            Csim = (T)0.0;
        mul(lx, Csim, grid.size);
        mul(ly, Csim, grid.size);
    }
}

template < typename T >
T compute_Cback(T* lx, T* ly, T* tx, T* ty, T* bx, T* by, T* w, T* psi, const T base_width, const uniGrid2d< T >& grid)
{
    T epsilon = std::numeric_limits<T>::min();
    T answer;

    T px[grid.size], py[grid.size];
    T sx[grid.size], sy[grid.size];
    T betax[grid.size], betay[grid.size];

    nabla(px, py, psi, grid);
    nabla(sx, sy, w, grid);

    T en_ens_C = sqr(grid.dx * base_width) / (T)12.0;
    assign(betax, (T)1.0, px, -en_ens_C, sx, grid.size);
    assign(betay, (T)1.0, py, -en_ens_C, sy, grid.size);

    T Cssm = integrate_xy(lx, ly, betax, betay, grid);
    T Ct   = integrate_xy(tx, ty, betax, betay, grid);
    T Cb   = integrate_xy(bx, by, betax, betay, grid);

    if (fabs(Cb) > epsilon)
        answer = min(max(-(Cssm+Ct)/Cb, (T)0.0), (T)30.0);
    else 
        answer = (T)0.0;
    return answer;  
}

template < typename T >
void AD_iteration(T* u, T* uc, int Niter, const T test_width, const T base_width, const uniGrid2d< T >& grid)
{
    memcpy(u, uc, grid.size * sizeof(T));
    T uf[grid.size];

    for (int i = 0; i < Niter; i++)
    {
        apply_filter(uf, u, test_width, base_width, grid);
        update(u, (T)1.0, uc, -(T)1.0, uf, grid.size);
    }
}

// ADM minus SSM, i.e. only ADM approximation to cross and reynolds terms
template < typename T >
void compute_adm_vector(T* lx, T* ly, T* w, T* u, T* v, int Niter, const T test_width, const T base_width, const uniGrid2d< T >& grid)
{
    T udf[grid.size], vdf[grid.size], wdf[grid.size]; // defiltered variables
    T u_v[grid.size], v_u[grid.size], w_u[grid.size], w_v[grid.size], uw[grid.size], vw[grid.size];
    T wc[grid.size], uc[grid.size], vc[grid.size];
    T ufdf[grid.size], vfdf[grid.size], wfdf[grid.size];
    
    // ----- first part of ADM vector ---- //
    AD_iteration(udf, u, Niter, test_width, base_width, grid);
    AD_iteration(vdf, v, Niter, test_width, base_width, grid);
    AD_iteration(wdf, w, Niter, test_width, base_width, grid);
    
    // u,v in lx, ly points
    u_to_v(u_v, udf, grid);
    v_to_u(v_u, vdf, grid);

    // w in lx, ly points
    w_to_u(w_u, wdf, grid);
    w_to_v(w_v, wdf, grid);
    
    mul(uw, u_v, w_v, grid.size);
    mul(vw, v_u, w_u, grid.size);
    
    apply_filter(lx, uw, test_width, base_width, grid);
    apply_filter(ly, vw, test_width, base_width, grid);
    
    // ----- second part of ADM vector ---- //
    #ifdef ADM_LAYTON

    u_to_v(u_v, u, grid);
    v_to_u(v_u, v, grid);

    w_to_u(w_u, w, grid);
    w_to_v(w_v, w, grid);
    
    mul(uw, u_v, w_v, grid.size);
    mul(vw, v_u, w_u, grid.size);

    #endif

    #ifdef ADM_CHOW

    apply_filter(ufdf, udf, test_width, base_width, grid);
    apply_filter(vfdf, vdf, test_width, base_width, grid);
    apply_filter(wfdf, wdf, test_width, base_width, grid);

    u_to_v(u_v, ufdf, grid);
    v_to_u(v_u, vfdf, grid);

    w_to_u(w_u, wfdf, grid);
    w_to_v(w_v, wfdf, grid);
    
    mul(uw, u_v, w_v, grid.size);
    mul(vw, v_u, w_u, grid.size);

    #endif
    
    
    // ---------- full ADM vector -------- //
    
    update(lx, -(T)1.0, uw, grid.size);
    update(ly, -(T)1.0, vw, grid.size);

    // ---------- subtract SSM --------- //
    T ssmx[grid.size], ssmy[grid.size];
    compute_leonard_vector(ssmx, ssmy, w, u, v, test_width, base_width, (T)1.0, grid);
    update(lx, -(T)1.0, ssmx, grid.size);
    update(ly, -(T)1.0, ssmy, grid.size);
}

// ---------------- models UV --------------------- //

// nu_p in p points
// sxx, syy in p points, sxy in w points
template < typename T >
void lap_UV_model(T* fx, T* fy, T* u, T* v, T* nu_p, const uniGrid2d< T >& grid)
{   
    T sxx[grid.size], sxy[grid.size], syy[grid.size];
    T txx[grid.size], txy[grid.size], tyy[grid.size];
    T nu_w[grid.size];

    p_to_w(nu_w, nu_p, grid);
    strain_tensor(sxx, sxy, syy, u, v, grid);

    for (int idx = 0; idx < grid.size; idx++) {
        txx[idx] = - (T)2.0 * nu_p[idx] * sxx[idx];
        txy[idx] = - (T)2.0 * nu_w[idx] * sxy[idx];
        tyy[idx] = - (T)2.0 * nu_p[idx] * syy[idx];
    }

    // divergence with minus, i.e. velocity tendency
    divergence_tensor(fx, fy, txx, txy, tyy, grid);
}

template< typename T >
void velocity_to_vorticity(
	T* w, T* u, T* v,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;

    grid.mpi_com.exchange_halo(u, v,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);

	for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {

		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			w[idx] =
				(v[idx] - v[idx - grid.ny]) * grid.dxi
				- (u[idx] - u[idx-1]) * grid.dyi;
		}
	}
}

// ------------------- models --------------------- //
// alphax, alphay -- model, dw/dt = - d_j(nu*alpha_j)
// sx, sy = nabla(w)
//--------------------------------------------------//
template < typename T >
void lap_model(T* alphax, T* alphay, T* sx, T* sy, const T mix_length, const nse::uniGrid2d< T >& grid)
{   
    assign(alphax, -sqr(mix_length), sx, grid.size);
    assign(alphay, -sqr(mix_length), sy, grid.size);
}

template < typename T >
void lap_nu_model(T* alphax, T* alphay, T* w, T* nub, const T mix_length, const nse::uniGrid2d< T >& grid)
{   
    int i, j, idx;
    T mix_length2 = sqr(mix_length);
    T w_nub[grid.size];

    mul(w_nub, w, nub, grid.size);

    grid.mpi_com.exchange_halo(w_nub,
        grid.nx, grid.ny, grid.gcx, grid.gcy,
        1, 1, 1, 1);

    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
            alphax[idx] = - mix_length2 * (w_nub[idx+grid.ny] - w_nub[idx]) * grid.dxi;
            alphay[idx] = - mix_length2 * (w_nub[idx+1      ] - w_nub[idx]) * grid.dyi;
        }
    }
}

template < typename T >
void bilap_model(T* alphax, T* alphay, T* sx, T* sy, const T mix_length, const nse::uniGrid2d< T >& grid)
{    
    T lap_sx[grid.size], lap_sy[grid.size];
    
    laplacian(lap_sx, sx, grid);
    laplacian(lap_sy, sy, grid);
    
    assign(alphax, sqr(sqr(mix_length)), lap_sx, grid.size);
    assign(alphay, sqr(sqr(mix_length)), lap_sy, grid.size);
}

template < typename T >
void lap_leith_model(T* alphax, T* alphay, T* sx, T* sy, const T mix_length, const uniGrid2d< T >& grid)
{   
    T abs_x[grid.size], abs_y[grid.size], abs_p[grid.size];
    
    abs_vector(abs_p, sx, sy, grid);
    p_to_v(abs_x, abs_p, grid);
    p_to_u(abs_y, abs_p, grid);
    
    mul(alphax, sx, abs_x, grid.size);
    mul(alphay, sy, abs_y, grid.size);
    
    assign(alphax, - sqr(mix_length) * mix_length, alphax, grid.size);
    assign(alphay, - sqr(mix_length) * mix_length, alphay, grid.size);
}

template < typename T >
void lap_smagorinsky_model(T* alphax, T* alphay, T* sx, T* sy, T* u, T* v, const T mix_length, const uniGrid2d< T >& grid)
{   
    T sxx[grid.size], sxy[grid.size], syy[grid.size], S_p[grid.size], S_x[grid.size], S_y[grid.size];
    
    strain_tensor(sxx, sxy, syy, u, v, grid);
    
    compute_S(S_p, sxx, sxy, syy, grid);
    p_to_v(S_x, S_p, grid);
    p_to_u(S_y, S_p, grid);
    
    mul(alphax, sx, S_x, grid.size);
    mul(alphay, sy, S_y, grid.size);
    
    assign(alphax, - sqr(mix_length), alphax, grid.size);
    assign(alphay, - sqr(mix_length), alphay, grid.size);
}

template < typename T >
void lap_w_smagorinsky_model(T* alphax, T* alphay, T* sx, T* sy, T* u, T* v, T* w, const T mix_length, const uniGrid2d< T >& grid)
{   
    T sxx[grid.size], sxy[grid.size], syy[grid.size], S_p[grid.size], S_x[grid.size], S_y[grid.size];
    T abs_w[grid.size], abs_w_p[grid.size];
    
    strain_tensor(sxx, sxy, syy, u, v, grid);
    compute_S(S_p, sxx, sxy, syy, grid);

    for (int i = 0; i < grid.size; i++)
        abs_w[i] = fabs(w[i]);
    w_to_p(abs_w_p, abs_w, grid);
    for (int i = 0; i < grid.size; i++)
        S_p[i] = max(S_p[i] - abs_w_p[i], (T)0.0);

    p_to_v(S_x, S_p, grid);
    p_to_u(S_y, S_p, grid);
    
    mul(alphax, sx, S_x, grid.size);
    mul(alphay, sy, S_y, grid.size);
    
    assign(alphax, - sqr(mix_length), alphax, grid.size);
    assign(alphay, - sqr(mix_length), alphay, grid.size);
}

template < typename T >
void bilap_smagorinsky_model(T* alphax, T* alphay, T* sx, T* sy, T* u, T* v, const T mix_length, const uniGrid2d< T >& grid)
{   
    T sxx[grid.size], sxy[grid.size], syy[grid.size], S_p[grid.size], S_x[grid.size], S_y[grid.size];
    T lap_sx[grid.size], lap_sy[grid.size];

    strain_tensor(sxx, sxy, syy, u, v, grid);
    
    compute_S(S_p, sxx, sxy, syy, grid);
    p_to_v(S_x, S_p, grid);
    p_to_u(S_y, S_p, grid);
    
    laplacian(lap_sx, sx, grid);
    laplacian(lap_sy, sy, grid);

    mul(alphax, lap_sx, S_x, grid.size);
    mul(alphay, lap_sy, S_y, grid.size);
    
    assign(alphax, sqr(sqr(mix_length)), alphax, grid.size);
    assign(alphay, sqr(sqr(mix_length)), alphay, grid.size);
}

template < typename T >
void bilap_leith_model(T* alphax, T* alphay, T* sx, T* sy, const T mix_length, const uniGrid2d< T >& grid)
{   
    T abs_x[grid.size], abs_y[grid.size], abs_p[grid.size];
    T lap_sx[grid.size], lap_sy[grid.size];
    
    abs_vector(abs_p, sx, sy, grid);
    p_to_v(abs_x, abs_p, grid);
    p_to_u(abs_y, abs_p, grid);

    laplacian(lap_sx, sx, grid);
    laplacian(lap_sy, sy, grid);
    
    mul(alphax, lap_sx, abs_x, grid.size);
    mul(alphay, lap_sy, abs_y, grid.size);
    
    assign(alphax,  sqr(sqr(mix_length)) * mix_length, alphax, grid.size);
    assign(alphay,  sqr(sqr(mix_length)) * mix_length, alphay, grid.size);
}

template < typename T >
void bilap_w_smagorinsky_model(T* alphax, T* alphay, T* sx, T* sy, T* u, T* v, T* w, const T mix_length, const uniGrid2d< T >& grid)
{   
    T sxx[grid.size], sxy[grid.size], syy[grid.size], S_p[grid.size], S_x[grid.size], S_y[grid.size];
    T lap_sx[grid.size], lap_sy[grid.size];
    T abs_w[grid.size], abs_w_p[grid.size];

    strain_tensor(sxx, sxy, syy, u, v, grid);
    compute_S(S_p, sxx, sxy, syy, grid);

    for (int i = 0; i < grid.size; i++)
        abs_w[i] = fabs(w[i]);
    w_to_p(abs_w_p, abs_w, grid);
    for (int i = 0; i < grid.size; i++)
        S_p[i] = max(S_p[i] - abs_w_p[i], (T)0.0);

    p_to_v(S_x, S_p, grid);
    p_to_u(S_y, S_p, grid);
    
    laplacian(lap_sx, sx, grid);
    laplacian(lap_sy, sy, grid);

    mul(alphax, lap_sx, S_x, grid.size);
    mul(alphay, lap_sy, S_y, grid.size);
    
    assign(alphax, sqr(sqr(mix_length)), alphax, grid.size);
    assign(alphay, sqr(sqr(mix_length)), alphay, grid.size);
}

template < typename T >
void model_vector(T* alphax, T* alphay, T* sx, T* sy, T* u, T* v, T* w, const T mix_length, int viscosity_model, const uniGrid2d< T >& grid)
{
    switch (viscosity_model) {
        case lap:
            lap_model(alphax, alphay, sx, sy, mix_length, grid);
            break;
        case bilap:
            bilap_model(alphax, alphay, sx, sy, mix_length, grid);
            break;
        case lap_leith:
            lap_leith_model(alphax, alphay, sx, sy, mix_length, grid);
            break;
        case bilap_leith:
            bilap_leith_model(alphax, alphay, sx, sy, mix_length, grid);
            break;
        case lap_smag:
            lap_smagorinsky_model(alphax, alphay, sx, sy, u, v, mix_length, grid);
            break;
        case bilap_smag:
            bilap_smagorinsky_model(alphax, alphay, sx, sy, u, v, mix_length, grid);
            break;
        case lap_w_smag:
            lap_w_smagorinsky_model(alphax, alphay, sx, sy, u, v, w, mix_length, grid);
            break;
        case bilap_w_smag:
            bilap_w_smagorinsky_model(alphax, alphay, sx, sy, u, v, w, mix_length, grid);
            break;
        default:
            assert(1 == 2 && "viscosity model is wrong");
            break;
    }
}

template < typename T >
void Ediss_Eback_bilap_smag(T &Ediss_mean, T &Eback_mean, T* Ediss, T* Eback, T* w, T* u, T* v, const T* Cs2, const T* nub, const T mix_length, const uniGrid2d< T >& grid)
{
    int i, j, idx;
    T mix_length4 = sqr(sqr(mix_length));

    T sxx[grid.size], sxy[grid.size], syy[grid.size], S_p[grid.size], sx[grid.size], sy[grid.size], 
    s2[grid.size], w2_nu[grid.size];

    strain_tensor(sxx, sxy, syy, u, v, grid);
    compute_S(S_p, sxx, sxy, syy, grid);

    nabla(sx, sy, w, grid);
    scal_prod(s2, sx, sy, sx, sy, grid);

    for (i = 0; i < grid.size; i++)
        w2_nu[i] = sqr(w[i])*nub[i];

    w_to_p(Eback, w2_nu, grid);

    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
            Ediss[idx] = mix_length4 * S_p[idx] * Cs2[idx] * s2[idx]; //bilap smag
            //Ediss[idx] = (T)1.0 * mix_length4 * Cs2[idx] * s2[idx]; //bilap
        }
    }

    Ediss_mean = average_xy(Ediss, grid);
    Eback_mean = average_xy(Eback, grid);    
}

template< typename T >
void safe_division(T* nu, const T* numerator, const T* denominator, const T max_nu, const uniGrid2d< T >& grid)
{
    int i, j, idx;
    T epsilon = std::numeric_limits<T>::min();

    for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
    {
        idx = i * grid.ny + grid.gcy;
        for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
                nu[idx] = min(max(numerator[idx] / (denominator[idx] + epsilon), (T)0.0), max_nu);
        }
    }
}
