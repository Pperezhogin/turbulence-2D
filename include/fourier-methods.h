#pragma once

#include "unigrid2d.h"
#include "mtrand.h"
#include <limits>

#include <fftw3-mpi.h>

static fftw_plan b_plan_filter;
static fftw_plan f_plan_filter;
    
static double *wf;
static bool first_call = true;

namespace nse
{
	template <typename T>
	void power_exp_spectra(
        T* Psi, const T kp, const uniGrid2d< T >&grid);

	template <typename T>
	void gauss_filter(
        T* wc, T* w, const T filter_width, const uniGrid2d< T >&grid);

	template <typename T>
	void spectral_filter(
        T* wc, T* w, const int N_coarse, const uniGrid2d< T >&grid);

	template <typename T>
	void coarse_resolution(T* Xc, T* X, const int N_coarse, const uniGrid2d< T >& grid);
}

// mu -- expectation
// sigma -- std
template < typename T >
T gauss(const T mu, const T sigma)
{
	T epsilon = std::numeric_limits<T>::min();
	T two_pi = (T)2.0 * (T)M_PI;

	T u1, u2;
	do
	 {
	   u1 = rand() * (1.0 / RAND_MAX);
	   u2 = rand() * (1.0 / RAND_MAX);
	 }
	while ( u1 <= epsilon );

	return mu + sigma * (sqrt(-2.0 * log(u1)) * cos(two_pi * u2));
}

// uniform distribution on [a,b]
template < typename T >
T uniform_distribution(const T a, const T b)
{
	T r = rand() / (T)RAND_MAX;
	return a + r * (b-a);
}
