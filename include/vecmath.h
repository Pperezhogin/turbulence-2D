#pragma once

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <math.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <mpi.h>

#include "mpi-com.h"

#define _VEC_RESTRICT   __restrict

//#define _ALLOCATE_ALIGNED			// aligned allocation
#define _USE_ALIGNMENT		16		// alignment used for allocation (2^N value)

// *[vecmath.h]: DFT removed //

namespace nse
{
  	template< typename T > T sqr(const T a);

	template< typename T > T min(const T a, const T b);
	template< typename T > T max(const T a, const T b);

	template< typename T > T min(const T a, const T b, const T c);
	template< typename T > T max(const T a, const T b, const T c);

	template< typename T > T min(const T a, const T b, const T c, const T d);
	template< typename T > T max(const T a, const T b, const T c, const T d);

	template< typename T > T min(const T a, const T b, const T c, const T d, const T e);
	template< typename T > T max(const T a, const T b, const T c, const T d, const T e);

	template< typename T > T sign(const T x);
	template< typename T > T sign(const T x, const T eps);

	template< typename T > T minmod_limit(const T x, const T y);
	template< typename T > T superbee_limit(const T x, const T y);

	template< typename T > T delta(const T x, const T eps);

	template< typename T > T heavy_side(const T x);
	template< typename T > T heavy_side(const T x, const T eps);

	template< typename T > T linear_step(const T x, const T eps);

	template< typename T >
	void allocate(T** x, const int n);					// allocate memory(x) and null(x)
	template< typename T >
	void allocate(T** x, T** y, const int n);			// allocate memory(x,y) and null(x,y)
	template< typename T >
	void allocate(T** x, T** y, T** z, const int n);	// allocate memory(x,y,z) and null(x,y,z)

	template< typename T >
	void deallocate(T* x);								// deallocate memory(x)
	template< typename T >
	void deallocate(T* x, T* y);						// deallocate memory(x,y)
	template< typename T >
	void deallocate(T* x, T* y, T* z);					// deallocate memory(x,y,z)

	template< typename T >
	T min(const T* const x, const int n);

	template< typename T >
	T mpi_min(const T* const x, const int n);

	template< typename T >
	T max(const T* const x, const int n);

	template< typename T >
	T mpi_max(const T* const x, const int n);

	template< typename T >
	T sum(const T* const x, const int n);

	template< typename T >
	T mpi_sum(const T* const x, const int n);

	template< typename T >
	T dot_product(const T* const x, const T* const y, const int n);

	template< typename T >
	T mpi_dot_product(const T* const x, const T* const y, const int n);

	template< typename T >
	T sqr_sum(const T* const x, const int n);

	template< typename T >
	T mpi_sqr_sum(const T* const x, const int n);

	template< typename T >
	void sqr_sum_and_dp(const T* const x, const T* const y, const int n, T* sum, T* dp);

	template< typename T >
	void mpi_sqr_sum_and_dp(const T* const x, const T* const y, const int n, T* sum, T* dp);

	template< typename T >
	T lnorm(const T* const x, const int n);

	template< typename T >
	T mpi_lnorm(const T* const x, const int n);

	template< typename T >
	void lnorm_and_dp(const T* const x, const T* const y, const int n, T* norm, T* dp);

	template< typename T >
	void mpi_lnorm_and_dp(const T* const x, const T* const y, const int n, T* norm, T* dp);

	template< typename T >
	void lnorm_and_sqr_sum(const T* const x, const int n, T* norm, T* sum);

	template< typename T >
	void mpi_lnorm_and_sqr_sum(const T* const x, const int n, T* norm, T* sum);

	template< typename T >
	T cnorm(const T* const x, const int n);
	
	template< typename T >
	T mpi_cnorm(const T* const x, const int n);

	template< typename T >
	void cnorm_and_dp(const T* const x, const T* const y, const int n, T* norm, T* dp);

	template< typename T >
	void mpi_cnorm_and_dp(const T* const x, const T* const y, const int n, T* norm, T* dp);

	template< typename T >
	void cnorm_and_sqr_sum(const T* const x, const int n, T* norm, T* sum);

	template< typename T >
	void mpi_cnorm_and_sqr_sum(const T* const x, const int n, T* norm, T* sum);

	template< typename T >
	T l1norm(const T* const x, const int n);

	template< typename T >
	T mpi_l1norm(const T* const x, const int n);

	template< typename T >
	void null(T* x, const int n);

	template< typename T >
	void update(T* x,
		const T alpha, const int n);

	template< typename T >
	void update(T* _VEC_RESTRICT x,
		const T alpha, const T* const y, const int n);

	template< typename T >
	void update(T* _VEC_RESTRICT x,
		const T alpha, const T* const y,
		const T beta, const T* const z, const int n);

	template< typename T >
	void update(T* _VEC_RESTRICT x,
		const T alpha, const T* const y,
		const T beta, const T* const z,
		const T gamma, const T* const w, const int n);

	template< typename T >
	void update(T* _VEC_RESTRICT x, T* _VEC_RESTRICT y,
		const T alpha, const T beta,
		const T* const z, const T* const w, const int n);

	void update_sse(float* _VEC_RESTRICT x, float* _VEC_RESTRICT y,
		const float alpha, const float beta,
		const float* const z, const float* const w, const int n);
	void update_sse(double* _VEC_RESTRICT x, double* _VEC_RESTRICT y,
		const double alpha, const double beta,
		const double* const z, const double* const w, const int n);

	template< typename T >
	void assign(T* x, const T alpha, const int n);

	template< typename T >
	void assign(T* _VEC_RESTRICT x,
		const T alpha, const T* const y, const int n);

	template< typename T >
	void assign(T* _VEC_RESTRICT x,
		const T alpha, const T* const y,
		const T beta, const T* const z, const int n);

	template< typename T >
	void assign(T* _VEC_RESTRICT x,
		const T alpha, const T* const y,
		const T beta, const T* const z,
		const T gamma, const T* const w, const int n);

	template< typename T >
	void mul(T* x, const T value, const int n);

	template< typename T >
	void mul(T* _VEC_RESTRICT y,
		const T* const x, const T* const z, const int n);

	template< typename T >  // y[n] = matrix[n * n] * x[n]
	void matvec(T* _VEC_RESTRICT y,
		const T* const matrix, const T* const x, const int n);

	template< typename T >  // y[n] = x[n] * matrix[n * n]
	void vecmat(T* _VEC_RESTRICT y,
		const T* const x, const T* const matrix, const int n);

	template< typename T >  // y[n] = rhs[ n ] - matrix[n * n] * x[n]
	void resvec(T* _VEC_RESTRICT res,
		const T* const rhs, const T* const matrix, const T* const x, const int n);

	template< typename T > // ssor preconditioner
	void ssor(T* _VEC_RESTRICT x,
		const T* const rhs, const T* const i_diagonal, const T* const matrix,
		const T omega, const int piters, const int n);

	// * matrix operations * //
	template< typename T >
	T det(const T a11, const T a12,
		const T a21, const T a22);

	template< typename T >
	T det(const T a11, const T a12, const T a13,
		const T a21, const T a22, const T a23,
		const T a31, const T a32, const T a33);

	template< typename T >
	T det(const T a11, const T a12, const T a13, const T a14,
		const T a21, const T a22, const T a23, const T a24,
		const T a31, const T a32, const T a33, const T a34,
		const T a41, const T a42, const T a43, const T a44);

	template< typename T >  // matrix[n * n]
	T det(const T* const matrix, const int n);

	template< typename T >  // matrix[n * n] cofactor(i, j) determinant 
	T cofactor(const int i, const int j,
		const T* const matrix, const int size);

	template< typename T >  // matrix[n * n] inverse
	bool inverse(T* _VEC_RESTRICT inv_matrix,
		const T* const matrix, const int n);

	// * interpolation * //
	template< typename T >
	T interp_bilinear(const T x, const T y,
		const T v00, const T v10,       // - x line
		const T v01, const T v11);     // - x line with y shift

	template< typename T >
	T interp_bilinear(const T x, const T y,
		const T sx, const T sy, const T dx, const T dy,
		const T v00, const T v10,       // - x line
		const T v01, const T v11);     // - x line with y shift

	template< typename T >
	T interp_trilinear(const T x, const T y, const T z,
		const T v000, const T v100,     // - x line
		const T v010, const T v110,     // - x line with y shift
		const T v001, const T v101,     // - x line with z shift
		const T v011, const T v111);   // - x line with y,z shift

	template< typename T >
	T interp_trilinear(const T x, const T y, const T z,
		const T sx, const T sy, const T sz, const T dx, const T dy, const T dz,
		const T v000, const T v100,     // - x line
		const T v010, const T v110,     // - x line with y shift
		const T v001, const T v101,     // - x line with z shift
		const T v011, const T v111);   // - x line with y,z shift

	// * ODE solvers * //
	template< typename T >
	void runge_kutta_o4(T* ynext, T* unext,
		const T y0, const T u0, const T f,
		const T alpha, const T beta,

		const T dt, const int niters);

	template< typename T >
	// * conjugate gradients (jacobi), [memory req. = 4 * n] * //
	int cg_jacobi(T* _VEC_RESTRICT x, 
		const T* const rhs, const T* const matrix, const T* const jacobi, T* memory,
		const int n, const T retol, const T abstol, const int maxiters, T* error);

	template< typename T >
	// * conjugate gradients (ssor), [memory req. = 4 * n] * //
	int cg_ssor(T* _VEC_RESTRICT x, 
		const T* const rhs, const T* const matrix, const T* const jacobi, T* memory,
		const int n, const T retol, const T abstol, const int maxiters, T* error);

	template< typename T >
	// * bi-conjugate gradients stabilized (jacobi), [memory req. = 6 * n] * //
	int bicg_jacobi(T* _VEC_RESTRICT x, 
		const T* const rhs, const T* const matrix, const T* const jacobi, T* memory,
		const int n, const T retol, const T abstol, const int maxiters, T* error);

	template< typename T >
	// * bi-conjugate gradients stabilized (ssor), [memory req. = 6 * n] * //
	int bicg_ssor(T* _VEC_RESTRICT x, 
		const T* const rhs, const T* const matrix, const T* const jacobi, T* memory,
		const int n, const T retol, const T abstol, const int maxiters, T* error);
}

namespace nse
{
	template< typename T >
	inline T sqr(
		const T a)
	{
		return a * a;
	}
    
	template< typename T > 
	inline T min(
		const T a, const T b) 
	{
		return (a < b) ? a : b;
	}

	template< typename T > 
	inline T max(
		const T a, const T b) 
	{
		return (a > b) ? a : b;
	}

	template< typename T > 
	inline T min(
		const T a, const T b, const T c)
	{
		return (a < b) ? ((a < c) ? a : c) : ((b < c) ? b : c);
	}

	template< typename T > 
	inline T max(
		const T a, const T b, const T c)
	{
		return (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c);
	}

	template< typename T > 
	inline T min(
		const T a, const T b, const T c, const T d) 
	{
		return min(min(a, b, c), d);
	}

	template< typename T > 
	inline T max(
		const T a, const T b, const T c, const T d) 
	{
		return max(max(a, b, c), d);
	}

	template< typename T > 
	inline T min(
		const T a, const T b, const T c, const T d, const T e) 
	{
		return min(min(a, b), min(c, d, e));
	}

	template< typename T > 
	inline T max(
		const T a, const T b, const T c, const T d, const T e) 
	{
		return max(max(a, b), max(c, d, e));
	}


	template< typename T > 
	inline T sign(
		const T x)
	{
		return (x > (T)0) ? (T)1 :
			((x < (T)0) ? (T)-1 : (T)0);
	}

	template< typename T > 
	inline T sign(const T x, const T eps)
	{
		if (fabs(x) < eps) return (T)0;
		else
			return (x < (T)0) ? -(T)1 : (T)1;
	}

	template< typename T > 
	inline T minmod_limit(const T x, const T y)
	{
		if (x * y <= (T)0) return (T)0;

		return (fabs(y) < fabs(x)) ? y : x;
	}

	template< typename T > 
	inline T superbee_limit(const T x, const T y)
	{
		if (x * y <= (T)0) return (T)0;

		const T xabs = fabs(x), yabs = fabs(y);
		const T sig_x = (T)(((T)0 < x) - (x < (T)0));

		if ((xabs > yabs + yabs) || (yabs > xabs + xabs))
			return (T) 2.0 * sig_x * ((xabs < yabs) ? xabs : yabs);
		else
			return sig_x * ((xabs > yabs) ? xabs : yabs);
	}

	template< typename T > 
	inline T delta(const T x, const T eps)
	{
		if ((x > eps) || (x < -eps)) return (T) 0.0;

		T ieps = (T)1 / eps;
		return (T) 0.5 * (
			(T)1 + cos((T)M_PI * x * ieps)) * ieps;
	}

	template< typename T > 
	inline T heavy_side(const T x)
	{
		return (x < (T)0) ? (T)0 : (T)1;
	}

	template< typename T > 
	inline T heavy_side(const T x, const T eps)
	{
		if (x < -eps) return (T)0;
		if (x > eps) return (T)1;

		const T z = x / eps;

		return (T) 0.5 * (
			(T)1 + z + (sin((T)M_PI * z) * (T)M_1_PI));
	}

	template< typename T > 
	inline T linear_step(const T x, const T eps)
	{
		if (x < -eps) return (T)0;
		if (x > eps) return (T)1;

		return (T) 0.5 * ((T)1 + (x / eps));

	}

	template< typename T > 
	inline void allocate(
		T** x, const int n)
	{
#ifndef _ALLOCATE_ALIGNED
		(*x) = new T[n];
#else
#ifdef __INTEL_COMPILER
		(*x) = (T*)_mm_malloc(
			n * sizeof(T), _USE_ALIGNMENT);
#else
		(*x) = (T*)_aligned_malloc(
			n * sizeof(T), _USE_ALIGNMENT);
#endif
#endif
		null((*x), n);
	}

	template< typename T > 
	inline void allocate(
		T** x, T** y, const int n)
	{
#ifndef _ALLOCATE_ALIGNED
		(*x) = new T[n];
		(*y) = new T[n];
#else
#ifdef __INTEL_COMPILER
		(*x) = (T*)_mm_malloc(
			n * sizeof(T), _USE_ALIGNMENT);
		(*y) = (T*)_mm_malloc(
			n * sizeof(T), _USE_ALIGNMENT);
#else
		(*x) = (T*)_aligned_malloc(
			n * sizeof(T), _USE_ALIGNMENT);
		(*y) = (T*)_aligned_malloc(
			n * sizeof(T), _USE_ALIGNMENT);
#endif
#endif
		null((*x), n);
		null((*y), n);
	}

	template< typename T > 
	inline void allocate(
		T** x, T** y, T** z, const int n)
	{
#ifndef _ALLOCATE_ALIGNED
		(*x) = new T[n];
		(*y) = new T[n];
		(*z) = new T[n];
#else
#ifdef __INTEL_COMPILER
		(*x) = (T*)_mm_malloc(
			n * sizeof(T), _USE_ALIGNMENT);
		(*y) = (T*)_mm_malloc(
			n * sizeof(T), _USE_ALIGNMENT);
		(*z) = (T*)_mm_malloc(
			n * sizeof(T), _USE_ALIGNMENT);
#else
		(*x) = (T*)_aligned_malloc(
			n * sizeof(T), _USE_ALIGNMENT);
		(*y) = (T*)_aligned_malloc(
			n * sizeof(T), _USE_ALIGNMENT);
		(*z) = (T*)_aligned_malloc(
			n * sizeof(T), _USE_ALIGNMENT);
#endif
#endif
		null((*x), n);
		null((*y), n);
		null((*z), n);
	}

	template< typename T > 
	inline void deallocate(
		T* x)
	{
#ifndef _ALLOCATE_ALIGNED
		delete[] x;
#else
#ifdef __INTEL_COMPILER
		_mm_free((void*)x);
#else
		_aligned_free((void*)x);
#endif
#endif
	}

	template< typename T > 
	inline void deallocate(
		T* x, T* y)
	{
#ifndef _ALLOCATE_ALIGNED
		delete[] x;
		delete[] y;
#else
#ifdef __INTEL_COMPILER
		_mm_free((void*)x);
		_mm_free((void*)y);
#else
		_aligned_free((void*)x);
		_aligned_free((void*)y);
#endif
#endif
	}

	template< typename T > 
	inline void deallocate(
		T* x, T* y, T* z)
	{
#ifndef _ALLOCATE_ALIGNED
		delete[] x;
		delete[] y;
		delete[] z;
#else
#ifdef __INTEL_COMPILER
		_mm_free((void*)x);
		_mm_free((void*)y);
		_mm_free((void*)z);
#else
		_aligned_free((void*)x);
		_aligned_free((void*)y);
		_aligned_free((void*)z);
#endif
#endif
	}

	template< typename T > 
	inline T min(
		const T* const x, const int n)
	{
		if (n <= 0) return (T)0;

		int i;
		T _min = (T)(*x);

		for (i = 1; i < n; i++)
		if (x[i] < _min) _min = x[i];

		return _min;
	}

	template< typename T > 
	inline T mpi_min(
		const T* const x, const int n)
	{
		return mpi_allreduce(min(x, n), MPI_MIN);
	}

	template< typename T > 
	inline T max(
		const T* const x, const int n)
	{
		if (n <= 0) return (T)0;

		int i;
		T _max = (T)(*x);
		
		for (i = 1; i < n; i++)
		if (x[i] > _max) _max = x[i];

		return _max;
	}

	template< typename T > 
	inline T mpi_max(
		const T* const x, const int n)
	{
		return mpi_allreduce(max(x, n), MPI_MAX);
	}

	template< typename T > 
	inline T sum(
		const T* const x, const int n)
	{
		int i;
		T _sum = (T)0;

#pragma omp parallel for private( i ) reduction( + : _sum )
		for (i = 0; i < n - (n % 4); i += 4) {
			_sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
		}

		for (i = n - (n % 4); i < n; i++)
			_sum += x[i];

		return _sum;
	}

	template< typename T > 
	inline T mpi_sum(
		const T* const x, const int n)
	{
		return mpi_allreduce(sum(x, n), MPI_SUM);
	}

	template< typename T > 
	inline T dot_product(
		const T* const x, const T* const y, const int n)
	{
		int i;
		T _dp = (T)0;

#pragma omp parallel for private( i ) reduction( + : _dp )
		for (i = 0; i < n - (n % 4); i += 4) {
			_dp += x[i] * y[i] +
				x[i + 1] * y[i + 1] +
				x[i + 2] * y[i + 2] +
				x[i + 3] * y[i + 3];
		}

		for (i = n - (n % 4); i < n; i++)
			_dp += x[i] * y[i];
		return _dp;
	}

	template< typename T > 
	inline T mpi_dot_product(
		const T* const x, const T* const y, const int n)
	{
		return mpi_allreduce(dot_product(x, y, n), MPI_SUM);
	}

	template< typename T > 
	inline T sqr_sum(
		const T* const x, const int n)
	{
		int i;
		T _sum = (T)0;

#pragma omp parallel for private( i ) reduction( + : _sum )
		for (i = 0; i < n - (n % 4); i += 4) {
			_sum += x[i] * x[i] +
				x[i + 1] * x[i + 1] +
				x[i + 2] * x[i + 2] +
				x[i + 3] * x[i + 3];
		}

		for (i = n - (n % 4); i < n; i++)
			_sum += x[i] * x[i];

		return _sum;
	}

	template< typename T > 
	inline T mpi_sqr_sum(
		const T* const x, const int n)
	{
		return mpi_allreduce(sqr_sum(x, n), MPI_SUM);
	}

	template< typename T > 
	inline void sqr_sum_and_dp(
		const T* const x, const T* const y, const int n, T* sum, T* dp)
	{
		int i;
		T _dp = (T)0, _sum = (T)0;

		T Xi, Xip, Xipp, Xippp;

#pragma omp parallel for private( i, Xi, Xip, Xipp, Xippp ) reduction( + : _dp, _sum )
		for (i = 0; i < n - (n % 4); i += 4) {
			Xi = x[i];
			Xip = x[i + 1];
			Xipp = x[i + 2];
			Xippp = x[i + 3];

			_sum += Xi * Xi +
				Xip * Xip +
				Xipp * Xipp +
				Xippp * Xippp;

			_dp += Xi * y[i] +
				Xip * y[i + 1] +
				Xipp * y[i + 2] +
				Xippp * y[i + 3];
		}

		for (i = n - (n % 4); i < n; i++) {
			_sum += x[i] * x[i];
			_dp += x[i] * y[i];
		}

		(*dp) = _dp;
		(*sum) = _sum;
	}

	template< typename T > 
	inline void mpi_sqr_sum_and_dp(
		const T* const x, const T* const y, const int n, T* sum, T* dp)
	{
		sqr_sum_and_dp(x, y, n, sum, dp);
		mpi_allreduce(sum, dp, MPI_SUM);
	}

	template< typename T > 
	inline T lnorm(
		const T* const x, const int n)
	{
		int i;
		T _norm = (T)0;

#pragma omp parallel for private( i ) reduction( + : _norm )
		for (i = 0; i < n - (n % 4); i += 4) {
			_norm += x[i] * x[i] +
				x[i + 1] * x[i + 1] +
				x[i + 2] * x[i + 2] +
				x[i + 3] * x[i + 3];
		}

		for (i = n - (n % 4); i < n; i++)
			_norm += x[i] * x[i];

		return sqrt(_norm);
	}


	template< typename T > 
	inline T mpi_lnorm(
		const T* const x, const int n)
	{
		return sqrt(mpi_allreduce(sqr_sum(x, n), MPI_SUM));
	}

	template< typename T > 
	inline void lnorm_and_dp(
		const T* const x, const T* const y, const int n, T* norm, T* dp)
	{
		int i;
		T _dp = (T)0, _norm = (T)0;

		T Xi, Xip, Xipp, Xippp;

#pragma omp parallel for private( i, Xi, Xip, Xipp, Xippp ) reduction( + : _dp, _norm )
		for (i = 0; i < n - (n % 4); i += 4) {
			Xi = x[i];
			Xip = x[i + 1];
			Xipp = x[i + 2];
			Xippp = x[i + 3];

			_norm += Xi * Xi +
				Xip * Xip +
				Xipp * Xipp +
				Xippp * Xippp;

			_dp += Xi * y[i] +
				Xip * y[i + 1] +
				Xipp * y[i + 2] +
				Xippp * y[i + 3];
		}

		for (i = n - (n % 4); i < n; i++) {
			_norm += x[i] * x[i];
			_dp += x[i] * y[i];
		}

		(*dp) = _dp;
		(*norm) = sqrt(_norm);
	}

	template< typename T > 
	inline void mpi_lnorm_and_dp(
		const T* const x, const T* const y, const int n, T* norm, T* dp)
	{
		sqr_sum_and_dp(x, y, n, norm, dp);
		mpi_allreduce(norm, dp, MPI_SUM);

		(*norm) = sqrt((*norm));
	}

	template< typename T > 
	inline void lnorm_and_sqr_sum(
		const T* const x, const int n, T* norm, T* sum)
	{
		(*sum) = sqr_sum(x, n);
		(*norm) = sqrt((*sum));
	}

	template< typename T > 
	inline void mpi_lnorm_and_sqr_sum(
		const T* const x, const int n, T* norm, T* sum)
	{
		(*sum) = sqr_sum(x, n);
		mpi_allreduce(sum, MPI_SUM);

		(*norm) = sqrt((*sum));
	}

	template< typename T > 
	inline T cnorm(
		const T* const x, const int n)
	{
		int i;
		T _norm1 = (T)0,
			_norm2 = (T)0,
			_norm3 = (T)0,
			_norm4 = (T)0;

		for (i = 0; i < n - (n % 4); i += 4) {

			if (fabs(x[i]) > _norm1) _norm1 = fabs(x[i]);
			if (fabs(x[i + 1]) > _norm2) _norm2 = fabs(x[i + 1]);
			if (fabs(x[i + 2]) > _norm3) _norm3 = fabs(x[i + 2]);
			if (fabs(x[i + 3]) > _norm4) _norm4 = fabs(x[i + 3]);
		}

		for (i = n - (n % 4); i < n; i++) {
			if (fabs(x[i]) > _norm1) _norm1 = fabs(x[i]);
		}

		if (_norm1 < _norm3) _norm1 = _norm3;
		if (_norm2 < _norm4) _norm2 = _norm4;

		return (_norm1 > _norm2) ? _norm1 : _norm2;
	}

	template< typename T > 
	inline T mpi_cnorm(
		const T* const x, const int n)
	{
		return mpi_allreduce(cnorm(x, n), MPI_MAX);
	}

	template< typename T > 
	inline void cnorm_and_dp(
		const T* const x, const T* const y, const int n, T* norm, T* dp)
	{
		int i;
		T _dp = (T)0,
			_norm1 = (T)0,
			_norm2 = (T)0,
			_norm3 = (T)0,
			_norm4 = (T)0;

		for (i = 0; i < n - (n % 4); i += 4) {

			if (fabs(x[i]) > _norm1) _norm1 = fabs(x[i]);
			if (fabs(x[i + 1]) > _norm2) _norm2 = fabs(x[i + 1]);
			if (fabs(x[i + 2]) > _norm3) _norm3 = fabs(x[i + 2]);
			if (fabs(x[i + 3]) > _norm4) _norm4 = fabs(x[i + 3]);

			_dp += x[i] * y[i] +
				x[i + 1] * y[i + 1] +
				x[i + 2] * y[i + 2] +
				x[i + 3] * y[i + 3];
		}

		for (i = n - (n % 4); i < n; i++) {
			if (fabs(x[i]) > _norm1) _norm1 = fabs(x[i]);

			_dp += x[i] * y[i];
		}

		if (_norm1 < _norm3) _norm1 = _norm3;
		if (_norm2 < _norm4) _norm2 = _norm4;

		(*norm) = (_norm1 > _norm2) ? _norm1 : _norm2;
		(*dp) = _dp;
	}

	template< typename T > 
	inline void mpi_cnorm_and_dp(
		const T* const x, const T* const y, const int n, T* norm, T* dp)
	{
		cnorm_and_dp(x, y, n, norm, dp);

		mpi_allreduce(norm, MPI_MAX);
		mpi_allreduce(dp, MPI_SUM);
	}

	template< typename T > 
	inline void cnorm_and_sqr_sum(
		const T* const x, const int n, T* norm, T* sum)
	{
		int i;
		T _sum = (T)0,
			_norm1 = (T)0,
			_norm2 = (T)0,
			_norm3 = (T)0,
			_norm4 = (T)0;

		for (i = 0; i < n - (n % 4); i += 4) {

			if (fabs(x[i]) > _norm1) _norm1 = fabs(x[i]);
			if (fabs(x[i + 1]) > _norm2) _norm2 = fabs(x[i + 1]);
			if (fabs(x[i + 2]) > _norm3) _norm3 = fabs(x[i + 2]);
			if (fabs(x[i + 3]) > _norm4) _norm4 = fabs(x[i + 3]);

			_sum += x[i] * x[i] +
				x[i + 1] * x[i + 1] +
				x[i + 2] * x[i + 2] +
				x[i + 3] * x[i + 3];
		}

		for (i = n - (n % 4); i < n; i++) {
			if (fabs(x[i]) > _norm1) _norm1 = fabs(x[i]);

			_sum += x[i] * x[i];
		}

		if (_norm1 < _norm3) _norm1 = _norm3;
		if (_norm2 < _norm4) _norm2 = _norm4;

		(*norm) = (_norm1 > _norm2) ? _norm1 : _norm2;
		(*sum) = _sum;
	}

	template< typename T > 
	inline void mpi_cnorm_and_sqr_sum(
		const T* const x, const int n, T* norm, T* sum)
	{
		cnorm_and_sqr_sum(x, n, norm, sum);

		mpi_allreduce(norm, MPI_MAX);
		mpi_allreduce(sum, MPI_SUM);
	}

	template< typename T > 
	inline T l1norm(
		const T* const x, const int n)
	{
		int i;
		T _norm = (T)0;

#pragma omp parallel for private( i ) reduction( + : _norm )
		for (i = 0; i < n - (n % 4); i += 4) {
			_norm += fabs(x[i]) +
				fabs(x[i + 1]) +
				fabs(x[i + 2]) +
				fabs(x[i + 3]);
		}

		for (i = n - (n % 4); i < n; i++)
			_norm += fabs(x[i]);

		return _norm;
	}

	template< typename T > 
	inline T mpi_l1norm(
		const T* const x, const int n)
	{
		return mpi_allreduce(l1norm(x, n), MPI_SUM);
	}

	template< typename T > 
	inline void null(
		T* x, const int n)
	{
		int i;

#pragma omp parallel for private( i ) shared( x )
		for (i = 0; i < n - (n % 4); i += 4) {
			x[i] = (T)0;
			x[i + 1] = (T)0;
			x[i + 2] = (T)0;
			x[i + 3] = (T)0;
		}

		for (i = n - (n % 4); i < n; i++)
			x[i] = (T)0;
	}

	template< typename T > 
	inline void update(
		T* x,
		const T alpha, const int n)
	{
		int i;

#pragma omp parallel for private( i ) shared( x )
		for (i = 0; i < n - (n % 4); i += 4) {
			x[i] += alpha;
			x[i + 1] += alpha;
			x[i + 2] += alpha;
			x[i + 3] += alpha;
		}

		for (i = n - (n % 4); i < n; i++)
			x[i] += alpha;
	}

	template< typename T > 
	inline void update(
		T* _VEC_RESTRICT x,
		const T alpha, const T* const y, const int n)
	{
		int i;
#pragma omp parallel for private( i ) shared( x )
		for (i = 0; i < n; i++)
			x[i] += alpha * y[i];
	}

	template< typename T > 
	inline void update(
		T* _VEC_RESTRICT x,
		const T alpha, const T* const y,
		const T beta, const T* const z, const int n)
	{
		int i;

#pragma omp parallel for private( i ) shared( x )
		for (i = 0; i < n; i++)
			x[i] += alpha * y[i] + beta * z[i];
	}

	template< typename T > 
	inline void update(
		T* _VEC_RESTRICT x,
		const T alpha, const T* const y,
		const T beta, const T* const z,
		const T gamma, const T* const w, const int n)
	{
		int i;

#pragma omp parallel for private( i ) shared( x )
		for (i = 0; i < n; i++)
			x[i] += alpha * y[i] + beta * z[i] + gamma * w[i];
	}

	template< typename T > 
	inline void update(
		T* _VEC_RESTRICT x, T* _VEC_RESTRICT y,
		const T alpha, const T beta,
		const T* const z, const T* const w, const int n)
	{
		int i;

#pragma omp parallel private( i ) shared( x, y )
		{

#pragma omp for nowait
			for (i = 0; i < n - (n % 4); i += 4)
			{
				x[i] += alpha * z[i];
				x[i + 1] += alpha * z[i + 1];
				x[i + 2] += alpha * z[i + 2];
				x[i + 3] += alpha * z[i + 3];
			}

#pragma omp for nowait
			for (i = 0; i < n - (n % 4); i += 4)
			{
				y[i] += beta * w[i];
				y[i + 1] += beta * w[i + 1];
				y[i + 2] += beta * w[i + 2];
				y[i + 3] += beta * w[i + 3];
			}
		}

		for (i = n - (n % 4); i < n; i++)
		{
			x[i] += alpha * z[i];
			y[i] += beta * w[i];
		}
	}

	inline void update_sse(
		float* _VEC_RESTRICT x, float* _VEC_RESTRICT y,
		const float alpha, const float beta,
		const float* const z, const float* const w, const int n)
	{
		__m128 m_z, m_w, m_x, m_y;
		__m128 m_alpha = _mm_set_ps1(alpha);
		__m128 m_beta = _mm_set_ps1(beta);

		int i;

#pragma omp parallel for private( i, m_z, m_w, m_x, m_y ) shared( x, y, m_alpha, m_beta )
		for (i = 0; i < n - (n % 4); i += 4) {
			m_z = _mm_load_ps(&z[i]);
			m_w = _mm_load_ps(&w[i]);
			m_x = _mm_load_ps(&x[i]);
			m_y = _mm_load_ps(&y[i]);

			m_z = _mm_mul_ps(m_z, m_alpha);
			m_w = _mm_mul_ps(m_w, m_beta);

			m_x = _mm_add_ps(m_x, m_z);
			m_y = _mm_add_ps(m_y, m_w);

			_mm_store_ps(&x[i], m_x);
			_mm_store_ps(&y[i], m_y);
		}

		for (i = n - (n % 4); i < n; i++) {
			x[i] += alpha * z[i];
			y[i] += beta * w[i];
		}
	}

	inline void update_sse(
		double* _VEC_RESTRICT x, double* _VEC_RESTRICT y,
		const double alpha, const double beta,
		const double* const z, const double* const w, const int n)
	{
		__m128d m_z, m_w, m_x, m_y;
		__m128d m_alpha = _mm_set1_pd(alpha);
		__m128d m_beta = _mm_set1_pd(beta);

		int i;

#pragma omp parallel for private( i, m_z, m_w, m_x, m_y ) shared( x, y, m_alpha, m_beta )
		for (i = 0; i < n - (n % 2); i += 2) {
			m_z = _mm_load_pd(&z[i]);
			m_w = _mm_load_pd(&w[i]);
			m_x = _mm_load_pd(&x[i]);
			m_y = _mm_load_pd(&y[i]);

			m_z = _mm_mul_pd(m_z, m_alpha);
			m_w = _mm_mul_pd(m_w, m_beta);

			m_x = _mm_add_pd(m_x, m_z);
			m_y = _mm_add_pd(m_y, m_w);

			_mm_store_pd(&x[i], m_x);
			_mm_store_pd(&y[i], m_y);
		}

		for (i = n - (n % 2); i < n; i++) {
			x[i] += alpha * z[i];
			y[i] += beta * w[i];
		}
	}

	template< typename T > 
	inline void assign(
		T* x,
		const T alpha, const int n)
	{
		int i;

#pragma omp parallel for private( i ) shared( x )
		for (i = 0; i < n - (n % 4); i += 4) {
			x[i] = alpha;
			x[i + 1] = alpha;
			x[i + 2] = alpha;
			x[i + 3] = alpha;
		}

		for (i = n - (n % 4); i < n; i++)
			x[i] = alpha;
	}

	template< typename T > 
	inline void assign(
		T* _VEC_RESTRICT x,
		const T alpha, const T* const y, const int n)
	{
		int i;

#pragma omp parallel for private( i ) shared( x )
		for (i = 0; i < n; i++)
			x[i] = alpha * y[i];
	}

	template< typename T > 
	inline void assign(
		T* _VEC_RESTRICT x,
		const T alpha, const T* const y,
		const T beta, const T* const z, const int n)
	{
		int i;

#pragma omp parallel for private( i ) shared( x )
		for (i = 0; i < n; i++)
			x[i] = alpha * y[i] + beta * z[i];
	}

	template< typename T > 
	inline void assign(
		T* _VEC_RESTRICT x,
		const T alpha, const T* const y,
		const T beta, const T* const z,
		const T gamma, const T* const w, const int n)
	{
		int i;

#pragma omp parallel for private( i ) shared( x )
		for (i = 0; i < n; i++)
			x[i] = alpha * y[i] + beta * z[i] + gamma * w[i];
	}

	template< typename T > 
	inline void mul(
		T* x, const T value, const int n)
	{
		int i;

#pragma omp parallel for private( i ) shared( x )
		for (i = 0; i < n - (n % 4); i += 4) {
			x[i] *= value;
			x[i + 1] *= value;
			x[i + 2] *= value;
			x[i + 3] *= value;
		}

		for (i = n - (n % 4); i < n; i++)
			x[i] *= value;
	}

	template< typename T > 
	inline void mul(
		T* _VEC_RESTRICT y,
		const T* const x, const T* const z, const int n)
	{
		int i;

#pragma omp parallel for private( i ) shared( y )
		for (i = 0; i < n; i++)
			y[i] = x[i] * z[i];
	}

	template< typename T > 
	inline void matvec(
		T* _VEC_RESTRICT y,
		const T* const matrix, const T* const x, const int n)
	{
		int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( y )
		for (i = 0; i < n; i++) {
			y[i] = (T)0;

			idx = i * n;
			for (j = 0; j < n; j++, idx++)
				y[i] += matrix[idx] * x[j];
		}
	}

	template< typename T > 
	inline void vecmat(
		T* _VEC_RESTRICT y,
		const T* const x, const T* const matrix, const int n)
	{
		int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( y )
		for (i = 0; i < n; i++) {
			y[i] = (T)0;

			idx = i;
			for (j = 0; j < n; j++, idx += n)
				y[i] += matrix[idx] * x[j];
		}
	}

	template< typename T > 
	inline void resvec(
		T* _VEC_RESTRICT res,
		const T* const rhs, const T* const matrix, const T* x, const int n)
	{
		int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( res )
		for (i = 0; i < n; i++) {
			res[i] = rhs[i];

			idx = i * n;
			for (j = 0; j < n; j++, idx++)
				res[i] -= matrix[idx] * x[j];
		}
	}

	template< typename T > 
	inline void ssor(
		T* _VEC_RESTRICT x,
		const T* const rhs, const T* const i_diagonal, const T* const matrix,
		const T omega, const int piters, const int n)
	{
		int i, j, k, idx;

		T mv_down, mv_up;
		for (i = 0; i < n; i++) {
			idx = i * n;

			mv_down = rhs[i];
			for (j = 0; j < i; j++)
				mv_down -= matrix[idx + j] * x[j];

			x[i] = omega * i_diagonal[i] * mv_down;
		}

		for (i = n - 1; i >= 0; i--) {
			idx = i * n;

			mv_down = rhs[i];
			mv_up = (T)0;
			for (j = 0; j < i; j++)
				mv_down -= matrix[idx + j] * x[j];
			for (j = i + 1; j < n; j++)
				mv_up -= matrix[idx + j] * x[j];

			x[i] += omega * (i_diagonal[i] * (mv_up + mv_down) - x[i]);
		}

		for (k = 0; k < piters - 1; k++) {

			for (i = 0; i < n; i++) {
				idx = i * n;

				mv_down = rhs[i];
				mv_up = (T)0;
				for (j = 0; j < i; j++)
					mv_down -= matrix[idx + j] * x[j];
				for (j = i + 1; j < n; j++)
					mv_up -= matrix[idx + j] * x[j];

				x[i] += omega * (i_diagonal[i] * (mv_up + mv_down) - x[i]);
			}

			for (i = n - 1; i >= 0; i--) {
				idx = i * n;

				mv_down = rhs[i];
				for (j = 0; j < i; j++)
					mv_down -= matrix[idx + j] * x[j];
				for (j = i + 1; j < n; j++)
					mv_up -= matrix[idx + j] * x[j];

				x[i] += omega * (i_diagonal[i] * (mv_up + mv_down) - x[i]);
			}
		}
	}

	// * matrix operations * //
	// --------------------- //
	template< typename T > 
	inline T det(
		const T a11, const T a12,
		const T a21, const T a22)
	{
		return a11 * a22 - a12 * a21;
	}

	template< typename T > 
	inline T det(
		const T a11, const T a12, const T a13,
		const T a21, const T a22, const T a23,
		const T a31, const T a32, const T a33)
	{
		return a11 * (a22 * a33 - a23 * a32)
			- a12 * (a21 * a33 - a23 * a31)
			+ a13 * (a21 * a32 - a22 * a31);
	}

	template< typename T > 
	inline T det(
		const T a11, const T a12, const T a13, const T a14,
		const T a21, const T a22, const T a23, const T a24,
		const T a31, const T a32, const T a33, const T a34,
		const T a41, const T a42, const T a43, const T a44)
	{
		return	(a11 * a22 - a12 * a21) * (a33 * a44 - a34 * a43)
			- (a11 * a23 - a13 * a21) * (a32 * a44 - a34 * a42)
			+ (a11 * a24 - a14 * a21) * (a32 * a43 - a33 * a42)
			+ (a12 * a23 - a13 * a22) * (a31 * a44 - a34 * a41)
			- (a12 * a24 - a14 * a22) * (a31 * a43 - a33 * a41)
			+ (a13 * a24 - a14 * a23) * (a31 * a42 - a32 * a41);
	}

	template< typename T >
	T det( // matrix[n * n] determinant
		const T* const matrix, const int n)
	{
		if (n <= 0) return (T)0;
		if (n == 1) return (*matrix);

		if (n == 2)
			return det(matrix[0], matrix[1],
			matrix[2], matrix[3]);

		if (n == 3)
			return det(matrix[0], matrix[1], matrix[2],
			matrix[3], matrix[4], matrix[5],
			matrix[6], matrix[7], matrix[8]);

		if (n == 4)
			return det(matrix[0], matrix[1], matrix[2], matrix[3],
			matrix[4], matrix[5], matrix[6], matrix[7],
			matrix[8], matrix[9], matrix[10], matrix[11],
			matrix[12], matrix[13], matrix[14], matrix[15]);

		T *sub = new T[(n - 1) * (n - 1)];
		int i, j, k, sub_idx;
		T sub_det = (T)0, sub_sign;

		for (k = 0; k < n; k++) {
			for (i = 1; i < n; i++) {
				sub_idx = (i - 1) * (n - 1);
				for (j = 0; j < n; j++) {
					if (j == k) continue;

					sub[sub_idx] = matrix[i * n + j];
					sub_idx++;
				}
			}
			sub_sign = ((k % 2) == 0) ? (T) 1.0 : -(T) 1.0;
			sub_det += sub_sign * matrix[k] * det(sub, n - 1);
		}

		delete[] sub;
		return sub_det;
	}

	template< typename T >
	T cofactor( // matrix[n * n] cofactor(i, j) determinant
		const int ic, const int jc, const T* const matrix, const int n)
	{
		if (n <= 0) return (T)0;

		int i, j, k, sub_idx;

		T *sub = new T[(n - 1) * (n - 1)];
		T sub_det = (T)0;
		T sub_sign = (((ic + jc) % 2) == 0) ? (T) 1.0 : -(T) 1.0;

		k = 0;
		for (i = 0; i < n; i++) {
			if ((i + 1) == ic) continue;

			sub_idx = k * (n - 1);
			for (j = 0; j < n; j++) {
				if ((j + 1) == jc) continue;
				sub[sub_idx] = matrix[i * n + j];
				sub_idx++;
			}
			k++;
		}
		sub_det = det(sub, n - 1);

		delete[] sub;
		return sub_sign * sub_det;
	}

	template< typename T >
	bool inverse( // matrix[n * n] inverse
		T* _VEC_RESTRICT inv_matrix,
		const T* const matrix, const int n)
	{
		T _det = det(matrix, n);
		if (fabs(_det) == (T)0) return false;

		T inv_det = (T)1 / _det;
		int i, j, idx;
		for (i = 0; i < n; i++)
		{
			idx = i;
			for (j = 0; j < n; j++, idx += n)
				inv_matrix[idx] = inv_det *
				cofactor(i + 1, j + 1, matrix, n);
		}

		return true;
	}

	// * interpolation * //
	// ----------------- //
	template< typename T >
	inline T interp_bilinear(
		const T x, const T y,
		const T v00, const T v10,
		const T v01, const T v11)
	{
		return v00 +
			x * (v10 - v00) +
			y * (v01 - v00) +
			x * y * (v00 - v10 - v01 + v11);
	}

	template< typename T >
	inline T interp_bilinear(
		const T x, const T y,
		const T sx, const T sy, const T dx, const T dy,
		const T v00, const T v10,
		const T v01, const T v11)
	{
		return interp_bilinear((x - sx) / dx, (y - sy) / dy,
			v00, v10, v01, v11);
	}

	template< typename T >
	inline T interp_trilinear(
		const T x, const T y, const T z,
		const T v000, const T v100,
		const T v010, const T v110,
		const T v001, const T v101,
		const T v011, const T v111)
	{
		const T C0 = v000,
			C1 = v100 - v000,
			C2 = v010 - v000,
			C3 = v001 - v000,
			C4 = v110 - v010 - v100 + v000,
			C5 = v011 - v001 - v010 + v000,
			C6 = v101 - v001 - v100 + v000,
			C7 = v111 - v011 - v101 - v110 + v100 + v001 + v010 - v000;

		return C0 + C1 * x + C2 * y + C3 * z +
			C4 * x * y + C5 * y * z + C6 * x * z +
			C7 * x * y * z;
	}

	template< typename T >
	inline T interp_trilinear(
		const T x, const T y, const T z,
		const T sx, const T sy, const T sz,
		const T dx, const T dy, const T dz,
		const T v000, const T v100,
		const T v010, const T v110,
		const T v001, const T v101,
		const T v011, const T v111)
	{
		return interp_trilinear(
			(x - sx) / dx, (y - sy) / dy, (z - sz) / dz,
			v000, v100, v010, v110, v001, v101, v011, v111);
	}

	template< typename T >
	void runge_kutta_o4(
		T* ynext, T* unext,
		const T y0, const T u0, const T f,
		const T alpha, const T beta,

		const T dt, const int niters)
	{
		T y = y0, u = u0;

		T k1, k2, k3, k4,
			l1, l2, l3, l4;

		for (int i = 0; i < niters; i++) {
			k1 = dt * u;
			l1 = dt * (f - beta * y - alpha * u);

			k2 = dt * (u + (T) 0.5 * l1);
			l2 = dt * (f - beta * (y + (T) 0.5 * k1) - alpha * (u + (T) 0.5 * l1));

			k3 = dt * (u + (T) 0.5 * l2);
			l3 = dt * (f - beta * (y + (T) 0.5 * k2) - alpha * (u + (T) 0.5 * l2));

			k4 = dt * (u + l3);
			l4 = dt * (f - beta * (y + k3) - alpha * (u + l3));

			y += ((T) 1.0 / (T) 6.0) * (k1 + (T) 2.0 * k2 + (T) 2.0 * k3 + k4);
			u += ((T) 1.0 / (T) 6.0) * (l1 + (T) 2.0 * l2 + (T) 2.0 * l3 + l4);
		}

		(*ynext) = y;
		(*unext) = u;
	}
}

template< typename T >
int nse::cg_jacobi(T* x, const T* const rhs, const T* const matrix, const T* const jacobi, T* memory,
	const int n, const T retol, const T abstol, const int maxiters, T* resnorm)
{
	T alpha, beta, omega, rho, rho_star;
	T norm_star, norm_current = (T) 0.0;

	T *residual = memory,
		*p = &memory[n],
		*q = &memory[2 * n],
		*w = &memory[3 * n];

	resvec(residual, rhs, matrix, x, n);
	norm_star = lnorm(residual, n);
	if (norm_star < abstol) {
		(*resnorm) = norm_star;
		return 0;
	}

	mul(w, residual, jacobi, n);

	rho = dot_product(residual, w, n);
	memcpy(p, w, sizeof(T)* n);
	for (int k = 1; k <= maxiters; k++)
	{
		matvec(q, matrix, p, n);
		omega = dot_product(p, q, n);
		alpha = rho / omega;

		update(x, residual, alpha, -alpha, p, q, n);

		norm_current = lnorm(residual, n);
		if ((norm_current < retol * norm_star) || (norm_current < abstol)) {
			(*resnorm) = norm_current;
			return k;
		}

		mul(w, residual, jacobi, n);

		rho_star = rho;

		rho = dot_product(residual, w, n);
		beta = rho / rho_star;
		assign(p, (T) 1.0, w, beta, p, n);
	}

	(*resnorm) = norm_current;
	return -maxiters;
}

template< typename T >
int nse::cg_ssor(T* x, const T* const rhs, const T* const matrix, const T* const jacobi, T* memory,
	const int n, const T retol, const T abstol, const int maxiters, T* resnorm)
{
	const int ssor_piters = 1;
	const T ssor_omega = (T) 1.59;

	T alpha, beta, omega, rho, rho_star;
	T norm_star, norm_current = (T) 0.0;

	T *residual = memory,
		*p = &memory[n],
		*q = &memory[2 * n],
		*w = &memory[3 * n];

	resvec(residual, rhs, matrix, x, n);
	norm_star = lnorm(residual, n);
	if (norm_star < abstol) {
		(*resnorm) = norm_star;
		return 0;
	}

	ssor(w, residual, jacobi, matrix, ssor_omega, ssor_piters, n);

	rho = dot_product(residual, w, n);
	memcpy(p, w, sizeof(T)* n);
	for (int k = 1; k <= maxiters; k++)
	{
		matvec(q, matrix, p, n);
		omega = dot_product(p, q, n);
		alpha = rho / omega;

		update(x, residual, alpha, -alpha, p, q, n);

		norm_current = lnorm(residual, n);
		if ((norm_current < retol * norm_star) || (norm_current < abstol)) {
			(*resnorm) = norm_current;
			return k;
		}

		ssor(w, residual, jacobi, matrix, ssor_omega, ssor_piters, n);

		rho_star = rho;

		rho = dot_product(residual, w, n);
		beta = rho / rho_star;
		assign(p, (T) 1.0, w, beta, p, n);
	}

	(*resnorm) = norm_current;
	return -maxiters;
}

template< typename T >
int nse::bicg_jacobi(T* x, const T* const rhs, const T* const matrix, const T* const jacobi, T* memory,
	const int n, const T retol, const T abstol, const int maxiters, T* error)
{
	T alpha, beta, gamma, delta, epsilon, rho, rho_star;
	T norm_star, norm_current = (T) 0.0;

	T *residual = memory,
		*residual_star = &memory[n],
		*p = &memory[2 * n],
		*q = &memory[3 * n],
		*v = &memory[4 * n],
		*w = &memory[5 * n];


	resvec(residual, rhs, matrix, x, n);

	norm_star = lnorm(residual, n);
	if (norm_star < abstol) {
		(*error) = norm_star;
		return 0;
	}
	memcpy(residual_star, residual, n * sizeof(T));

	rho = dot_product(residual, residual_star, n);
	if (rho == (T)0) {
		(*error) = norm_star;
		return -1;
	}

	memcpy(p, residual, sizeof(T)* n);

	for (int k = 1; k <= maxiters; k++)
	{
		mul(q, p, jacobi, n);
		matvec(v, matrix, q, n);

		gamma = dot_product(residual_star, v, n);
		alpha = rho / gamma;

		update(x, residual, alpha, -alpha, q, v, n);

		// - additional skirmish norm check //
		/*
		norm_current = lnorm( residual, n );
		if ( norm_current < abstol ) {

		( *error ) = norm_current;
		return k;
		}
		*/
		// --------------------------------- //

		mul(q, residual, jacobi, n);
		matvec(w, matrix, q, n);

		delta = sqr_sum(w, n);
		epsilon = dot_product(w, residual, n);

		gamma = epsilon / delta;

		update(x, residual, gamma, -gamma, q, w, n);


		if (gamma == (T)0) {
			(*error) = norm_current;
			return -k - 1;
		}

		rho_star = rho;

		rho = dot_product(residual, residual_star, n);
		norm_current = lnorm(residual, n);

		if ((norm_current < retol * norm_star) || (norm_current < abstol)) {
			(*error) = norm_current;
			return k;
		}

		if (rho == (T)0) {
			(*error) = norm_current;
			return -k - 1;
		}

		beta = (rho / rho_star) * (alpha / gamma);
		assign(p, (T) 1.0, residual, beta, p, -gamma * beta, v, n);
	}

	(*error) = norm_current;
	return -maxiters;
}

template< typename T >
int nse::bicg_ssor(T* x, const T* const rhs, const T* const matrix, const T* const jacobi, T* memory,
	const int n, const T retol, const T abstol, const int maxiters, T* error)
{
	const int ssor_piters = 1;
	const T ssor_omega = (T) 1.59;

	T alpha, beta, gamma, delta, epsilon, rho, rho_star;
	T norm_star, norm_current = (T) 0.0;

	T *residual = memory,
		*residual_star = &memory[n],
		*p = &memory[2 * n],
		*q = &memory[3 * n],
		*v = &memory[4 * n],
		*w = &memory[5 * n];


	resvec(residual, rhs, matrix, x, n);

	norm_star = lnorm(residual, n);
	if (norm_star < abstol) {
		(*error) = norm_star;
		return 0;
	}
	memcpy(residual_star, residual, n * sizeof(T));

	rho = dot_product(residual, residual_star, n);
	if (rho == (T)0) {
		(*error) = norm_star;
		return -1;
	}

	memcpy(p, residual, sizeof(T)* n);

	for (int k = 1; k <= maxiters; k++)
	{
		ssor(q, p, jacobi, matrix, ssor_omega, ssor_piters, n);

		matvec(v, matrix, q, n);

		gamma = dot_product(residual_star, v, n);
		alpha = rho / gamma;

		update(x, residual, alpha, -alpha, q, v, n);

		// - additional skirmish norm check //
		/*
		norm_current = lnorm( residual, n );
		if ( norm_current < abstol ) {

		( *error ) = norm_current;
		return k;
		}
		*/
		// --------------------------------- //

		ssor(q, residual, jacobi, matrix, ssor_omega, ssor_piters, n);

		matvec(w, matrix, q, n);

		delta = sqr_sum(w, n);
		epsilon = dot_product(w, residual, n);

		gamma = epsilon / delta;

		update(x, residual, gamma, -gamma, q, w, n);


		if (gamma == (T)0) {
			(*error) = norm_current;
			return -k - 1;
		}

		rho_star = rho;

		rho = dot_product(residual, residual_star, n);
		norm_current = lnorm(residual, n);

		if ((norm_current < retol * norm_star) || (norm_current < abstol)) {
			(*error) = norm_current;
			return k;
		}

		if (rho == (T)0) {
			(*error) = norm_current;
			return -k - 1;
		}

		beta = (rho / rho_star) * (alpha / gamma);
		assign(p, (T) 1.0, residual, beta, p, -gamma * beta, v, n);
	}

	(*error) = norm_current;
	return -maxiters;
}
