#pragma once

#include <string.h>

// *[grid-common2d.h]: index removed //

namespace nse
{
	enum nodeType { nodeU = 0, nodeV = 1, nodeC = 2, nodeUV = 3 };

	// * null halo cells * //
	// ------------------- //
	template< typename T >
	void null_halo(T* x,
		const int nx, const int ny,
		const int gcx, const int gcy);

	// * apply -x, -y periodicity * //
	// ---------------------------- //
	template< typename T >
	void apply_periodic_x(T* x, const int nx, const int ny,
		const int gcx, const int gcy, const int hx, const int hy);
	template< typename T >
	void apply_periodic_y(T* x, const int nx, const int ny,
		const int gcx, const int gcy, const int hx, const int hy);

	// * apply -x, -y periodicity - colored * //
	// -------------------------------------- //
	template< typename T >
	void apply_periodic_x(T* x, const int color,
		const int nx, const int ny,
		const int gcx, const int gcy, const int hx, const int hy);
	template< typename T >
	void apply_periodic_y(T* x, const int color,
		const int nx, const int ny,
		const int gcx, const int gcy, const int hx, const int hy);

	// * get 2d sub array //
	// ------------------ //
	template< typename T >
	void get_sub_array(const T* in, const int nx, const int ny,
		const int ib, const int ie, const int jb, const int je, T* out);

	// * put 2d sub array //
	// ------------------ //
	template< typename T >
	void put_sub_array(T* out, const int nx, const int ny,
		const int ib, const int ie, const int jb, const int je, const T* in);

	// * get 2d sub array - colored //
	// ---------------------------- //
	template< typename T >
	void get_sub_array(const T* in, const int color,
		const int nx, const int ny,
		const int ib, const int ie, const int jb, const int je,
		T* out);

	// * put 2d sub array - colored //
	// ---------------------------- //
	template< typename T >
	void put_sub_array(T* out, const int color,
		const int nx, const int ny,
		const int ib, const int ie, const int jb, const int je,
		const T* in);


	// * get number of colored elements //
	// -------------------------------- //
	int get_num_colored(const int color,
		const int nx, const int ny,
		const int ib, const int ie, const int jb, const int je);
}

// * implementation: null halo cells * //
// ----------------------------------- //
template< typename T >
void nse::null_halo(
	T* x,
	const int nx, const int ny,
	const int gcx, const int gcy)
{
	int i, j, idx;

	// null columns //
	for (i = 0; i < gcx; i++) {

		idx = i * ny;
		for (j = 0; j < ny; j++, idx++) {
			x[idx] = (T)0;
		}

		idx = (nx - gcx + i) * ny;
		for (j = 0; j < ny; j++, idx++) {
			x[idx] = (T)0;
		}
	}

	// null rows
	for (j = 0; j < gcy; j++) {

		idx = gcx * ny + j;
		for (i = gcx; i < nx - gcx; i++, idx += ny) {
			x[idx] = (T)0;
			x[idx + ny - gcy] = (T)0;
		}
	}
}

// * apply -x, -y periodicity * //
// ---------------------------- //
template< typename T >
void nse::apply_periodic_x(T* x, const int nx, const int ny,
	const int gcx, const int gcy,
	const int hx, const int hy)
{
	const int stride = (nx - 2 * gcx) * ny;
	const int shx = hx * ny;
	const int jb = gcy - hy, je = ny - gcy + hy - 1;
	const int block_size = (je - jb + 1) * sizeof(T);

	int i, j, idx;

	if (block_size < 256) {     // magic number of inefficient memcpy()
#pragma omp parallel for private(i, j, idx) shared(x)
		for (i = gcx - hx; i < gcx; i++) {

			idx = i * ny;
			for (j = jb; j <= je; j++)	// west periodic //
				x[idx + j] = x[idx + stride + j];

			idx += stride + shx;
			for (j = jb; j <= je; j++)	// east periodic //
				x[idx + j] = x[idx - stride + j];
		}
	}
	else
	{
#pragma omp parallel for private(i, idx) shared(x)
		for (i = gcx - hx; i < gcx; i++) 
		{
			idx = i * ny + jb;
			memcpy(&x[idx],
				&x[idx + stride], block_size);	// west periodic //
			memcpy(&x[idx + stride + shx],
				&x[idx + shx], block_size);		// east periodic //
		}
	}
}
template< typename T >
void nse::apply_periodic_y(T* x, const int nx, const int ny,
	const int gcx, const int gcy,
	const int hx, const int hy)
{
	const int stride = (ny - 2 * gcy);
	const int ib = gcx - hx, ie = nx - gcx + hx - 1;

	int i, j, idx;

#pragma omp parallel for private(i, j, idx) shared(x)
	for (i = ib; i <= ie; i++) {

		idx = i * ny;
		for (j = gcy - hy; j < gcy; j++)				// south periodic //
			x[idx + j] = x[idx + stride + j];
		for (j = ny - gcy; j < ny - gcy + hy; j++)		// north periodic //
			x[idx + j] = x[idx - stride + j];
	}
}

// * apply -x, -y periodicity - colored * //
// -------------------------------------- //
template< typename T >
void nse::apply_periodic_x(T* x, const int color,
	const int nx, const int ny,
	const int gcx, const int gcy,
	const int hx, const int hy)
{
	const int stride = (nx - 2 * gcx) * ny;
	const int ish = hx + nx - 2 * gcx;
	const int shx = hx * ny;
	const int jb = gcy - hy, je = ny - gcy + hy - 1;

	int i, j, idx;
	int csh;

#pragma omp parallel for private(i, j, idx, csh) shared(x)
	for (i = gcx - hx; i < gcx; i++) {

		csh = (i + jb + color) & 1;
		idx = i * ny;
		for (j = jb + csh; j <= je; j += 2)		// west periodic //
			x[idx + j] = x[idx + stride + j];

		csh = (i + ish + jb + color) & 1;
		idx += stride + shx;
		for (j = jb + csh; j <= je; j += 2)		// east periodic //
			x[idx + j] = x[idx - stride + j];
	}
}
template< typename T >
void nse::apply_periodic_y(T* x, const int color,
	const int nx, const int ny,
	const int gcx, const int gcy,
	const int hx, const int hy)
{
	const int stride = (ny - 2 * gcy);
	const int ib = gcx - hx, ie = nx - gcx + hx - 1;

	int i, j, idx;
	int csh;

#pragma omp parallel for private(i, j, idx, csh) shared(x)
	for (i = ib; i <= ie; i++) {
		idx = i * ny;

		csh = (i + gcy - hy + color) & 1;
		for (j = gcy - hy + csh; j < gcy; j += 2)				// south periodic //
			x[idx + j] = x[idx + stride + j];

		csh = (i + ny - gcy + color) & 1;
		for (j = ny - gcy + csh; j < ny - gcy + hy; j += 2)		// north periodic //
			x[idx + j] = x[idx - stride + j];

	}
}

// * implementation: get 2d sub array * //
// ------------------------------------ //
template< typename T >
void nse::get_sub_array(const T* in, const int nx, const int ny,
	const int ib, const int ie, const int jb, const int je, T* out)
{
	const int cy = je - jb + 1;
	const int block_size = cy * sizeof(T);

	int i, j;
	int idx, odx;

	if (block_size < 256) {     // magic number of inefficient memcpy()

#pragma omp parallel for private( i, j, idx, odx ) shared( out, in )
		for (i = ib; i <= ie; i++)
		{
			idx = i * ny + jb;
			odx = (i - ib) * cy;
			for (j = jb; j <= je; j++, idx++, odx++)
				out[odx] = in[idx];
		}
	}
	else
	{
#pragma omp parallel for private( i ) shared( out, in )
		for (i = ib; i <= ie; i++)
			memcpy(&out[(i - ib) * cy], &in[i * ny + jb], block_size);
	}
}

// * implementation: put 2d sub array * //
// ------------------------------------ //
template< typename T >
void nse::put_sub_array(T* out, const int nx, const int ny,
	const int ib, const int ie, const int jb, const int je, const T* in)
{
	const int cy = je - jb + 1;
	const int block_size = cy * sizeof(T);

	int i, j;
	int idx, odx;

	if (block_size < 256) {     // magic number of inefficient memcpy()

#pragma omp parallel for private( i, j, idx, odx ) shared( out, in )
		for (i = ib; i <= ie; i++)
		{
			idx = (i - ib) * cy;
			odx = i * ny + jb;
			for (j = jb; j <= je; j++, idx++, odx++)
				out[odx] = in[idx];
		}
	}
	else
	{
#pragma omp parallel for private( i ) shared( out, in )
		for (i = ib; i <= ie; i++)
			memcpy(&out[i * ny + jb], &in[(i - ib) * cy], block_size);
	}
}

// * implementation: get 2d sub array - colored * //
// ---------------------------------------------- //
template< typename T >
void nse::get_sub_array(const T* in, const int color,
	const int nx, const int ny,
	const int ib, const int ie, const int jb, const int je,
	T* out)
{
	int i, j;
	int idx, odx = 0;
	int sh = (ib + jb + color) & 1;

	for (i = ib; i <= ie; i++) {

		idx = i * ny + jb + sh;
		for (j = jb + sh; j <= je; j += 2, odx++, idx += 2) {
			out[odx] = in[idx];
		}
		sh = !sh;       // change shift for next column //
	}
}

// * implementation: put 2d sub array - colored * //
// ---------------------------------------------- //
template< typename T >
void nse::put_sub_array(T* out, const int color,
	const int nx, const int ny,
	const int ib, const int ie, const int jb, const int je,
	const T* in)
{
	int i, j;
	int idx, odx = 0;
	int sh = (ib + jb + color) & 1;

	for (i = ib; i <= ie; i++) {

		idx = i * ny + jb + sh;
		for (j = jb + sh; j <= je; j += 2, odx++, idx += 2) {
			out[idx] = in[odx];
		}
		sh = !sh;       // change shift for next column //
	}
}

// * implementation: get number of colored elements //
// -------------------------------- //
inline
int nse::get_num_colored(const int color,
const int nx, const int ny,
const int ib, const int ie, const int jb, const int je)
{
	const int length = ie - ib + 1;
	const int width = je - jb + 1;

	if ((length & 1) && (width & 1)) {

		const int sh = !((ib + jb + color) & 1);
		return ((length - 1) >> 1) * width + ((width + sh) >> 1);
	}
	else
		return ((length * width) >> 1);
}
