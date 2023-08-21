#include "nse2d.h"

#include <math.h>
#include "vecmath.h"
#include <stdio.h>


template< typename T >
void nse::remove_const(
	T* w,
	const uniGrid2d< T >& grid)
{	
	T s = (T)0;
	T a;
	T alpha = -(T)1/(T)((grid.mpi_nx-2*grid.gcx)*(grid.mpi_ny-2*grid.gcy));
	T px, py;
	
	int i, j, idx;

	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		px = grid.x + (i - grid.gcx) * grid.dx + grid.dxh; // center of the cell
		py = grid.y + grid.dyh;
	  	idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
		      s += w[idx];
		      
		}
	}
	a = mpi_allreduce(s, MPI_SUM);
	alpha = alpha * a;
//	if (grid.mpi_com.rank == 0)
//	      printf("average w = %.16f \n", alpha);
	
	update(w, alpha, grid.size);
		
}

template< typename T >
void nse::check_const(
	const T* w, const char* message,
	const uniGrid2d< T >& grid)
{	
	T s = (T)0;
	T a;
	T alpha = -(T)1.0/(T)((grid.mpi_nx-2*grid.gcx)*(grid.mpi_ny-2*grid.gcy));
	T px, py;
	
	int i, j, idx;

	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		px = grid.x + (i - grid.gcx) * grid.dx + grid.dxh; // center of the cell
		py = grid.y + grid.dyh;
	  	idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
		      s += w[idx];		      
		}
	}
	a = mpi_allreduce(s, MPI_SUM);
	alpha = alpha * a;
	if (grid.mpi_com.rank == 0)
	      printf("%s average w = %E\n", message, alpha);
		
}

template< typename T >
void nse::noise(
	T* w, const T deviation, const uniGrid2d< T >& grid)
{
	T a = (T)2.0 * sqrt((T)3.0) * (T)(grid.mpi_nx - 2 * grid.gcx) * deviation;
	T r;
	int i, j, idx;

        for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
        {
                idx = i * grid.ny + grid.gcy;
                for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
                      	r = (T)rand() / (T)RAND_MAX;
			w[idx] += (r - (T)0.5) * a;
                }
        }
}

// -------------------------------------------------------------------------//
// * large-scale friction * //
// ------------------------------------------------------------------------ //


template< typename T >
void nse::u_friction( // - Rayleigh linear large-scale friction //
	T* Uinterm, const T* U, const T mu,
	const uniGrid2d< T >& grid)
{	
	int i, j, idx;
	
#pragma omp parallel for private( i, j, idx ) shared( Uinterm, U )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
	  
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
		
		  Uinterm[idx] -= mu*U[idx];
		  			
		}
	}
}


template< typename T >
void nse::v_friction( // - Rayleigh linear large-scale friction //
	T* Vinterm, const T* V, const T mu,
	const uniGrid2d< T >& grid)
{	
	int i, j, idx;
	
#pragma omp parallel for private( i, j, idx) shared( Vinterm, V )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
	  
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
		
		  Vinterm[idx] -= mu*V[idx];
		  			
		}
	}
}

template< typename T >
void nse::w_friction( // - Rayleigh linear large-scale friction //
        T* winterm, const T* w, const T mu,
        const uniGrid2d< T >& grid)
{
        int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( winterm, w )
        for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
        {

                idx = i * grid.ny + grid.gcy;
                for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

                  winterm[idx] -= mu*w[idx];

                }
        }
}

// wim -= beta dpsi/dx
template< typename T >
void nse::w_beta_effect( 
        T* winterm, const T* psi, const T beta,
        const uniGrid2d< T >& grid)
{
        int i, j, idx;

        for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
        {

                idx = i * grid.ny + grid.gcy;
                for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

                  winterm[idx] -= beta * (psi[idx + grid.ny] - psi[idx - grid.ny]) / ((T)2.0 * grid.dx);

                }
        }
}

// -------------------------------------------------------------------------//
// * Kabaret * //
// -------------------------------------------------------------------------//

template< typename T >
void nse::w_advection_kabaret(
	T* winterm, const T* wx, const T* wy, const T* U, const T* V,
	const uniGrid2d< T >& grid)
{
int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, winterm, wx, wy )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			winterm[idx] = -(
				(U[idx + grid.ny] * wx[idx + grid.ny] - U[idx] * wx[idx] ) * grid.dxi
				
				+ (V[idx + 1] * wy[idx+1] - V[idx] * wy[idx] ) * grid.dyi);
		}
	}
}

template< typename T >
void nse::w_extrapolation_kabaret( 
	T* wx_n, T* wy_n, const T* wx, const T* wy, const T* U, const T* V, const T* wim,
	const uniGrid2d< T >& grid)
{
int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, wx, wy, wx_n, wy_n, wim )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			if (U[idx] < (T)0)
			    wx_n[idx] = (T)2 * wim[idx] - (T)1 * wx[idx + grid.ny]; 
			else 
			    wx_n[idx] = (T)2 * wim[idx - grid.ny] - (T)1 * wx[idx - grid.ny];
			
			if (V[idx] < (T)0)
			    wy_n[idx] = (T)2 * wim[idx] - (T)1 * wy[idx + 1];
			else
			    wy_n[idx] = (T)2 * wim[idx - 1] - (T)1 * wy[idx - 1];			
			  
		}
	}
}

template< typename T >
void nse::w_rhs_kabaret(
	T* rhs, const T* wx, const T* wy,
	const uniGrid2d< T >& grid)
{
int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( rhs, wx, wy )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			rhs[idx] = (wy[idx] + wx[idx] + wy[idx - grid.ny] + wx[idx - 1]) / (T)4;
		}
	}
}

template< typename T>
void nse::w_kabaret(
	T* wx, T* wy, const T* w,
	const uniGrid2d< T >& grid)
{
int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( wx, w, wy )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			wx[idx] = (w[idx] + w[idx - grid.ny]) / (T)2;
			wy[idx] = (w[idx] + w[idx - 1]) / (T)2;
		}
	}	
}

template< typename T>
void nse::g2(T* wx_n, T* wy_n,
	const T* g1, const T* U,
	const T* V, const T* w, const T* wx, const T* wy, const T dt,
	const uniGrid2d< T >& grid)
{
int i, j, idx;
T qx, qy;
T maxw, minw;
T ksi;
T difference;
ksi = (T)0.47 * grid.dx + (T)0.03;

#pragma omp parallel for private( i, j, idx, qx, qy, maxw, minw ) shared( wx, wy, wx_n, wy_n, w, U, V, g1, ksi, difference )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
		    qx = (g1[idx] + g1[idx - grid.ny]) / (T)2 + U[idx] * (w[idx] - w[idx - grid.ny]) * grid.dxi;
		    if (U[idx]<(T)0) {
			maxw = max(wx[idx], wx[idx + grid.ny], w[idx]) + dt * qx;
			minw = min(wx[idx], wx[idx + grid.ny], w[idx]) + dt * qx;
		      	//maxw = max(wx[idx], wx[idx + grid.ny], w[idx]);
			//minw = min(wx[idx], wx[idx + grid.ny], w[idx]);
			//difference = maxw - minw;
			//maxw = maxw + ksi * difference;
			//minw = minw - ksi * difference;
			if (wx_n[idx] > maxw)
			  wx_n[idx] = maxw;
			if (wx_n[idx] < minw)
			  wx_n[idx] = minw;
		    }
		    else {
			maxw = max(wx[idx], wx[idx - grid.ny], w[idx - grid.ny]) + dt * qx;
			minw = min(wx[idx], wx[idx - grid.ny], w[idx - grid.ny]) + dt * qx;
//			maxw = max(wx[idx], wx[idx - grid.ny], w[idx - grid.ny]);
//			minw = min(wx[idx], wx[idx - grid.ny], w[idx - grid.ny]);
//			difference = maxw - minw;
//			maxw = maxw + ksi * difference;
//			minw = minw - ksi * difference;
			if (wx_n[idx] > maxw)
			  wx_n[idx] = maxw;
			if (wx_n[idx] < minw)
			  wx_n[idx] = minw;			
		    }
		    
		    qy = (g1[idx] + g1[idx - 1]) / (T)2 + V[idx] * (w[idx] - w[idx - 1]) * grid.dyi;
		    if (V[idx]<(T)0) {
			maxw = max(wy[idx], wy[idx + 1], w[idx]) + dt * qy;
			minw = min(wy[idx], wy[idx + 1], w[idx]) + dt * qy;
//			maxw = max(wy[idx], wy[idx + 1], w[idx]);
//			minw = min(wy[idx], wy[idx + 1], w[idx]);
//			difference = maxw - minw;
//			maxw = maxw + ksi * difference;
//			minw = minw - ksi * difference;
			if (wy_n[idx] > maxw)
			  wy_n[idx] = maxw;
			if (wy_n[idx] < minw)
			  wy_n[idx] = minw;
		    }
		    else {
		      	maxw = max(wy[idx], wy[idx - 1], w[idx - 1]) + dt * qy;
			minw = min(wy[idx], wy[idx - 1], w[idx - 1]) + dt * qy;
//			maxw = max(wy[idx], wy[idx - 1], w[idx - 1]);
//			minw = min(wy[idx], wy[idx - 1], w[idx - 1]);
//			difference = maxw - minw;
//			maxw = maxw + ksi * difference;
//			minw = minw - ksi * difference;
			if (wy_n[idx] > maxw)
			  wy_n[idx] = maxw;
			if (wy_n[idx] < minw)
			  wy_n[idx] = minw;			
		    }
		}
	}		
}

// -------------------------------------------------------------------------//
// * forcing * //
// ------------------------------------------------------------------------ //
//forcing on staggered grid C
template< typename T >
void nse::forcing( // - narrow spectral shell forcing with random phase //
	T* wim, const T k, const T kb, const T dt, const T E_in,
	const uniGrid2d< T >& grid)
{	
	int i, j, idx;
	
	T px, py;
	T phase, kx, ky;
	T A;
	if (grid.mpi_com.rank == 0)
	{
	  T r1, r2;
	  T kx_, ky_;
	  phase = (T)rand() / (T)RAND_MAX * (T)2.0 * (T)M_PI; 
	  r1 = (T)rand() / (T)RAND_MAX;
	  r2 = (T)rand() / (T)RAND_MAX * (T)2.0 * (T)M_PI;
	  kx = (k+(r1-(T)0.5)*kb) * (T)cos(r2);
	  ky = (k+(r1-(T)0.5)*kb) * (T)sin(r2);
	  kx = sign(kx) * floor(fabs(kx));
          ky = sign(ky) * floor(fabs(ky));
	  
	  kx_ = (T)2.0 * sin(kx * grid.dx / (T)2.0) * grid.dxi;
	  ky_ = (T)2.0 * sin(ky * grid.dy / (T)2.0) * grid.dyi;
	  
          A = sqrt((T)4.0 * E_in / dt) * sqrt(kx_ * kx_ + ky_ * ky_);
	  if ((fabs(kx) < (T)0.5)&&(fabs(ky) < (T)0.5)) {
		A = (T)0.0;
		printf("Forcing fail\n");
	  }
	}
	MPI_Bcast(&phase, 1, mpi_type< T >(), 0, MPI_COMM_WORLD);
	MPI_Bcast(&kx, 1, mpi_type< T >(), 0, MPI_COMM_WORLD);
 	MPI_Bcast(&ky, 1, mpi_type< T >(), 0, MPI_COMM_WORLD);
	MPI_Bcast(&A, 1, mpi_type< T >(), 0, MPI_COMM_WORLD);
//	printf(" in forcing %.4f %.4f %.4f\n", phx, phy, k);
#pragma omp parallel for private( i, j, idx, px, py ) shared( wim )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
	  
		px = grid.x + (i - grid.gcx) * grid.dx + grid.dxh; // center of the cell
		py = grid.y + grid.dyh;
		
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++, py += grid.dy) {		
		  wim[idx] += A * (T)sin(kx * px + ky * py + phase);// 30 is equivalent to projection method
		}
	}
}
//forcing on collocated grid
template< typename T >
void nse::forcing_collocated( // - narrow spectral shell forcing with random phase //
	T* wim, const T k, const T kb, const T dt, const T E_in,
	const uniGrid2d< T >& grid)
{	
	int i, j, idx;
	
	T px, py;
	T phase, kx, ky;
	T A;
	if (grid.mpi_com.rank == 0)
	{
	  T r1, r2;
	  T kx1, ky1;
	  T kx2, ky2;
	  phase = (T)rand() / (T)RAND_MAX * (T)2.0 * (T)M_PI; 
	  r1 = (T)rand() / (T)RAND_MAX;
	  r2 = (T)rand() / (T)RAND_MAX * (T)2.0 * (T)M_PI;
	  kx = (k+(r1-(T)0.5)*kb) * (T)cos(r2);
	  ky = (k+(r1-(T)0.5)*kb) * (T)sin(r2);
	  kx = sign(kx) * floor(fabs(kx));
          ky = sign(ky) * floor(fabs(ky));
	  
	  kx2 = sin(kx * grid.dx) / grid.dx;
	  ky2 = sin(ky * grid.dy) / grid.dy;
	  kx1 = (T)2.0 * sin(kx * grid.dx / (T)2.0) / grid.dx;
	  ky1 = (T)2.0 * sin(ky * grid.dy / (T)2.0) / grid.dy;
	  
          
	  A = sqrt((T)4.0 * E_in / dt) * (kx1 * kx1 + ky1 * ky1) / sqrt(kx2 * kx2 + ky2 * ky2);
	  if ((fabs(kx) < (T)0.5)&&(fabs(ky) < (T)0.5)) {
		A = (T)0.0;
		printf("Forcing fail\n");
	  }
	}
	MPI_Bcast(&phase, 1, mpi_type< T >(), 0, MPI_COMM_WORLD);
	MPI_Bcast(&kx, 1, mpi_type< T >(), 0, MPI_COMM_WORLD);
 	MPI_Bcast(&ky, 1, mpi_type< T >(), 0, MPI_COMM_WORLD);
	MPI_Bcast(&A, 1, mpi_type< T >(), 0, MPI_COMM_WORLD);
//	printf(" in forcing %.4f %.4f %.4f\n", phx, phy, k);
#pragma omp parallel for private( i, j, idx, px, py ) shared( wim )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
	  
		px = grid.x + (i - grid.gcx) * grid.dx + grid.dxh; // center of the cell
		py = grid.y + grid.dyh;
		
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++, py += grid.dy) {		
		  wim[idx] += A * (T)sin(kx * px + ky * py + phase);// 30 is equivalent to projection method
		}
	}
}

// -------------------------------------------------------------------------//
// * advection (Velocity) * //
// ------------------------------------------------------------------------ //
template< typename T >
void nse::u_advection( // - u advection skew-symmetric //
	T* Uinterm, const T* U, const T* V,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;
//	printf(" in u_adv %i %i\n", grid.nx, grid.ny);
//      grid.nx and grid.ny size of a mesh on 1 proc  
#pragma omp parallel for private( i, j, idx ) shared( U, V, Uinterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			Uinterm[idx] = -(
				(U[idx + grid.ny] * (U[idx + grid.ny] + U[idx])
				- U[idx - grid.ny] * (U[idx - grid.ny] + U[idx])) * grid.dxiq

				+ (U[idx + 1] * (V[idx + 1] + V[idx - grid.ny + 1])
				- U[idx - 1] * (V[idx] + V[idx - grid.ny])) * grid.dyiq);
		}
	}
}

template< typename T >
void nse::v_advection( // - v advection skew-symmetric //
	T* Vinterm, const T* U, const T* V,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, Vinterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			Vinterm[idx] = -(
				(V[idx + grid.ny] * (U[idx + grid.ny] + U[idx + grid.ny - 1])
				- V[idx - grid.ny] * (U[idx] + U[idx - 1])) * grid.dxiq

				+ (V[idx + 1] * (V[idx + 1] + V[idx])
				- V[idx - 1] * (V[idx - 1] + V[idx])) * grid.dyiq);
		}
	}
}

// add mean flow along x
template < typename T>
void nse::w_mean_flow(
	T* winterm, const T* w, const T Umean,
	const uniGrid2d< T >&grid)
{
	int i, j, idx;
	T coef = Umean / ((T)2.0 * grid.dx);

	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			winterm[idx] -= Umean * (w[idx + grid.ny] - w[idx - grid.ny]);
		}
	}      
}

//w advection by the Arakawa classification of the jacobians J1, J2, J3
template < typename T>
void nse::w_J1(
	T* winterm, const T* w, const T* Psi,
	const uniGrid2d< T >&grid)
{
int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( Psi, winterm, w )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			winterm[idx] += -(
				- (Psi[idx + 1] - Psi[idx - 1]) * (w[idx + grid.ny] - w[idx - grid.ny]) * grid.dxih * grid.dyih
				+ (Psi[idx + grid.ny] - Psi[idx - grid.ny]) * (w[idx + 1] - w[idx - 1]) * grid.dxih * grid.dyih
					 );
		}
	}      
}

template < typename T>
void nse::w_J2(
	T* winterm, const T* w, const T* Psi,
	const uniGrid2d< T >&grid)
{
int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( Psi, winterm, w )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			winterm[idx] += -(
				- (w[idx + grid.ny] * (Psi[idx + grid.ny + 1] - Psi[idx + grid.ny - 1]) * grid.dyih
				  - (w[idx - grid.ny] * (Psi[idx - grid.ny + 1] - Psi[idx - grid.ny - 1]) * grid.dyih) 
				  ) * grid.dxih
				+ (w[idx + 1] * (Psi[idx + 1 + grid.ny] - Psi[idx + 1 - grid.ny]) * grid.dxih
				  - (w[idx - 1] * (Psi[idx - 1 + grid.ny] - Psi[idx - 1 - grid.ny]) * grid.dxih)
				  ) * grid.dyih
					 );
		}
	}      
}

template < typename T>
void nse::w_J3(
	T* winterm, const T* w, const T* Psi,
	const uniGrid2d< T >&grid)
{
int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( Psi, winterm, w )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			winterm[idx] += -(
				+ (Psi[idx + grid.ny] * (w[idx + grid.ny + 1] - w[idx + grid.ny - 1]) * grid.dyih
				  - (Psi[idx - grid.ny] * (w[idx - grid.ny + 1] - w[idx - grid.ny - 1]) * grid.dyih) 
				  ) * grid.dxih
				- (Psi[idx + 1] * (w[idx + 1 + grid.ny] - w[idx + 1 - grid.ny]) * grid.dxih
				  - (Psi[idx - 1] * (w[idx - 1 + grid.ny] - w[idx - 1 - grid.ny]) * grid.dxih)
				  ) * grid.dyih
					 );
		}
	}      
}

template < typename T>
void nse::J_EZ(
    T* winterm, const T* w, const T* Psi,
	const uniGrid2d< T >&grid)
{
    T wim[grid.size];
    null(wim, grid.size);
    
    T alpha=1.0/3;
    
    w_J1(wim, w, Psi, grid);
    w_J2(wim, w, Psi, grid);
    w_J3(wim, w, Psi, grid);
    
    update(winterm, alpha, wim, grid.size);
}

// mean shear of 1 for two-layer fluid
template < typename T>
void nse::add_vertical_shear(
	T* qim1, T* qim2, const T* q1, const T* q2, const T* V1, const T* V2, const T kd,
	const uniGrid2d< T >&grid)
{
int i, j, idx;

	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
			qim1[idx] -= (q1[idx + grid.ny] - q1[idx - grid.ny]) * grid.dxih + nse::sqr(kd) * (V1[idx] + V1[idx - grid.ny]) * (T)0.5;
			qim2[idx] += (q2[idx + grid.ny] - q2[idx - grid.ny]) * grid.dxih + nse::sqr(kd) * (V2[idx] + V2[idx - grid.ny]) * (T)0.5;
		}
	}
}

// w advection in some usual forms
template< typename T >
void nse::w_advection( // - w advection in skew form, enstrophy conserving scheme //
	T* winterm, const T* w, const T* U, const T* V,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, winterm, w )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			winterm[idx] = -(
				(U[idx] * (w[idx + grid.ny] - w[idx - grid.ny])
				+ U[idx + grid.ny] * w[idx + grid.ny] - U[idx - grid.ny] * w[idx - grid.ny] ) * grid.dxiq
				
				+ (V[idx] * (w[idx + 1] - w[idx - 1])
				+ V[idx + 1] * w[idx + 1] - V[idx - 1] * w[idx - 1] ) * grid.dyiq);
		}
	}
}

template< typename T >
void nse::w_advection_div( // - w advection in divergent form, vortex conserving scheme //
	T* winterm, const T* w, const T* U, const T* V,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, winterm, w )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			winterm[idx] = -(
				(U[idx + grid.ny] * w[idx + grid.ny] - U[idx - grid.ny] * w[idx - grid.ny] ) * grid.dxih
				
				+ (V[idx + 1] * w[idx + 1] - V[idx - 1] * w[idx - 1] ) * grid.dyih);
		}
	}
}

template< typename T >
void nse::w_advection_div_x4( // - w advection in divergent form, vortex conserving scheme //
	T* winterm, const T* w, const T* U, const T* V,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;
	T C1, C2;
	C1 = (T)2.0 / (T)3.0;
	C2 = (T)1.0 / (T)12.0;

#pragma omp parallel for private( i, j, idx ) shared( U, V, winterm, w )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			winterm[idx] = -(
				(
				  C1 * (U[idx + grid.ny] * w[idx + grid.ny] - U[idx - grid.ny] * w[idx - grid.ny]) -
				  C2 * (U[idx + 2 * grid.ny] * w[idx + 2 * grid.ny] - U[idx - 2 * grid.ny] * w[idx - 2 * grid.ny]) 
				) * grid.dxi
				
				+ (
				  C1 * (V[idx + 1] * w[idx + 1] - V[idx - 1] * w[idx - 1]) - 
				  C2 * (V[idx + 2] * w[idx + 2] - V[idx - 2] * w[idx - 2])
				  ) * grid.dyi
				);
		}
	}
}

template< typename T >
void nse::w_advection_div_stag( // - w advection in divergent form, vortex conserving scheme //
	T* winterm, const T* w, const T* U, const T* V,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, winterm, w )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			winterm[idx] = -(
				(
				  (U[idx] + U[idx + grid.ny] + U[idx - 1] + U[idx -1 + grid.ny]) * (w[idx + grid.ny] + w[idx])
				  - (U[idx - grid.ny] + U[idx] + U[idx - 1 - grid.ny] + U[idx -1]) * (w[idx] + w[idx - grid.ny])
				) * grid.dxiqh
				
				+ (
				  (V[idx] + V[idx + 1] + V[idx - grid.ny] + V[idx - grid.ny + 1] ) * (w[idx] + w[idx + 1])
				  - (V[idx] + V[idx - grid.ny] + V[idx - 1] + V[idx - 1 - grid.ny]) * (w[idx] + w[idx - 1])
				) * grid.dyiqh
				);
		}
	}
}

template< typename T >
void nse::w_advection_div_stag_1( // - w advection in divergent form, vortex conserving scheme //
	T* winterm, const T* w, const T* U, const T* V,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, winterm, w )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			winterm[idx] = -(
				(
				 (w[idx + grid.ny] + w[idx + grid.ny + 1]) * U[idx + grid.ny] 
				  - (w[idx - grid.ny] + w[idx - grid.ny + 1]) * U[idx - grid.ny]
				  - (w[idx - grid.ny] + w[idx - grid.ny -1]) * U[idx - 1 - grid.ny]
				 + (w[idx + grid.ny] + w[idx + grid.ny - 1]) * U[idx + grid.ny - 1] 				  
				) * grid.dxiqh
				
				+ 
				(
				 (w[idx + 1 + grid.ny] + w[idx + 1]) * V[idx + 1]
				 + (w[idx + 1] + w[idx + 1 - grid.ny]) * V[idx + 1 -grid.ny]
				 - (w[idx - 1] + w[idx - 1 - grid.ny]) * V[idx - grid.ny -1]
				 - (w[idx - 1] + w[idx - 1 + grid.ny]) * V[idx - 1]
				) * grid.dyiqh
				);
		}
	}
}

template< typename T >
void nse::w_advection_en_ens( // enstrophy and energy conserving scheme //
	T* winterm, const T* w, const T* U, const T* V, const T* Psi,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, winterm, Psi, w )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
		/*
			winterm[idx] = - (T)1.0/(T)3.0 * (
				((T)0.5 * (U[idx] + U[idx - 1]) * (w[idx + grid.ny] - w[idx - grid.ny])
				+ (T)0.5 * (U[idx + grid.ny] + U[idx + grid.ny - 1]) * w[idx + grid.ny] - (T)0.5 * (U[idx - grid.ny] + U[idx - grid.ny - 1]) * w[idx - grid.ny]
				+ grid.dyih * (Psi[idx + grid.ny] * (w[idx + 1 + grid.ny] - w[idx - 1 + grid.ny]) 
				- Psi[idx - grid.ny] * (w[idx + 1 - grid.ny] - w[idx - 1 - grid.ny]))) * grid.dxih
				
				+ ((T)0.5 * (V[idx] + V[idx - grid.ny]) * (w[idx + 1] - w[idx - 1])
				+ (T)0.5 * (V[idx + 1] + V[idx + 1 - grid.ny]) * w[idx + 1] - (T)0.5 * (V[idx - 1] + V[idx - 1 - grid.ny]) * w[idx - 1] 
				- grid.dxih * (Psi[idx + 1] * (w[idx + 1 + grid.ny] - w[idx + 1 - grid.ny])
				- Psi[idx - 1] * (w[idx - 1 + grid.ny] - w[idx - 1 - grid.ny]))) * grid.dyih);
		*/
		
			winterm[idx] = - (T)1.0/(T)3.0 * (
			       (- grid.dyih * (Psi[idx + 1] - Psi[idx - 1]) * (w[idx + grid.ny] - w[idx - grid.ny])
				- grid.dyih * ((Psi[idx + 1 + grid.ny] - Psi[idx - 1 + grid.ny]) * w[idx + grid.ny] 
				-  (Psi[idx + 1 - grid.ny] - Psi[idx - 1 - grid.ny]) * w[idx - grid.ny])
				+ grid.dyih * (Psi[idx + grid.ny] * (w[idx + 1 + grid.ny] - w[idx - 1 + grid.ny]) 
				-  Psi[idx - grid.ny] * (w[idx + 1 - grid.ny] - w[idx - 1 - grid.ny]))) * grid.dxih +
				
			       (+ grid.dxih * (Psi[idx + grid.ny] - Psi[idx - grid.ny]) * (w[idx + 1] - w[idx - 1])
				+ grid.dxih * ((Psi[idx + 1 + grid.ny] - Psi[idx + 1 - grid.ny]) * w[idx + 1] 
				- (Psi[idx - 1 + grid.ny] - Psi[idx - 1 - grid.ny]) * w[idx - 1]) 
				- grid.dxih * (Psi[idx + 1] * (w[idx + 1 + grid.ny] - w[idx + 1 - grid.ny])
				- Psi[idx - 1] * (w[idx - 1 + grid.ny] - w[idx - 1 - grid.ny]))) * grid.dyih);
							
		}
	}
}

template< typename T >
void nse::u_advection_div_x4(
	T* Uinterm, const T* U, const T* V,
	const uniGrid2d< T >& grid)
{
	const T C1 = (T) 9.0 / (T) 8.0,
		C2 = (T) 1.0 / (T) 24.0,
		_3C2 = (T) 1.0 / (T) 8.0;
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, Uinterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++)
		{
			Uinterm[idx] = -(

				// d(UU)/dx
				(
				C1 * (
				C1 *
				(U[idx + grid.ny] - U[idx - grid.ny]) *
				(U[idx + grid.ny] + (T) 2.0 * U[idx] + U[idx - grid.ny]) -
				_3C2 * (
				(U[idx + 2 * grid.ny] + U[idx - grid.ny]) * (U[idx + grid.ny] + U[idx]) -
				(U[idx - 2 * grid.ny] + U[idx + grid.ny]) * (U[idx - grid.ny] + U[idx]))
				) -

				C2 * (
				C1 * (
				(U[idx + 2 * grid.ny] + U[idx + grid.ny]) * (U[idx + 3 * grid.ny] + U[idx]) -
				(U[idx - 2 * grid.ny] + U[idx - grid.ny]) * (U[idx - 3 * grid.ny] + U[idx])) -
				_3C2 *
				(U[idx + 3 * grid.ny] - U[idx - 3 * grid.ny]) *
				(U[idx + 3 * grid.ny] + (T) 2.0 * U[idx] + U[idx - 3 * grid.ny])
				)

				) * grid.dxiq +
				// ------------

				// d(UV)/dy
				(
				C1 * (
				C1 * (
				(V[idx - grid.ny + 1] + V[idx + 1]) * (U[idx + 1] + U[idx]) -
				(V[idx - grid.ny] + V[idx]) * (U[idx - 1] + U[idx])) -
				_3C2 * (
				(V[idx + grid.ny + 1] + V[idx - 2 * grid.ny + 1]) * (U[idx + 1] + U[idx]) -
				(V[idx + grid.ny] + V[idx - 2 * grid.ny]) * (U[idx - 1] + U[idx]))
				) -

				C2 * (
				C1 * (
				(V[idx - grid.ny + 2] + V[idx + 2]) * (U[idx + 3] + U[idx]) -
				(V[idx - grid.ny - 1] + V[idx - 1]) * (U[idx - 3] + U[idx])) -
				_3C2 * (
				(V[idx - 2 * grid.ny + 2] + V[idx + grid.ny + 2]) * (U[idx + 3] + U[idx]) -
				(V[idx - 2 * grid.ny - 1] + V[idx + grid.ny - 1]) * (U[idx - 3] + U[idx]))
				)

				) * grid.dyiq);
				// ------------
		}
	}
}
template< typename T >
void nse::v_advection_div_x4(
	T* Vinterm, const T* U, const T* V,
	const uniGrid2d< T >& grid)
{
	const T C1 = (T) 9.0 / (T) 8.0,
		C2 = (T) 1.0 / (T) 24.0,
		_3C2 = (T) 1.0 / (T) 8.0;
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, Vinterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++)
		{

			Vinterm[idx] = -(

				// d(UV)/dx
				(
				C1 * (
				C1 * (
				(U[idx - 1 + grid.ny] + U[idx + grid.ny]) * (V[idx + grid.ny] + V[idx]) -
				(U[idx - 1] + U[idx]) * (V[idx - grid.ny] + V[idx])) -
				_3C2 * (
				(U[idx + 1 + grid.ny] + U[idx - 2 + grid.ny]) * (V[idx + grid.ny] + V[idx]) -
				(U[idx + 1] + U[idx - 2]) * (V[idx - grid.ny] + V[idx]))
				) -

				C2 * (
				C1 * (
				(U[idx - 1 + 2 * grid.ny] + U[idx + 2 * grid.ny]) * (V[idx + 3 * grid.ny] + V[idx]) -
				(U[idx - 1 - grid.ny] + U[idx - grid.ny]) * (V[idx - 3 * grid.ny] + V[idx])) -
				_3C2 * (
				(U[idx - 2 + 2 * grid.ny] + U[idx + 1 + 2 * grid.ny]) * (V[idx + 3 * grid.ny] + V[idx]) -
				(U[idx - 2 - grid.ny] + U[idx + 1 - grid.ny]) * (V[idx - 3 * grid.ny] + V[idx]))
				)

				) * grid.dxiq +
				// ------------

				// d(VV)/dy
				(
				C1 * (
				C1 *
				(V[idx + 1] - V[idx - 1]) *
				(V[idx + 1] + (T) 2.0 * V[idx] + V[idx - 1]) -
				_3C2 * (
				(V[idx + 2] + V[idx - 1]) * (V[idx + 1] + V[idx]) -
				(V[idx - 2] + V[idx + 1]) * (V[idx - 1] + V[idx]))
				) -

				C2 * (
				C1 * (
				(V[idx + 2] + V[idx + 1]) * (V[idx + 3] + V[idx]) -
				(V[idx - 2] + V[idx - 1]) * (V[idx - 3] + V[idx])) -
				_3C2 *
				(V[idx + 3] - V[idx - 3]) *
				(V[idx + 3] + (T) 2.0 * V[idx] + V[idx - 3])
				)

				) * grid.dyiq);
				// ----------
		}
	}
}

template< typename T >
void nse::u_advection_skew_x4(
	T* Uinterm, const T* U, const T* V,
	const uniGrid2d< T >& grid)
{
	const T C1 = (T) 9.0 / (T) 8.0,
		C2 = (T) 1.0 / (T) 24.0,
		_3C2 = (T) 1.0 / (T) 8.0;
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, Uinterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++)
		{

			Uinterm[idx] = -(

				// d(UU)/dx
				(
				C1 * (
				C1 * (
				U[idx + grid.ny] * (U[idx] + U[idx + grid.ny]) -
				U[idx - grid.ny] * (U[idx] + U[idx - grid.ny])) -
				_3C2 * (
				U[idx + grid.ny] * (U[idx - grid.ny] + U[idx + 2 * grid.ny]) -
				U[idx - grid.ny] * (U[idx - 2 * grid.ny] + U[idx + grid.ny]))
				) -

				C2 * (
				C1 * (
				U[idx + 3 * grid.ny] * (U[idx + 2 * grid.ny] + U[idx + grid.ny]) -
				U[idx - 3 * grid.ny] * (U[idx - 2 * grid.ny] + U[idx - grid.ny])) -
				_3C2 * (
				U[idx + 3 * grid.ny] * (U[idx] + U[idx + 3 * grid.ny]) -
				U[idx - 3 * grid.ny] * (U[idx] + U[idx - 3 * grid.ny]))
				)

				) * grid.dxiq +
				// ------------

				// d(UV)/dy
				(
				C1 * (
				C1 * (
				U[idx + 1] * (V[idx - grid.ny + 1] + V[idx + 1]) -
				U[idx - 1] * (V[idx - grid.ny] + V[idx])) -
				_3C2 * (
				U[idx + 1] * (V[idx + grid.ny + 1] + V[idx - 2 * grid.ny + 1]) -
				U[idx - 1] * (V[idx + grid.ny] + V[idx - 2 * grid.ny]))
				) -

				C2 * (
				C1 * (
				U[idx + 3] * (V[idx - grid.ny + 2] + V[idx + 2]) -
				U[idx - 3] * (V[idx - grid.ny - 1] + V[idx - 1])) -
				_3C2 * (
				U[idx + 3] * (V[idx - 2 * grid.ny + 2] + V[idx + grid.ny + 2]) -
				U[idx - 3] * (V[idx - 2 * grid.ny - 1] + V[idx + grid.ny - 1]))
				)

				) * grid.dyiq);
				// ------------
		}
	}
}
template< typename T >
void nse::v_advection_skew_x4(
	T* Vinterm, const T* U, const T* V,
	const uniGrid2d< T >& grid)
{
	const T C1 = (T) 9.0 / (T) 8.0,
		C2 = (T) 1.0 / (T) 24.0,
		_3C2 = (T) 1.0 / (T) 8.0;
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, Vinterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++)
		{

			Vinterm[idx] = -(

				// d(UV)/dx
				(
				C1 * (
				C1 * (
				V[idx + grid.ny] * (U[idx - 1 + grid.ny] + U[idx + grid.ny]) -
				V[idx - grid.ny] * (U[idx - 1] + U[idx])) -
				_3C2 * (
				V[idx + grid.ny] * (U[idx + 1 + grid.ny] + U[idx - 2 + grid.ny]) -
				V[idx - grid.ny] * (U[idx + 1] + U[idx - 2]))
				) -

				C2 * (
				C1 * (
				V[idx + 3 * grid.ny] * (U[idx - 1 + 2 * grid.ny] + U[idx + 2 * grid.ny]) -
				V[idx - 3 * grid.ny] * (U[idx - 1 - grid.ny] + U[idx - grid.ny])) -
				_3C2 * (
				V[idx + 3 * grid.ny] * (U[idx - 2 + 2 * grid.ny] + U[idx + 1 + 2 * grid.ny]) -
				V[idx - 3 * grid.ny] * (U[idx - 2 - grid.ny] + U[idx + 1 - grid.ny]))
				)

				) * grid.dxiq +
				// ------------

				// d(VV)/dy
				(
				C1 * (
				C1 * (
				V[idx + 1] * (V[idx] + V[idx + 1]) -
				V[idx - 1] * (V[idx] + V[idx - 1])) -
				_3C2 * (
				V[idx + 1] * (V[idx - 1] + V[idx + 2]) -
				V[idx - 1] * (V[idx - 2] + V[idx + 1]))
				) -

				C2 * (
				C1 * (
				V[idx + 3] * (V[idx + 2] + V[idx + 1]) -
				V[idx - 3] * (V[idx - 2] + V[idx - 1])) -
				_3C2 * (
				V[idx + 3] * (V[idx] + V[idx + 3]) -
				V[idx - 3] * (V[idx] + V[idx - 3]))
				)

				) * grid.dyiq);
				// ----------
		}
	}
}
// ------------------------------------------------------------------------ //

// * advection (Velocity-WENO) * //
// ------------------------------------------------------------------------ //
template< typename T >
void nse::u_advection_weno(T* Uinterm,
	const T* U, const T * V,
	const uniGrid2d< T >& grid)
{
	const T min_eps = (T) 1e-6;

	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, Uinterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			T Uij = U[idx];
			T Vij = (T) 0.25 * (V[idx] + V[idx + 1] + V[idx - grid.ny] + V[idx - grid.ny + 1]);

			// - x //
			T vx_1 = (U[idx + (grid.ny << 1) + grid.ny] - U[idx + (grid.ny << 1)]) * grid.dxi;
			T vx_2 = (U[idx + (grid.ny << 1)] - U[idx + grid.ny]) * grid.dxi;
			T vx_3 = (U[idx + grid.ny] - Uij) * grid.dxi;
			T vx_4 = (Uij - U[idx - grid.ny]) * grid.dxi;
			T vx_5 = (U[idx - grid.ny] - U[idx - (grid.ny << 1)]) * grid.dxi;
			T vx_6 = (U[idx - (grid.ny << 1)] - U[idx - (grid.ny << 1) - grid.ny]) * grid.dxi;

			T Fx_1 = ((T) 1.0 / (T) 3.0) * vx_1 -
				((T) 7.0 / (T) 6.0) * vx_2 +
				((T) 11.0 / (T) 6.0) * vx_3;

			T Fx_2 = ((T) 5.0 / (T) 6.0) * vx_3 -
				((T) 1.0 / (T) 6.0) * vx_2 +
				((T) 1.0 / (T) 3.0) * vx_4;

			T Fx_3 = ((T) 1.0 / (T) 3.0) * vx_3 +
				((T) 5.0 / (T) 6.0) * vx_4 -
				((T) 1.0 / (T) 6.0) * vx_5;

			T Fx_4 = ((T) 1.0 / (T) 3.0) * vx_6 -
				((T) 7.0 / (T) 6.0) * vx_5 +
				((T) 11.0 / (T) 6.0) * vx_4;

			T vx_max = max(vx_2 * vx_2, vx_3 * vx_3,
				vx_4 * vx_4, vx_5 * vx_5);

			T vpx_eps = (T) 1e-6 * max(vx_1 * vx_1, vx_max) + min_eps;
			T vmx_eps = (T) 1e-6 * max(vx_6 * vx_6, vx_max) + min_eps;

			T Spx_1 =
				((T) 13.0 / (T) 12.0) *
				(vx_1 - vx_2 - vx_2 + vx_3) * (vx_1 - vx_2 - vx_2 + vx_3) +
				(T) 0.25 *
				(vx_1 - vx_2 - vx_2 + vx_3 + vx_3 + vx_3 - vx_2 - vx_2) *
				(vx_1 - vx_2 - vx_2 + vx_3 + vx_3 + vx_3 - vx_2 - vx_2);

			T Smx_1 =
				((T) 13.0 / (T) 12.0) *
				(vx_4 - vx_5 - vx_5 + vx_6) * (vx_4 - vx_5 - vx_5 + vx_6) +
				(T) 0.25 *
				(vx_4 - vx_5 - vx_5 + vx_6 + vx_4 + vx_4 - vx_5 - vx_5) *
				(vx_4 - vx_5 - vx_5 + vx_6 + vx_4 + vx_4 - vx_5 - vx_5);

			T cx_2 = vx_2 - vx_3 - vx_3 + vx_4;
			T cx_3 = vx_3 - vx_4 - vx_4 + vx_5;

			T Spx_2 = ((T) 13.0 / (T) 12.0) * cx_2 * cx_2 +
				(T) 0.25 * (vx_2 - vx_4) * (vx_2 - vx_4);

			T Smx_3 = ((T) 13.0 / (T) 12.0) * cx_2 * cx_2 +
				(T) 0.25 * (cx_2 + vx_4 + vx_4 - vx_3 - vx_3) * (cx_2 + vx_4 + vx_4 - vx_3 - vx_3);

			T Smx_2 = ((T) 13.0 / (T) 12.0) * cx_3 * cx_3 +
				(T) 0.25 * (vx_5 - vx_3) * (vx_5 - vx_3);

			T Spx_3 = ((T) 13.0 / (T) 12.0) * cx_3 * cx_3 +
				(T) 0.25 * (cx_3 + vx_3 + vx_3 - vx_4 - vx_4) * (cx_3 + vx_3 + vx_3 - vx_4 - vx_4);

			// - x omega //
			Spx_1 = (T) 0.1 / ((Spx_1 + vpx_eps) * (Spx_1 + vpx_eps));
			Spx_2 = (T) 0.6 / ((Spx_2 + vpx_eps) * (Spx_2 + vpx_eps));
			Spx_3 = (T) 0.3 / ((Spx_3 + vpx_eps) * (Spx_3 + vpx_eps));

			Smx_1 = (T) 0.1 / ((Smx_1 + vmx_eps) * (Smx_1 + vmx_eps));
			Smx_2 = (T) 0.6 / ((Smx_2 + vmx_eps) * (Smx_2 + vmx_eps));
			Smx_3 = (T) 0.3 / ((Smx_3 + vmx_eps) * (Smx_3 + vmx_eps));

			// - x omega inverse //
			T Ipx = (Uij < (T)0) ? (Uij / (Spx_1 + Spx_2 + Spx_3)) : (T)0;
			T Imx = (Uij >(T) 0) ? (Uij / (Smx_1 + Smx_2 + Smx_3)) : (T)0;


			// - y //
			T vy_1 = (U[idx + 3] - U[idx + 2]) * grid.dyi;
			T vy_2 = (U[idx + 2] - U[idx + 1]) * grid.dyi;
			T vy_3 = (U[idx + 1] - Uij) * grid.dyi;
			T vy_4 = (Uij - U[idx - 1]) * grid.dyi;
			T vy_5 = (U[idx - 1] - U[idx - 2]) * grid.dyi;
			T vy_6 = (U[idx - 2] - U[idx - 3]) * grid.dyi;

			T Fy_1 = ((T) 1.0 / (T) 3.0) * vy_1 -
				((T) 7.0 / (T) 6.0) * vy_2 +
				((T) 11.0 / (T) 6.0) * vy_3;
			T Fy_2 = ((T) 5.0 / (T) 6.0) * vy_3 -
				((T) 1.0 / (T) 6.0) * vy_2 +
				((T) 1.0 / (T) 3.0) * vy_4;

			T Fy_3 = ((T) 1.0 / (T) 3.0) * vy_3 +
				((T) 5.0 / (T) 6.0) * vy_4 -
				((T) 1.0 / (T) 6.0) * vy_5;

			T Fy_4 = ((T) 1.0 / (T) 3.0) * vy_6 -
				((T) 7.0 / (T) 6.0) * vy_5 +
				((T) 11.0 / (T) 6.0) * vy_4;

			T vy_max = max(vy_2 * vy_2, vy_3 * vy_3,
				vy_4 * vy_4, vy_5 * vy_5);
			T vpy_eps = (T) 1e-6 * max(vy_1 * vy_1, vy_max) + min_eps;
			T vmy_eps = (T) 1e-6 * max(vy_6 * vy_6, vy_max) + min_eps;

			T Spy_1 =
				((T) 13.0 / (T) 12.0) *
				(vy_1 - vy_2 - vy_2 + vy_3) * (vy_1 - vy_2 - vy_2 + vy_3) +
				(T) 0.25 *
				(vy_1 - vy_2 - vy_2 + vy_3 + vy_3 + vy_3 - vy_2 - vy_2) *
				(vy_1 - vy_2 - vy_2 + vy_3 + vy_3 + vy_3 - vy_2 - vy_2);

			T Smy_1 = ((T) 13.0 / (T) 12.0) *
				(vy_4 - vy_5 - vy_5 + vy_6) * (vy_4 - vy_5 - vy_5 + vy_6) +
				(T) 0.25 *
				(vy_4 - vy_5 - vy_5 + vy_6 + vy_4 + vy_4 - vy_5 - vy_5) *
				(vy_4 - vy_5 - vy_5 + vy_6 + vy_4 + vy_4 - vy_5 - vy_5);

			T cy_2 = vy_2 - vy_3 - vy_3 + vy_4;
			T cy_3 = vy_3 - vy_4 - vy_4 + vy_5;

			T Spy_2 = ((T) 13.0 / (T) 12.0) * cy_2 * cy_2 +
				(T) 0.25 * (vy_2 - vy_4) * (vy_2 - vy_4);

			T Smy_3 = ((T) 13.0 / (T) 12.0) * cy_2 * cy_2 +
				(T) 0.25 * (cy_2 + vy_4 + vy_4 - vy_3 - vy_3) * (cy_2 + vy_4 + vy_4 - vy_3 - vy_3);

			T Smy_2 = ((T) 13.0 / (T) 12.0) * cy_3 * cy_3 +
				(T) 0.25 * (vy_5 - vy_3) * (vy_5 - vy_3);

			T Spy_3 = ((T) 13.0 / (T) 12.0) * cy_3 * cy_3 +
				(T) 0.25 * (cy_3 + vy_3 + vy_3 - vy_4 - vy_4) * (cy_3 + vy_3 + vy_3 - vy_4 - vy_4);

			// - y omega //
			Spy_1 = (T) 0.1 / ((Spy_1 + vpy_eps) * (Spy_1 + vpy_eps));
			Spy_2 = (T) 0.6 / ((Spy_2 + vpy_eps) * (Spy_2 + vpy_eps));
			Spy_3 = (T) 0.3 / ((Spy_3 + vpy_eps) * (Spy_3 + vpy_eps));

			Smy_1 = (T) 0.1 / ((Smy_1 + vmy_eps) * (Smy_1 + vmy_eps));
			Smy_2 = (T) 0.6 / ((Smy_2 + vmy_eps) * (Smy_2 + vmy_eps));
			Smy_3 = (T) 0.3 / ((Smy_3 + vmy_eps) * (Smy_3 + vmy_eps));

			// - y omega inverse //
			T Ipy = (Vij < (T)0) ? (Vij / (Spy_1 + Spy_2 + Spy_3)) : (T)0;
			T Imy = (Vij >(T) 0) ? (Vij / (Smy_1 + Smy_2 + Smy_3)) : (T)0;


			Uinterm[idx] = -(
				Imx * (Smx_1 * Fx_4 + Smx_2 * Fx_3 + Smx_3 * Fx_2) +
				Ipx * (Spx_1 * Fx_1 + Spx_2 * Fx_2 + Spx_3 * Fx_3) +
				Imy * (Smy_1 * Fy_4 + Smy_2 * Fy_3 + Smy_3 * Fy_2) +
				Ipy * (Spy_1 * Fy_1 + Spy_2 * Fy_2 + Spy_3 * Fy_3));
		}
	}
}

template< typename T >
void nse::v_advection_weno(T* Vinterm,
	const T* U, const T* V,
	const uniGrid2d< T >& grid)
{
	const T min_eps = (T) 1e-6;

	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, Vinterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			T Vij = V[idx];
			T Uij = (T) 0.25 * (U[idx] + U[idx - 1] + U[idx + grid.ny] + U[idx + grid.ny - 1]);

			// - x //
			T vx_1 = (V[idx + (grid.ny << 1) + grid.ny] - V[idx + (grid.ny << 1)]) * grid.dxi;
			T vx_2 = (V[idx + (grid.ny << 1)] - V[idx + grid.ny]) * grid.dxi;
			T vx_3 = (V[idx + grid.ny] - Vij) * grid.dxi;
			T vx_4 = (Vij - V[idx - grid.ny]) * grid.dxi;
			T vx_5 = (V[idx - grid.ny] - V[idx - (grid.ny << 1)]) * grid.dxi;
			T vx_6 = (V[idx - (grid.ny << 1)] - V[idx - (grid.ny << 1) - grid.ny]) * grid.dxi;

			T Fx_1 = ((T) 1.0 / (T) 3.0) * vx_1 -
				((T) 7.0 / (T) 6.0) * vx_2 +
				((T) 11.0 / (T) 6.0) * vx_3;
			T Fx_2 = ((T) 5.0 / (T) 6.0) * vx_3 -
				((T) 1.0 / (T) 6.0) * vx_2 +
				((T) 1.0 / (T) 3.0) * vx_4;

			T Fx_3 = ((T) 1.0 / (T) 3.0) * vx_3 +
				((T) 5.0 / (T) 6.0) * vx_4 -
				((T) 1.0 / (T) 6.0) * vx_5;

			T Fx_4 = ((T) 1.0 / (T) 3.0) * vx_6 -
				((T) 7.0 / (T) 6.0) * vx_5 +
				((T) 11.0 / (T) 6.0) * vx_4;

			T vx_max = max(vx_2 * vx_2, vx_3 * vx_3,
				vx_4 * vx_4, vx_5 * vx_5);
			T vpx_eps = (T) 1e-6 * max(vx_1 * vx_1, vx_max) + min_eps;
			T vmx_eps = (T) 1e-6 * max(vx_6 * vx_6, vx_max) + min_eps;

			T Spx_1 =
				((T) 13.0 / (T) 12.0) *
				(vx_1 - vx_2 - vx_2 + vx_3) * (vx_1 - vx_2 - vx_2 + vx_3) +
				(T) 0.25 *
				(vx_1 - vx_2 - vx_2 + vx_3 + vx_3 + vx_3 - vx_2 - vx_2) *
				(vx_1 - vx_2 - vx_2 + vx_3 + vx_3 + vx_3 - vx_2 - vx_2);

			T Smx_1 =
				((T) 13.0 / (T) 12.0) *
				(vx_4 - vx_5 - vx_5 + vx_6) *
				(vx_4 - vx_5 - vx_5 + vx_6) +
				(T) 0.25 *
				(vx_4 - vx_5 - vx_5 + vx_6 + vx_4 + vx_4 - vx_5 - vx_5) *
				(vx_4 - vx_5 - vx_5 + vx_6 + vx_4 + vx_4 - vx_5 - vx_5);

			T cx_2 = vx_2 - vx_3 - vx_3 + vx_4;
			T cx_3 = vx_3 - vx_4 - vx_4 + vx_5;

			T Spx_2 = ((T) 13.0 / (T) 12.0) * cx_2 * cx_2 +
				(T) 0.25 * (vx_2 - vx_4) * (vx_2 - vx_4);

			T Smx_3 = ((T) 13.0 / (T) 12.0) * cx_2 * cx_2 +
				(T) 0.25 * (cx_2 + vx_4 + vx_4 - vx_3 - vx_3) * (cx_2 + vx_4 + vx_4 - vx_3 - vx_3);

			T Smx_2 = ((T) 13.0 / (T) 12.0) * cx_3 * cx_3 +
				(T) 0.25 * (vx_5 - vx_3) * (vx_5 - vx_3);

			T Spx_3 = ((T) 13.0 / (T) 12.0) * cx_3 * cx_3 +
				(T) 0.25 * (cx_3 + vx_3 + vx_3 - vx_4 - vx_4) * (cx_3 + vx_3 + vx_3 - vx_4 - vx_4);

			// - x omega //
			Spx_1 = (T) 0.1 / ((Spx_1 + vpx_eps) * (Spx_1 + vpx_eps));
			Spx_2 = (T) 0.6 / ((Spx_2 + vpx_eps) * (Spx_2 + vpx_eps));
			Spx_3 = (T) 0.3 / ((Spx_3 + vpx_eps) * (Spx_3 + vpx_eps));

			Smx_1 = (T) 0.1 / ((Smx_1 + vmx_eps) * (Smx_1 + vmx_eps));
			Smx_2 = (T) 0.6 / ((Smx_2 + vmx_eps) * (Smx_2 + vmx_eps));
			Smx_3 = (T) 0.3 / ((Smx_3 + vmx_eps) * (Smx_3 + vmx_eps));

			// - x omega inverse //
			T Ipx = (Uij < (T)0) ? (Uij / (Spx_1 + Spx_2 + Spx_3)) : (T)0;
			T Imx = (Uij >(T) 0) ? (Uij / (Smx_1 + Smx_2 + Smx_3)) : (T)0;


			// - y //
			T vy_1 = (V[idx + 3] - V[idx + 2]) * grid.dyi;
			T vy_2 = (V[idx + 2] - V[idx + 1]) * grid.dyi;
			T vy_3 = (V[idx + 1] - Vij) * grid.dyi;
			T vy_4 = (Vij - V[idx - 1]) * grid.dyi;
			T vy_5 = (V[idx - 1] - V[idx - 2]) * grid.dyi;
			T vy_6 = (V[idx - 2] - V[idx - 3]) * grid.dyi;

			T Fy_1 = ((T) 1.0 / (T) 3.0) * vy_1 -
				((T) 7.0 / (T) 6.0) * vy_2 +
				((T) 11.0 / (T) 6.0) * vy_3;
			T Fy_2 = ((T) 5.0 / (T) 6.0) * vy_3 -
				((T) 1.0 / (T) 6.0) * vy_2 +
				((T) 1.0 / (T) 3.0) * vy_4;

			T Fy_3 = ((T) 1.0 / (T) 3.0) * vy_3 +
				((T) 5.0 / (T) 6.0) * vy_4 -
				((T) 1.0 / (T) 6.0) * vy_5;

			T Fy_4 = ((T) 1.0 / (T) 3.0) * vy_6 -
				((T) 7.0 / (T) 6.0) * vy_5 +
				((T) 11.0 / (T) 6.0) * vy_4;

			T vy_max = max(vy_2 * vy_2, vy_3 * vy_3,
				vy_4 * vy_4, vy_5 * vy_5);
			T vpy_eps = (T) 1e-6 * max(vy_1 * vy_1, vy_max) + min_eps;
			T vmy_eps = (T) 1e-6 * max(vy_6 * vy_6, vy_max) + min_eps;

			T Spy_1 =
				((T) 13.0 / (T) 12.0) *
				(vy_1 - vy_2 - vy_2 + vy_3) * (vy_1 - vy_2 - vy_2 + vy_3) +
				(T) 0.25 *
				(vy_1 - vy_2 - vy_2 + vy_3 + vy_3 + vy_3 - vy_2 - vy_2) *
				(vy_1 - vy_2 - vy_2 + vy_3 + vy_3 + vy_3 - vy_2 - vy_2);

			T Smy_1 = ((T) 13.0 / (T) 12.0) * (vy_4 - vy_5 - vy_5 + vy_6) * (vy_4 - vy_5 - vy_5 + vy_6) +
				(T) 0.25 * (vy_4 - vy_5 - vy_5 + vy_6 + vy_4 + vy_4 - vy_5 - vy_5) *
				(vy_4 - vy_5 - vy_5 + vy_6 + vy_4 + vy_4 - vy_5 - vy_5);

			T cy_2 = vy_2 - vy_3 - vy_3 + vy_4;
			T cy_3 = vy_3 - vy_4 - vy_4 + vy_5;

			T Spy_2 = ((T) 13.0 / (T) 12.0) * cy_2 * cy_2 +
				(T) 0.25 * (vy_2 - vy_4) * (vy_2 - vy_4);

			T Smy_3 = ((T) 13.0 / (T) 12.0) * cy_2 * cy_2 +
				(T) 0.25 * (cy_2 + vy_4 + vy_4 - vy_3 - vy_3) * (cy_2 + vy_4 + vy_4 - vy_3 - vy_3);

			T Smy_2 = ((T) 13.0 / (T) 12.0) * cy_3 * cy_3 +
				(T) 0.25 * (vy_5 - vy_3) * (vy_5 - vy_3);

			T Spy_3 = ((T) 13.0 / (T) 12.0) * cy_3 * cy_3 +
				(T) 0.25 * (cy_3 + vy_3 + vy_3 - vy_4 - vy_4) * (cy_3 + vy_3 + vy_3 - vy_4 - vy_4);

			// - y omega //
			Spy_1 = (T) 0.1 / ((Spy_1 + vpy_eps) * (Spy_1 + vpy_eps));
			Spy_2 = (T) 0.6 / ((Spy_2 + vpy_eps) * (Spy_2 + vpy_eps));
			Spy_3 = (T) 0.3 / ((Spy_3 + vpy_eps) * (Spy_3 + vpy_eps));

			Smy_1 = (T) 0.1 / ((Smy_1 + vmy_eps) * (Smy_1 + vmy_eps));
			Smy_2 = (T) 0.6 / ((Smy_2 + vmy_eps) * (Smy_2 + vmy_eps));
			Smy_3 = (T) 0.3 / ((Smy_3 + vmy_eps) * (Smy_3 + vmy_eps));

			// - y omega inverse //
			T Ipy = (Vij < (T)0) ? (Vij / (Spy_1 + Spy_2 + Spy_3)) : (T)0;
			T Imy = (Vij >(T) 0) ? (Vij / (Smy_1 + Smy_2 + Smy_3)) : (T)0;


			Vinterm[idx] = -(
				Imx * (Smx_1 * Fx_4 + Smx_2 * Fx_3 + Smx_3 * Fx_2) +
				Ipx * (Spx_1 * Fx_1 + Spx_2 * Fx_2 + Spx_3 * Fx_3) +
				Imy * (Smy_1 * Fy_4 + Smy_2 * Fy_3 + Smy_3 * Fy_2) +
				Ipy * (Spy_1 * Fy_1 + Spy_2 * Fy_2 + Spy_3 * Fy_3));
		}
	}
}
// ------------------------------------------------------------------------ //

// * advection (Scalar) * //
// ------------------------------------------------------------------------ //
template< typename T >
void nse::c_advection(
	T* Xinterm, const T* U, const T* V, const T* X,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, X, Xinterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			Xinterm[idx] = -(
				(U[idx + grid.ny] * X[idx + grid.ny] - U[idx] * X[idx - grid.ny]) * grid.dxih +
				(V[idx + 1] * X[idx + 1] - V[idx] * X[idx - 1]) * grid.dyih);
		}
	}
}

template< typename T >
void nse::c_advection_div_x4(
	T* Xinterm, const T* U, const T* V, const T* X,
	const uniGrid2d< T >& grid)
{
	const T C1 = (T) 9.0 / (T) 8.0,
		C2 = (T) 1.0 / (T) 24.0;
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, X, Xinterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++)
		{
			Xinterm[idx] = -(

				// d(CU)/dx
				(
				C1 * (
				U[idx + grid.ny] * (X[idx] + X[idx + grid.ny]) -
				U[idx] * (X[idx] + X[idx - grid.ny])
				) -

				C2 * (
				U[idx + 2 * grid.ny] * (X[idx] + X[idx + 3 * grid.ny]) -
				U[idx - grid.ny] * (X[idx - 3 * grid.ny] + X[idx]))

				) * grid.dxih +
				// ------------

				// d(CV)/dy
				(
				C1 * (
				V[idx + 1] * (X[idx] + X[idx + 1]) -
				V[idx] * (X[idx] + X[idx - 1])
				) -

				C2 * (
				V[idx + 2] * (X[idx] + X[idx + 3]) -
				V[idx - 1] * (X[idx - 3] + X[idx]))

				) * grid.dyih);
			// ------------
		}
	}
}

template< typename T >
void nse::c_advection_skew_x4(
	T* Xinterm, const T* U, const T* V, const T* X,
	const uniGrid2d< T >& grid)
{
	const T C1 = (T) 9.0 / (T) 8.0,
		C2 = (T) 1.0 / (T) 24.0;
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, X, Xinterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++)
		{

			Xinterm[idx] = -(

				// d(CU)/dx
				(
				C1 * (
				U[idx + grid.ny] * X[idx + grid.ny] -
				U[idx] * X[idx - grid.ny]
				) -

				C2 * (
				U[idx + 2 * grid.ny] * X[idx + 3 * grid.ny] -
				U[idx - grid.ny] * X[idx - 3 * grid.ny])

				) * grid.dxih +
				// ------------

				// d(CV)/dy
				(
				C1 * (
				V[idx + 1] * X[idx + 1] -
				V[idx] * X[idx - 1]
				) -

				C2 * (
				V[idx + 2] * X[idx + 3] -
				V[idx - 1] * X[idx - 3])

				) * grid.dyih);
				// ------------
		}
	}
}

template< typename T >
void nse::c_advection_upwind(
	T* Xinterm, const T* U, const T* V, const T* X,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;

	T Uij, Vij;
	T Dimj, Dipj, Dijm, Dijp;

#pragma omp parallel for private( i, j, idx, Uij, Vij, Dimj, Dipj, Dijm, Dijp ) \
	shared(U, V, X, Xinterm)
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			Uij = (T) 0.5 * (U[idx] + U[idx + grid.ny]);
			Vij = (T) 0.5 * (V[idx] + V[idx + 1]);

			Dimj = (X[idx] - X[idx - grid.ny]);
			Dipj = (X[idx + grid.ny] - X[idx]);

			Dijm = (X[idx] - X[idx - 1]);
			Dijp = (X[idx + 1] - X[idx]);

			Xinterm[idx] = -(
				(max(Uij, (T) 0.0) * Dimj + min(Uij, (T) 0.0) * Dipj) * grid.dxi +
				(max(Vij, (T) 0.0) * Dijm + min(Vij, (T) 0.0) * Dijp) * grid.dyi);
		}
	}
}

template< typename T >
void nse::c_advection_upwind_x2(
	T* Xinterm, const T* U, const T* V, const T* X,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;

	T Uij, Vij;
	T Dimj, Dipj, Dijm, Dijp;

#pragma omp parallel for private( i, j, idx, Uij, Vij, Dimj, Dipj, Dijm, Dijp ) \
	shared(U, V, X, Xinterm)
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			Uij = (T) 0.5 * (U[idx] + U[idx - 1]);
			Vij = (T) 0.5 * (V[idx] + V[idx - grid.ny]);

			Dimj = ((T) 3.0 * X[idx] - (T) 4.0 * X[idx - grid.ny] + X[idx - (grid.ny << 1)]);
			Dipj = (-X[idx + (grid.ny << 1)] + (T) 4.0 * X[idx + grid.ny] - (T) 3.0 * X[idx]);

			Dijm = ((T) 3.0 * X[idx] - (T) 4.0 * X[idx - 1] + X[idx - 2]);
			Dijp = (-X[idx + 2] + (T) 4.0 * X[idx + 1] - (T) 3.0 * X[idx]);

			Xinterm[idx] = -(
				(max(Uij, (T) 0.0) * Dimj + min(Uij, (T) 0.0) * Dipj) * grid.dxih +
				(max(Vij, (T) 0.0) * Dijm + min(Vij, (T) 0.0) * Dijp) * grid.dyih);
		}
	}
}

template< typename T >
void nse::c_advection_upwind_x2_div(
	T* winterm, const T* U, const T* V, const T* w,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;

	T Ul, Ur, Vb, Vu;
	T fl, fr, fb, fu;

	// 3/2 - 1/2 extrapolation
	
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			Ur = (T)0.25 * (U[idx] + U[idx - 1] + U[idx + grid.ny] + U[idx + grid.ny - 1]);
			Ul = (T)0.25 * (U[idx] + U[idx - 1] + U[idx - grid.ny] + U[idx - grid.ny - 1]);
			
			Vb = (T)0.25 * (V[idx] + V[idx - 1] + V[idx - grid.ny - 1] + V[idx - grid.ny]);
			Vu = (T)0.25 * (V[idx] + V[idx + 1] + V[idx - grid.ny + 1] + V[idx - grid.ny]);
			
			if (Ur > (T)0.0)
				fr = (T)0.75 * (U[idx] + U[idx - 1]) * w[idx] 
				- (T)0.25 * (U[idx - grid.ny] + U[idx - grid.ny - 1]) * w[idx - grid.ny];
			else
				fr = (T)0.75 * (U[idx + grid.ny] + U[idx + grid.ny - 1]) * w[idx + grid.ny] 
				- (T)0.25 * (U[idx + grid.ny + grid.ny] + U[idx + grid.ny + grid.ny - 1]) * w[idx + grid.ny + grid.ny];
			
			if (Ul > (T)0.0)
				fl = (T)0.75 * (U[idx - grid.ny] + U[idx - grid.ny - 1]) * w[idx - grid.ny] 
				- (T)0.25 * (U[idx - grid.ny - grid.ny] + U[idx - grid.ny - grid.ny - 1]) * w[idx - grid.ny - grid.ny];
			else
				fl = (T)0.75 * (U[idx] + U[idx - 1]) * w[idx]
				- (T)0.25 * (U[idx + grid.ny] + U[idx + grid.ny - 1]) * w[idx + grid.ny];
			
			if (Vu > (T)0.0)
				fu = (T)0.75 * (V[idx] + V[idx - grid.ny]) * w[idx]
				- (T)0.25 * (V[idx - 1] + V[idx - 1 - grid.ny]) * w[idx - 1];
			else
				fu = (T)0.75 * (V[idx + 1] + V[idx + 1 - grid.ny]) * w[idx + 1] 
				- (T)0.25 * (V[idx + 2] + V[idx + 2 - grid.ny]) * w[idx + 2];
			
			if (Vb > (T)0.0)
				fb = (T)0.75 * (V[idx - 1] + V[idx - 1 - grid.ny]) * w[idx - 1] 
				- (T)0.25 * (V[idx - 2] + V[idx - 2 - grid.ny]) * w[idx - 2];
			else
				fb = (T)0.75 * (V[idx] + V[idx - grid.ny]) * w[idx] 
				- (T)0.25 * (V[idx + 1] + V[idx + 1 - grid.ny]) * w[idx + 1];
			
			winterm[idx] -= (fr - fl) * grid.dxi + (fu - fb) * grid.dyi;
		}
	}
}

template< typename T >
void nse::c_advection_upwind_x2_conserv(
	T* winterm, const T* U, const T* V, const T* w,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;

	T Ul, Ur, Vb, Vu;
	T fl, fr, fb, fu;
	T cu1 = (T)3.0 / (T)2.0, // upwind constant 
	  cu2 = - (T)1.0 / (T)2.0; // upwind next point constant
	
	// 1/3 5/6 - 1/6  interpolation

	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			Ur = (T)0.25 * (U[idx] + U[idx - 1] + U[idx + grid.ny] + U[idx + grid.ny - 1]);
			Ul = (T)0.25 * (U[idx] + U[idx - 1] + U[idx - grid.ny] + U[idx - grid.ny - 1]);
			
			Vb = (T)0.25 * (V[idx] + V[idx - 1] + V[idx - grid.ny - 1] + V[idx - grid.ny]);
			Vu = (T)0.25 * (V[idx] + V[idx + 1] + V[idx - grid.ny + 1] + V[idx - grid.ny]);
			
			if (Ur > (T)0.0)
				fr = cu1 * w[idx] 
				   + cu2 * w[idx - grid.ny];
			else
				fr = cu1 * w[idx + grid.ny] 
				   + cu2 * w[idx + grid.ny + grid.ny];
			
			if (Ul > (T)0.0)
				fl = cu1 * w[idx - grid.ny] 
				   + cu2 * w[idx - grid.ny - grid.ny];
			else
				fl = cu1 * w[idx]
				   + cu2 * w[idx + grid.ny];
			
			if (Vu > (T)0.0)
				fu = cu1 * w[idx]
				   + cu2 * w[idx - 1];
			else
				fu = cu1 * w[idx + 1] 
				   + cu2 * w[idx + 2];
			
			if (Vb > (T)0.0)
				fb = cu1 * w[idx - 1] 
				   + cu2 * w[idx - 2];
			else
				fb = cu1 * w[idx] 
				   + cu2 * w[idx + 1];
			
			fr = fr * Ur;
			fl = fl * Ul;
			
			fb = fb * Vb;
			fu = fu * Vu;
			
			winterm[idx] -= (fr - fl) * grid.dxi + (fu - fb) * grid.dyi;
		}
	}
}

template< typename T >
void nse::c_advection_upwind_x3(
	T* winterm, const T* U, const T* V, const T* w,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;

	T Uij, Vij;
	T Xp, Xm, Yp, Ym;

	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			Uij = (T) 0.5 * (U[idx] + U[idx - 1]);
			Vij = (T) 0.5 * (V[idx] + V[idx - grid.ny]);

			Xp = (- w[idx + grid.ny + grid.ny] + (T)6.0 * w[idx + grid.ny] 
			      - (T)3.0 * w[idx] - (T)2.0 * w[idx - grid.ny]) / (T)6.0 / grid.dx;
			Xm = ((T)2.0 * w[idx + grid.ny] + (T)3.0 * w[idx] 
			      - (T)6.0 * w[idx - grid.ny] + w[idx - grid.ny - grid.ny]) / (T)6.0 / grid.dx;
			      
			Yp = (- w[idx + 2] + (T)6.0 * w[idx + 1] 
			      - (T)3.0 * w[idx] - (T)2.0 * w[idx - 1]) / (T)6.0 / grid.dy;
			Ym = ((T)2.0 * w[idx + 1] + (T)3.0 * w[idx] 
			      - (T)6.0 * w[idx - 1] + w[idx - 2]) / (T)6.0 / grid.dy;
			
			winterm[idx] = -(
				(max(Uij, (T) 0.0) * Xm + min(Uij, (T) 0.0) * Xp) +
				(max(Vij, (T) 0.0) * Ym + min(Vij, (T) 0.0) * Yp));
		}
	}
}

template< typename T >
void nse::c_advection_upwind_x3_div(
	T* winterm, const T* U, const T* V, const T* w,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;

	T Ul, Ur, Vb, Vu;
	T fl, fr, fb, fu;
	T cd = (T)1.0 / (T)6.0, // downwind constant
	  cu1 = (T)5.0 / (T)12.0, // upwind constant 
	  cu2 = - (T)1.0 / (T)12.0; // upwind next point constant
	
	// 1/3 5/6 - 1/6  interpolation

	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			Ur = (T)0.25 * (U[idx] + U[idx - 1] + U[idx + grid.ny] + U[idx + grid.ny - 1]);
			Ul = (T)0.25 * (U[idx] + U[idx - 1] + U[idx - grid.ny] + U[idx - grid.ny - 1]);
			
			Vb = (T)0.25 * (V[idx] + V[idx - 1] + V[idx - grid.ny - 1] + V[idx - grid.ny]);
			Vu = (T)0.25 * (V[idx] + V[idx + 1] + V[idx - grid.ny + 1] + V[idx - grid.ny]);
			
			if (Ur > (T)0.0)
				fr = cd * (U[idx + grid.ny] + U[idx + grid.ny - 1]) * w[idx + grid.ny]  
				+ cu1 * (U[idx] + U[idx - 1]) * w[idx] 
				+ cu2 * (U[idx - grid.ny] + U[idx - grid.ny - 1]) * w[idx - grid.ny];
			else
				fr = cd * (U[idx] + U[idx - 1]) * w[idx] 
				+ cu1 * (U[idx + grid.ny] + U[idx + grid.ny - 1]) * w[idx + grid.ny] 
				+ cu2 * (U[idx + grid.ny + grid.ny] + U[idx + grid.ny + grid.ny - 1]) * w[idx + grid.ny + grid.ny];
			
			if (Ul > (T)0.0)
				fl = cd * (U[idx] + U[idx - 1]) * w[idx]
				+ cu1 * (U[idx - grid.ny] + U[idx - grid.ny - 1]) * w[idx - grid.ny] 
				+ cu2 * (U[idx - grid.ny - grid.ny] + U[idx - grid.ny - grid.ny - 1]) * w[idx - grid.ny - grid.ny];
			else
				fl = cd * (U[idx - grid.ny] + U[idx - grid.ny - 1]) * w[idx - grid.ny] 
				+ cu1 * (U[idx] + U[idx - 1]) * w[idx]
				+ cu2 * (U[idx + grid.ny] + U[idx + grid.ny - 1]) * w[idx + grid.ny];
			
			if (Vu > (T)0.0)
				fu = cd * (V[idx + 1] + V[idx + 1 - grid.ny]) * w[idx + 1]
				+ cu1 * (V[idx] + V[idx - grid.ny]) * w[idx]
				+ cu2 * (V[idx - 1] + V[idx - 1 - grid.ny]) * w[idx - 1];
			else
				fu = cd * (V[idx] + V[idx - grid.ny]) * w[idx] 
				+ cu1 * (V[idx + 1] + V[idx + 1 - grid.ny]) * w[idx + 1] 
				+ cu2 * (V[idx + 2] + V[idx + 2 - grid.ny]) * w[idx + 2];
			
			if (Vb > (T)0.0)
				fb = cd * (V[idx] + V[idx - grid.ny]) * w[idx]  
				+ cu1 * (V[idx - 1] + V[idx - 1 - grid.ny]) * w[idx - 1] 
				+ cu2 * (V[idx - 2] + V[idx - 2 - grid.ny]) * w[idx - 2];
			else
				fb = cd * (V[idx - 1] + V[idx - 1 - grid.ny]) * w[idx - 1] 
				+ cu1 * (V[idx] + V[idx - grid.ny]) * w[idx] 
				+ cu2 * (V[idx + 1] + V[idx + 1 - grid.ny]) * w[idx + 1];
			
			winterm[idx] -= (fr - fl) * grid.dxi + (fu - fb) * grid.dyi;
		}
	}
}

template< typename T >
void nse::c_advection_upwind_x3_conserv(
	T* winterm, const T* U, const T* V, const T* w,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;

	T Ul, Ur, Vb, Vu;
	T fl, fr, fb, fu;
	T cd = (T)1.0 / (T)3.0, // downwind constant
	  cu1 = (T)5.0 / (T)6.0, // upwind constant 
	  cu2 = - (T)1.0 / (T)6.0; // upwind next point constant
	
	// 1/3 5/6 - 1/6  interpolation

	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			Ur = (T)0.25 * (U[idx] + U[idx - 1] + U[idx + grid.ny] + U[idx + grid.ny - 1]);
			Ul = (T)0.25 * (U[idx] + U[idx - 1] + U[idx - grid.ny] + U[idx - grid.ny - 1]);
			
			Vb = (T)0.25 * (V[idx] + V[idx - 1] + V[idx - grid.ny - 1] + V[idx - grid.ny]);
			Vu = (T)0.25 * (V[idx] + V[idx + 1] + V[idx - grid.ny + 1] + V[idx - grid.ny]);
			
			if (Ur > (T)0.0)
				fr = cd * w[idx + grid.ny]  
				+ cu1 * w[idx] 
				+ cu2 * w[idx - grid.ny];
			else
				fr = cd * w[idx] 
				+ cu1 * w[idx + grid.ny] 
				+ cu2 * w[idx + grid.ny + grid.ny];
			
			if (Ul > (T)0.0)
				fl = cd * w[idx]
				+ cu1 * w[idx - grid.ny] 
				+ cu2 * w[idx - grid.ny - grid.ny];
			else
				fl = cd * w[idx - grid.ny] 
				+ cu1 * w[idx]
				+ cu2 * w[idx + grid.ny];
			
			if (Vu > (T)0.0)
				fu = cd * w[idx + 1]
				+ cu1 * w[idx]
				+ cu2 * w[idx - 1];
			else
				fu = cd * w[idx] 
				+ cu1 * w[idx + 1] 
				+ cu2 * w[idx + 2];
			
			if (Vb > (T)0.0)
				fb = cd * w[idx]  
				+ cu1 * w[idx - 1] 
				+ cu2 * w[idx - 2];
			else
				fb = cd * w[idx - 1] 
				+ cu1 * w[idx] 
				+ cu2 * w[idx + 1];
			
			fr = fr * Ur;
			fl = fl * Ul;
			
			fb = fb * Vb;
			fu = fu * Vu;
			
			winterm[idx] -= (fr - fl) * grid.dxi + (fu - fb) * grid.dyi;
		}
	}
}

template< typename T >
void nse::c_advection_tvd(
	T* Xinterm, const T* U, const T* V, const T* X,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;

	T Fipj, Fimj, Fijp, Fijm;
	T Xij, Xipj, Xijp, Ximj, Xijm;
	T vx1, vx2, vx3, vx4;
	T vy1, vy2, vy3, vy4;

#pragma omp parallel for private( i, j, idx ) \
	private(Fipj, Fimj, Fijp, Fijm, Xij, Xipj, Xijp, Ximj, Xijm) \
	private(vx1, vx2, vx3, vx4, vy1, vy2, vy3, vy4) shared(U, V, X, Xinterm)
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;

		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			Xij = X[idx];
			Xipj = X[idx + grid.ny];
			Ximj = X[idx - grid.ny];
			Xijp = X[idx + 1];
			Xijm = X[idx - 1];

			vx1 = (X[idx + (grid.ny << 1)] - Xipj) * grid.dxi;
			vx2 = (Xipj - Xij) * grid.dxi;
			vx3 = (Xij - Ximj) * grid.dxi;
			vx4 = (Ximj - X[idx - (grid.ny << 1)]) * grid.dxi;

			vy1 = (X[idx + 2] - Xijp) * grid.dyi;
			vy2 = (Xijp - Xij) * grid.dyi;
			vy3 = (Xij - Xijm) * grid.dyi;
			vy4 = (Xijm - X[idx - 2]) * grid.dyi;


			// Pre-calculation of limiter of (vx2, vx3) and (vy2, vy3) should be
			//      based on stratification parameters 

			Fipj = (U[idx + grid.ny] >= (T)0) ?
				U[idx + grid.ny] * (Xij + grid.dxh * superbee_limit(vx2, vx3)) :
				U[idx + grid.ny] * (Xipj - grid.dxh * superbee_limit(vx1, vx2));
			Fimj = (U[idx] >= (T)0) ?
				U[idx] * (Ximj + grid.dxh * superbee_limit(vx3, vx4)) :
				U[idx] * (Xij - grid.dxh * superbee_limit(vx2, vx3));

			Fijp = (V[idx + 1] >= (T)0) ?
				V[idx + 1] * (Xij + grid.dyh * superbee_limit(vy2, vy3)) :
				V[idx + 1] * (Xijp - grid.dyh * superbee_limit(vy1, vy2));
			Fijm = (V[idx] >= (T)0) ?
				V[idx] * (Xijm + grid.dyh * superbee_limit(vy3, vy4)) :
				V[idx] * (Xij - grid.dyh * superbee_limit(vy2, vy3));

			Xinterm[idx] =
				-(Fipj - Fimj) * grid.dxi
				- (Fijp - Fijm) * grid.dyi;
		}
	}
}

template< typename T >
void nse::c_advection_weno(T* Xinterm,
	const T* U, const T* V, const T* X,
	const uniGrid2d< T >& grid)
{
	const T min_eps = (T) 1e-6;

	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, X, Xinterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;

		T vy_1;
		T vy_2 = (X[idx + 2] - X[idx + 1]) * grid.dyi;
		T vy_3 = (X[idx + 1] - X[idx]) * grid.dyi;
		T vy_4 = (X[idx] - X[idx - 1]) * grid.dyi;
		T vy_5 = (X[idx - 1] - X[idx - 2]) * grid.dyi;
		T vy_6 = (X[idx - 2] - X[idx - 3]) * grid.dyi;

		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			T Uij = (T) 0.5 * (U[idx] + U[idx + grid.ny]);
			T Vij = (T) 0.5 * (V[idx] + V[idx + 1]);

			// - x //
			T vx_1 = (X[idx + (grid.ny << 1) + grid.ny] - X[idx + (grid.ny << 1)]) * grid.dxi;
			T vx_2 = (X[idx + (grid.ny << 1)] - X[idx + grid.ny]) * grid.dxi;
			T vx_3 = (X[idx + grid.ny] - X[idx]) * grid.dxi;
			T vx_4 = (X[idx] - X[idx - grid.ny]) * grid.dxi;
			T vx_5 = (X[idx - grid.ny] - X[idx - (grid.ny << 1)]) * grid.dxi;
			T vx_6 = (X[idx - (grid.ny << 1)] - X[idx - (grid.ny << 1) - grid.ny]) * grid.dxi;


			T Fx_1 = ((T) 1.0 / (T) 3.0) * vx_1 -
				((T) 7.0 / (T) 6.0) * vx_2 +
				((T) 11.0 / (T) 6.0) * vx_3;
			T Fx_2 = ((T) 5.0 / (T) 6.0) * vx_3 -
				((T) 1.0 / (T) 6.0) * vx_2 +
				((T) 1.0 / (T) 3.0) * vx_4;

			T Fx_3 = ((T) 1.0 / (T) 3.0) * vx_3 +
				((T) 5.0 / (T) 6.0) * vx_4 -
				((T) 1.0 / (T) 6.0) * vx_5;

			T Fx_4 = ((T) 1.0 / (T) 3.0) * vx_6 -
				((T) 7.0 / (T) 6.0) * vx_5 +
				((T) 11.0 / (T) 6.0) * vx_4;


			T vx_max = max(vx_2 * vx_2, vx_3 * vx_3,
				vx_4 * vx_4, vx_5 * vx_5);
			T vpx_eps = (T) 1e-6 * max(vx_1 * vx_1, vx_max) + min_eps;
			T vmx_eps = (T) 1e-6 * max(vx_6 * vx_6, vx_max) + min_eps;

			T Spx_1 =
				((T) 13.0 / (T) 12.0) *
				(vx_1 - vx_2 - vx_2 + vx_3) * (vx_1 - vx_2 - vx_2 + vx_3) +
				(T) 0.25 *
				(vx_1 - vx_2 - vx_2 + vx_3 + vx_3 + vx_3 - vx_2 - vx_2) *
				(vx_1 - vx_2 - vx_2 + vx_3 + vx_3 + vx_3 - vx_2 - vx_2);

			T Smx_1 =
				((T) 13.0 / (T) 12.0) *
				(vx_4 - vx_5 - vx_5 + vx_6) *
				(vx_4 - vx_5 - vx_5 + vx_6) +
				(T) 0.25 *
				(vx_4 - vx_5 - vx_5 + vx_6 + vx_4 + vx_4 - vx_5 - vx_5) *
				(vx_4 - vx_5 - vx_5 + vx_6 + vx_4 + vx_4 - vx_5 - vx_5);

			T cx_2 = vx_2 - vx_3 - vx_3 + vx_4;
			T cx_3 = vx_3 - vx_4 - vx_4 + vx_5;

			T Spx_2 = ((T) 13.0 / (T) 12.0) * cx_2 * cx_2 +
				(T) 0.25 * (vx_2 - vx_4) * (vx_2 - vx_4);

			T Smx_3 = ((T) 13.0 / (T) 12.0) * cx_2 * cx_2 +
				(T) 0.25 * (cx_2 + vx_4 + vx_4 - vx_3 - vx_3) * (cx_2 + vx_4 + vx_4 - vx_3 - vx_3);

			T Smx_2 = ((T) 13.0 / (T) 12.0) * cx_3 * cx_3 +
				(T) 0.25 * (vx_5 - vx_3) * (vx_5 - vx_3);

			T Spx_3 = ((T) 13.0 / (T) 12.0) * cx_3 * cx_3 +
				(T) 0.25 * (cx_3 + vx_3 + vx_3 - vx_4 - vx_4) * (cx_3 + vx_3 + vx_3 - vx_4 - vx_4);

			// - x omega //
			Spx_1 = (T) 0.1 / ((Spx_1 + vpx_eps) * (Spx_1 + vpx_eps));
			Spx_2 = (T) 0.6 / ((Spx_2 + vpx_eps) * (Spx_2 + vpx_eps));
			Spx_3 = (T) 0.3 / ((Spx_3 + vpx_eps) * (Spx_3 + vpx_eps));

			Smx_1 = (T) 0.1 / ((Smx_1 + vmx_eps) * (Smx_1 + vmx_eps));
			Smx_2 = (T) 0.6 / ((Smx_2 + vmx_eps) * (Smx_2 + vmx_eps));
			Smx_3 = (T) 0.3 / ((Smx_3 + vmx_eps) * (Smx_3 + vmx_eps));

			// - x omega inverse //
			T Ipx = (Uij < (T)0) ? (Uij / (Spx_1 + Spx_2 + Spx_3)) : (T)0;
			T Imx = (Uij >(T) 0) ? (Uij / (Smx_1 + Smx_2 + Smx_3)) : (T)0;


			// - y //
			vy_1 = (X[idx + 3] - X[idx + 2]) * grid.dyi;


			T Fy_1 = ((T) 1.0 / (T) 3.0) * vy_1 -
				((T) 7.0 / (T) 6.0) * vy_2 +
				((T) 11.0 / (T) 6.0) * vy_3;
			T Fy_2 = ((T) 5.0 / (T) 6.0) * vy_3 -
				((T) 1.0 / (T) 6.0) * vy_2 +
				((T) 1.0 / (T) 3.0) * vy_4;

			T Fy_3 = ((T) 1.0 / (T) 3.0) * vy_3 +
				((T) 5.0 / (T) 6.0) * vy_4 -
				((T) 1.0 / (T) 6.0) * vy_5;

			T Fy_4 = ((T) 1.0 / (T) 3.0) * vy_6 -
				((T) 7.0 / (T) 6.0) * vy_5 +
				((T) 11.0 / (T) 6.0) * vy_4;

			T vy_max = max(vy_2 * vy_2, vy_3 * vy_3,
				vy_4 * vy_4, vy_5 * vy_5);
			T vpy_eps = (T) 1e-6 * max(vy_1 * vy_1, vy_max) + min_eps;
			T vmy_eps = (T) 1e-6 * max(vy_6 * vy_6, vy_max) + min_eps;

			T Spy_1 =
				((T) 13.0 / (T) 12.0) *
				(vy_1 - vy_2 - vy_2 + vy_3) * (vy_1 - vy_2 - vy_2 + vy_3) +
				(T) 0.25 *
				(vy_1 - vy_2 - vy_2 + vy_3 + vy_3 + vy_3 - vy_2 - vy_2) *
				(vy_1 - vy_2 - vy_2 + vy_3 + vy_3 + vy_3 - vy_2 - vy_2);

			T Smy_1 = ((T) 13.0 / (T) 12.0) * (vy_4 - vy_5 - vy_5 + vy_6) * (vy_4 - vy_5 - vy_5 + vy_6) +
				(T) 0.25 * (vy_4 - vy_5 - vy_5 + vy_6 + vy_4 + vy_4 - vy_5 - vy_5) *
				(vy_4 - vy_5 - vy_5 + vy_6 + vy_4 + vy_4 - vy_5 - vy_5);

			T cy_2 = vy_2 - vy_3 - vy_3 + vy_4;
			T cy_3 = vy_3 - vy_4 - vy_4 + vy_5;

			T Spy_2 = ((T) 13.0 / (T) 12.0) * cy_2 * cy_2 +
				(T) 0.25 * (vy_2 - vy_4) * (vy_2 - vy_4);

			T Smy_3 = ((T) 13.0 / (T) 12.0) * cy_2 * cy_2 +
				(T) 0.25 * (cy_2 + vy_4 + vy_4 - vy_3 - vy_3) * (cy_2 + vy_4 + vy_4 - vy_3 - vy_3);

			T Smy_2 = ((T) 13.0 / (T) 12.0) * cy_3 * cy_3 +
				(T) 0.25 * (vy_5 - vy_3) * (vy_5 - vy_3);

			T Spy_3 = ((T) 13.0 / (T) 12.0) * cy_3 * cy_3 +
				(T) 0.25 * (cy_3 + vy_3 + vy_3 - vy_4 - vy_4) * (cy_3 + vy_3 + vy_3 - vy_4 - vy_4);

			// - y omega //
			Spy_1 = (T) 0.1 / ((Spy_1 + vpy_eps) * (Spy_1 + vpy_eps));
			Spy_2 = (T) 0.6 / ((Spy_2 + vpy_eps) * (Spy_2 + vpy_eps));
			Spy_3 = (T) 0.3 / ((Spy_3 + vpy_eps) * (Spy_3 + vpy_eps));

			Smy_1 = (T) 0.1 / ((Smy_1 + vmy_eps) * (Smy_1 + vmy_eps));
			Smy_2 = (T) 0.6 / ((Smy_2 + vmy_eps) * (Smy_2 + vmy_eps));
			Smy_3 = (T) 0.3 / ((Smy_3 + vmy_eps) * (Smy_3 + vmy_eps));

			// - y omega inverse //
			T Ipy = (Vij < (T)0) ? (Vij / (Spy_1 + Spy_2 + Spy_3)) : (T)0;
			T Imy = (Vij >(T) 0) ? (Vij / (Smy_1 + Smy_2 + Smy_3)) : (T)0;


			Xinterm[idx] = -(
				Imx * (Smx_1 * Fx_4 + Smx_2 * Fx_3 + Smx_3 * Fx_2) +
				Ipx * (Spx_1 * Fx_1 + Spx_2 * Fx_2 + Spx_3 * Fx_3) +
				Imy * (Smy_1 * Fy_4 + Smy_2 * Fy_3 + Smy_3 * Fy_2) +
				Ipy * (Spy_1 * Fy_1 + Spy_2 * Fy_2 + Spy_3 * Fy_3));

			vy_6 = vy_5;
			vy_5 = vy_4;
			vy_4 = vy_3;
			vy_3 = vy_2;
			vy_2 = vy_1;
		}
	}

}
// ------------------------------------------------------------------------ //

// * diffusion (Velocity) * //
// ------------------------------------------------------------------------ //
template< typename T >
void nse::u_diffusion(
	T* Uinterm, const T* U, const T* V,
	const T c_viscosity, const uniGrid2d< T >& grid)
{
	const T c_visc_dx2i = c_viscosity * grid.dx2i;
	const T c_visc_dy2i = c_viscosity * grid.dy2i;

	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, Uinterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {

		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
			Uinterm[idx] +=
				(U[idx + grid.ny] - U[idx] - U[idx] + U[idx - grid.ny]) * c_visc_dx2i
				+ (U[idx + 1] - U[idx] - U[idx] + U[idx - 1]) * c_visc_dy2i;

		}
	}
}

template< typename T >
void nse::v_diffusion(
	T* Vinterm, const T* U, const T* V,
	const T c_viscosity, const uniGrid2d< T >& grid)
{
	const T c_visc_dx2i = c_viscosity * grid.dx2i;
	const T c_visc_dy2i = c_viscosity * grid.dy2i;

	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, Vinterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {

		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			Vinterm[idx] +=
				(V[idx + grid.ny] - V[idx] - V[idx] + V[idx - grid.ny]) * c_visc_dx2i
				+ (V[idx + 1] - V[idx] - V[idx] + V[idx - 1]) * c_visc_dy2i;
		}
	}
}
// ------------------------------------------------------------------------ //

// * diffusion (Vortex) * //
// ------------------------------------------------------------------------ //
template< typename T >
void nse::w_diffusion(
	T* winterm, const T* w,
	const T c_viscosity, const uniGrid2d< T >& grid)
{
	const T c_visc_dx2i = c_viscosity * grid.dx2i;
	const T c_visc_dy2i = c_viscosity * grid.dy2i;

	int i, j, idx;

	for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {

		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
			winterm[idx] +=
				(w[idx + grid.ny] - w[idx] - w[idx] + w[idx - grid.ny]) * c_visc_dx2i
				+ (w[idx + 1] - w[idx] - w[idx] + w[idx - 1]) * c_visc_dy2i;

		}
	}
}

template< typename T >
void nse::w_diffusion_2(
        T* winterm, const T* w,
        const T c_viscosity, const uniGrid2d< T >& grid)
{
        const T c_visc_dx2i = c_viscosity * grid.dx2i;
        const T c_visc_dy2i = c_viscosity * grid.dy2i;

        int i, j, idx;
	T* w2;
	allocate(&w2, grid.size);

#pragma omp parallel for private( i, j, idx ) shared( w, w2 )
        for (i = grid.gcx - 1; i < grid.nx - grid.gcx + 1; i++) {

                idx = i * grid.ny + grid.gcy - 1;
                for (j = grid.gcy - 1; j < grid.ny - grid.gcy + 1; j++, idx++) {
                        w2[idx] =
                                (w[idx + grid.ny] - w[idx] - w[idx] + w[idx - grid.ny]) * grid.dx2i
                                + (w[idx + 1] - w[idx] - w[idx] + w[idx - 1]) * grid.dy2i;

                }
        }

#pragma omp parallel for private( i, j, idx ) shared( w2, winterm )
        for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {

                idx = i * grid.ny + grid.gcy;
                for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
                        winterm[idx] +=
                                (w2[idx + grid.ny] - w2[idx] - w2[idx] + w2[idx - grid.ny]) * c_visc_dx2i
                                + (w2[idx + 1] - w2[idx] - w2[idx] + w2[idx - 1]) * c_visc_dy2i;

                }
        }
	deallocate(w2);

}


template< typename T >
void nse::w_diffusion_x4(
	T* winterm, const T* w,
	const T c_viscosity, const uniGrid2d< T >& grid)
{
	const T C1 = (T) 9.0 / (T) 8.0,
		C2 = (T) 1.0 / (T) 24.0;
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( w, winterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++)
		{

			winterm[idx] += c_viscosity * (

				(C1 * C1 * (w[idx + grid.ny] - w[idx] - w[idx] + w[idx - grid.ny]) +

				C2 * C2 * (w[idx + 3 * grid.ny] - w[idx] - w[idx] + w[idx - 3 * grid.ny]) -

				(T) 2.0 * C1 * C2 * (w[idx + 2 * grid.ny] - w[idx + grid.ny] - w[idx - grid.ny] + w[idx - 2 * grid.ny])) * grid.dx2i +



				(C1 * C1 * (w[idx + 1] - w[idx] - w[idx] + w[idx - 1]) +

				C2 * C2 * (w[idx + 3] - w[idx] - w[idx] + w[idx - 3]) -

				(T) 2.0 * C1 * C2 * (w[idx + 2] - w[idx + 1] - w[idx - 1] + w[idx - 2])) * grid.dy2i);
		}
	}
}
// ------------------------------------------------------------------------ //


// * diffusion (Scalar) * //
// ------------------------------------------------------------------------ //
template< typename T >
void nse::c_diffusion(
	T* Xinterm, const T* X,
	const T c_diffusivity, const uniGrid2d< T >& grid)
{
	const T c_diff_dx2i = c_diffusivity * grid.dx2i;
	const T c_diff_dy2i = c_diffusivity * grid.dy2i;

	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( X, Xinterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			Xinterm[idx] +=
				(X[idx + grid.ny] - X[idx] - X[idx] + X[idx - grid.ny]) * c_diff_dx2i
				+ (X[idx + 1] - X[idx] - X[idx] + X[idx - 1]) * c_diff_dy2i;
		}
	}
}
// ------------------------------------------------------------------------ //

// * eddy diffusion (Velocity) * //
// ------------------------------------------------------------------------ //
template< typename T >
void nse::u_eddy_diffusion(
	T* Uinterm, const T* U, const T* V,
	const T* visc, const uniGrid2d< T >& grid)
{
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, Uinterm, visc )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			Uinterm[idx] += (
				(
				visc[idx] * (U[idx + grid.ny] - U[idx]) -
				visc[idx - grid.ny] * (U[idx] - U[idx - grid.ny])
				) * grid.dx2id +

				((visc[idx] + visc[idx - grid.ny] + visc[idx + 1] + visc[idx - grid.ny + 1]) *
				((U[idx + 1] - U[idx]) * grid.dyiq +
				(V[idx + 1] - V[idx - grid.ny + 1]) * grid.dxiq
				)
				- (visc[idx] + visc[idx - grid.ny] + visc[idx - 1] + visc[idx - grid.ny - 1]) *
				((U[idx] - U[idx - 1]) * grid.dyiq +
				(V[idx] - V[idx - grid.ny]) * grid.dxiq
				)
				) * grid.dyi);
		}
	}
}

template< typename T >
void nse::v_eddy_diffusion(
	T* Vinterm, const T* U, const T* V,
	const T* visc, const uniGrid2d< T >& grid)
{
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, Vinterm, visc )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			Vinterm[idx] += (
				(
				visc[idx] * (V[idx + 1] - V[idx]) -
				visc[idx - 1] * (V[idx] - V[idx - 1])
				) * grid.dy2id +


				((visc[idx] + visc[idx - 1] + visc[idx + grid.ny] + visc[idx - 1 + grid.ny]) *
				((V[idx + grid.ny] - V[idx]) * grid.dxiq +
				(U[idx + grid.ny] - U[idx + grid.ny - 1]) * grid.dyiq
				)
				- (visc[idx] + visc[idx - 1] + visc[idx - grid.ny] + visc[idx - grid.ny - 1]) *
				((V[idx] - V[idx - grid.ny]) * grid.dxiq +
				(U[idx] - U[idx - 1]) * grid.dyiq
				)
				) * grid.dxi);
		}
	}
}
// ------------------------------------------------------------------------ //

// * variable density diffusion (Velocity) * //
// ------------------------------------------------------------------------ //
template< typename T >
void nse::u_diffusion(
	T* Uinterm, const T* U, const T* V,
	const T* visc, const T* i_density, const uniGrid2d< T >& grid)
{
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, Uinterm, visc, i_density )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			Uinterm[idx] += (T) 0.5 * (i_density[idx] + i_density[idx - grid.ny]) * (
				(
				visc[idx] * (U[idx + grid.ny] - U[idx]) -
				visc[idx - grid.ny] * (U[idx] - U[idx - grid.ny])
				) * grid.dx2id +

				(
				(visc[idx] + visc[idx - grid.ny] + visc[idx + 1] + visc[idx - grid.ny + 1]) *
				((U[idx + 1] - U[idx]) * grid.dyiq +
				(V[idx + 1] - V[idx - grid.ny + 1]) * grid.dxiq
				)
				- (visc[idx] + visc[idx - grid.ny] + visc[idx - 1] + visc[idx - grid.ny - 1]) *
				((U[idx] - U[idx - 1]) * grid.dyiq +
				(V[idx] - V[idx - grid.ny]) * grid.dxiq
				)
				) * grid.dyi);
		}
	}
}

template< typename T >
void nse::v_diffusion(
	T* Vinterm, const T* U, const T* V,
	const T* visc, const T* i_density, const uniGrid2d< T >& grid)
{
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, Vinterm, visc, i_density )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			Vinterm[idx] += (T) 0.5 * (i_density[idx] + i_density[idx - 1]) * (
				(
				visc[idx] * (V[idx + 1] - V[idx]) -
				visc[idx - 1] * (V[idx] - V[idx - 1])
				) * grid.dy2id +

				(
				(visc[idx] + visc[idx - 1] + visc[idx + grid.ny] + visc[idx - 1 + grid.ny]) *
				((V[idx + grid.ny] - V[idx]) * grid.dxiq +
				(U[idx + grid.ny] - U[idx + grid.ny - 1]) * grid.dyiq
				)
				- (visc[idx] + visc[idx - 1] + visc[idx - grid.ny] + visc[idx - grid.ny - 1]) *
				((V[idx] - V[idx - grid.ny]) * grid.dxiq +
				(U[idx] - U[idx - 1]) * grid.dyiq
				)
				) * grid.dxi);
		}
	}
}
// ------------------------------------------------------------------------ //

// * diffusion (Velocity-X4) * //
// ------------------------------------------------------------------------ //
template< typename T >
void nse::u_diffusion_x4(
	T* Uinterm, const T* U, const T* V,
	const T c_viscosity, const uniGrid2d< T >& grid)
{
	const T C1 = (T) 9.0 / (T) 8.0,
		C2 = (T) 1.0 / (T) 24.0;
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, Uinterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++)
		{

			Uinterm[idx] += c_viscosity * (

				(C1 * C1 * (U[idx + grid.ny] - U[idx] - U[idx] + U[idx - grid.ny]) +
				C2 * C2 * (U[idx + 3 * grid.ny] - U[idx] - U[idx] + U[idx - 3 * grid.ny]) -
				(T) 2.0 * C1 * C2 * (U[idx + 2 * grid.ny] - U[idx + grid.ny] - U[idx - grid.ny] + U[idx - 2 * grid.ny])) * grid.dx2i +

				(C1 * C1 * (U[idx + 1] - U[idx] - U[idx] + U[idx - 1]) +
				C2 * C2 * (U[idx + 3] - U[idx] - U[idx] + U[idx - 3]) -
				(T) 2.0 * C1 * C2 * (U[idx + 2] - U[idx + 1] - U[idx - 1] + U[idx - 2])) * grid.dy2i);
		}
	}
}

template< typename T >
void nse::v_diffusion_x4(
	T* Vinterm, const T* U, const T* V,
	const T c_viscosity, const uniGrid2d< T >& grid)
{
	const T C1 = (T) 9.0 / (T) 8.0,
		C2 = (T) 1.0 / (T) 24.0;
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, Vinterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++)
		{

			Vinterm[idx] += c_viscosity * (

				(C1 * C1 * (V[idx + grid.ny] - V[idx] - V[idx] + V[idx - grid.ny]) +
				C2 * C2 * (V[idx + 3 * grid.ny] - V[idx] - V[idx] + V[idx - 3 * grid.ny]) -
				(T) 2.0 * C1 * C2 * (V[idx + 2 * grid.ny] - V[idx + grid.ny] - V[idx - grid.ny] + V[idx - 2 * grid.ny])) * grid.dx2i +

				(C1 * C1 * (V[idx + 1] - V[idx] - V[idx] + V[idx - 1]) +
				C2 * C2 * (V[idx + 3] - V[idx] - V[idx] + V[idx - 3]) -
				(T) 2.0 * C1 * C2 * (V[idx + 2] - V[idx + 1] - V[idx - 1] + V[idx - 2])) * grid.dy2i);
		}
	}
}

template< typename T >
void nse::c_diffusion_x4(
	T* Xinterm, const T* X,
	const T c_diffusivity, const uniGrid2d< T >& grid)
{
	const T C1 = (T) 9.0 / (T) 8.0,
		C2 = (T) 1.0 / (T) 24.0;
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( X, Xinterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++)
		{

			Xinterm[idx] += c_diffusivity * (

				(C1 * C1 * (X[idx + grid.ny] - X[idx] - X[idx] + X[idx - grid.ny]) +
				C2 * C2 * (X[idx + 3 * grid.ny] - X[idx] - X[idx] + X[idx - 3 * grid.ny]) -
				(T) 2.0 * C1 * C2 * (X[idx + 2 * grid.ny] - X[idx + grid.ny] - X[idx - grid.ny] + X[idx - 2 * grid.ny])) * grid.dx2i +

				(C1 * C1 * (X[idx + 1] - X[idx] - X[idx] + X[idx - 1]) +
				C2 * C2 * (X[idx + 3] - X[idx] - X[idx] + X[idx - 3]) -
				(T) 2.0 * C1 * C2 * (X[idx + 2] - X[idx + 1] - X[idx - 1] + X[idx - 2])) * grid.dy2i);
		}
	}
}
// ------------------------------------------------------------------------ //
// * Velocity * //
//------------------------------------------------------------------------- //
template< typename T >
void nse::velocity(
	T* U, T* V, const T* Psi,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, Psi )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {

		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			U[idx] =
				-(Psi[idx + 1] - Psi[idx - 1]) * grid.dyih;
			V[idx] = 
				(Psi[idx + grid.ny] - Psi[idx - grid.ny]) * grid.dxih;
				
		}
	}
}

template< typename T >
void nse::velocity_x4(
	T* U, T* V, const T* Psi,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;
	T C1, C2;
	C1 = (T)2.0 / (T)3.0;
	C2 = (T)1.0 / (T)12.0;
#pragma omp parallel for private( i, j, idx ) shared( U, V, Psi )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {

		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			U[idx] = -(
				 C1 * (Psi[idx + 1] - Psi[idx - 1]) - C2 * (Psi[idx + 2] - Psi[idx - 2]) 
				  )* grid.dyi;
			V[idx] = (
				 C1 * (Psi[idx + grid.ny] - Psi[idx - grid.ny]) -
				 C2 * (Psi[idx + 2 * grid.ny] - Psi[idx - 2 * grid.ny])				  
				 ) * grid.dxi;
				
		}
	}
}

template< typename T >
void nse::velocity_stag(
	T* U, T* V, const T* Psi,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, Psi )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {

		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			U[idx] =
				-(Psi[idx + 1] - Psi[idx]) * grid.dyi;
			V[idx] = 
				(Psi[idx + grid.ny] - Psi[idx]) * grid.dxi;
				
		}
	}
}
//------------------------------------------------------------------------- //
// * Divergence - Vorticity * //
// ------------------------------------------------------------------------ //
template< typename T >
void nse::divergence(
	T* Div, const T* U, const T* V,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, Div )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {

		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			Div[idx] =
				(U[idx + grid.ny] - U[idx]) * grid.dxi
				+ (V[idx + 1] - V[idx]) * grid.dyi;
		}
	}
}

template< typename T >
void nse::vorticity(
	T* Vort, const T* U, const T* V,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;

	for (i = grid.gcx; i < grid.nx - grid.gcx; i++) {

		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			Vort[idx] =
				(V[idx] - V[idx - grid.ny]) * grid.dxi
				- (U[idx] - U[idx-1]) * grid.dyi;
		}
	}
}
// ------------------------------------------------------------------------ //

// * Divergence (Mass) * //
// ------------------------------------------------------------------------ //
template< typename T >
void nse::divergence(
	T* Div, const T* U, const T* V,
	const T* u_mass, const T* v_mass,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, Div )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			Div[idx] =
				(U[idx + grid.ny] * u_mass[idx + grid.ny] - U[idx] * u_mass[idx]) * grid.dxi
				+ (V[idx + 1] * v_mass[idx + 1] - V[idx] * v_mass[idx]) * grid.dyi;
		}
	}
}
// ------------------------------------------------------------------------ //

// * Divergence (-X4) * //
// ------------------------------------------------------------------------ //
template< typename T >
void nse::divergence_x4(
	T* Div, const T* U, const T* V,
	const uniGrid2d< T >& grid)
{
	const T C1 = (T) 9.0 / (T) 8.0,
		C2 = (T) 1.0 / (T) 24.0;
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( U, V, Div )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++)
		{
			Div[idx] =
				(C1 * (U[idx + grid.ny] - U[idx]) - C2 * (U[idx + 2 * grid.ny] - U[idx - grid.ny])) * grid.dxi +
				(C1 * (V[idx + 1] - V[idx]) - C2 * (V[idx + 2] - V[idx - 1])) * grid.dyi;
		}
	}
}
// ------------------------------------------------------------------------ //

// * Kinetic Energy * //
// ------------------------------------------------------------------------ //
template< typename T >
T nse::kinetic_energy(
	const T* U, const T* V,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;
	T ke_sum = (T)0;

#pragma omp parallel for private( i, j, idx ) shared( U, V ) reduction( + : ke_sum )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			ke_sum += (
				(U[idx] + U[idx + grid.ny]) * (U[idx] + U[idx + grid.ny]) +
				(V[idx] + V[idx + 1]) * (V[idx] + V[idx + 1]));
		}
	}

	mpi_allreduce_comm(&ke_sum, MPI_SUM, grid.mpi_com.comm);
	return (T) 0.25 * ke_sum * grid.dx * grid.dy;
}
// kinetic energy by unit square
template<typename T>
T nse::kinetic_energy_collocated(
	const T* U, const T* V,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;
	T ke_sum = (T)0;

	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			ke_sum += (
				U[idx] * U[idx] +
				V[idx] * V[idx] );
		}
	}

	mpi_allreduce_comm(&ke_sum, MPI_SUM, grid.mpi_com.comm);
	return (T) 0.5 * ke_sum * grid.dx * grid.dy / grid.mpi_length / grid.mpi_width;
}
// ------------------------------------------------------------------------ //
// * Sources of enstrophy and energy by unit square * //
// ------------------------------------------------------------------------ //

template<typename T>
void nse::sources(
	T* Ens, T* Ens_Source, T* En_Source,
	const T* w, const T* wim, const T* Psi,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;
	T ens = (T)0;
	T ens_s = (T)0;
	T en_s = (T)0;
	
#pragma omp parallel for private( i, j, idx ) shared( w, wim, Psi ) reduction( + : ens, ens_s, en_s )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			ens += w[idx] * w[idx];
			ens_s += w[idx] * wim[idx];
			en_s -= Psi[idx] * wim[idx];
		}
	}

	mpi_allreduce_comm(&ens, MPI_SUM, grid.mpi_com.comm);
	mpi_allreduce_comm(&ens_s, MPI_SUM, grid.mpi_com.comm);
	mpi_allreduce_comm(&en_s, MPI_SUM, grid.mpi_com.comm);
	(*Ens) = ens * 0.5 * grid.dx * grid.dy / grid.mpi_width / grid.mpi_length;
	(*Ens_Source) = ens_s * grid.dx * grid.dy / grid.mpi_width / grid.mpi_length;
	(*En_Source) = en_s * grid.dx * grid.dy / grid.mpi_width / grid.mpi_length;
}

template< typename T >
void nse::sources_sh(
	s_balance< T >& balance,
	const T* w, const T* wim, const T* Psi,
	const uniGrid2d< T >& grid)
{
if (balance.status == true) {
	int i, j, idx;
	T ens_s = (T)0;
	T en_s = (T)0;
	T *rhs;
	allocate(&rhs, grid.size);
	assign(rhs, (T)1.0, wim, -(T)1.0, balance.wim_p, grid.size);
		
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
			ens_s += w[idx] * rhs[idx];
			en_s -= Psi[idx] * rhs[idx];
		}
	}
	mpi_allreduce_comm(&ens_s, MPI_SUM, grid.mpi_com.comm);
	mpi_allreduce_comm(&en_s, MPI_SUM, grid.mpi_com.comm);
	
	balance.m_en_sh += en_s * grid.dx * grid.dy / grid.mpi_width / grid.mpi_length;
	balance.m_ens_sh += ens_s * grid.dx * grid.dy / grid.mpi_width / grid.mpi_length;
	memcpy(balance.wim_p, wim, grid.size * sizeof(T));
	
	deallocate(rhs);
}
}

template< typename T >
void nse::sources_visc(
	s_balance< T >& balance,
	const T* w, const T* wim, const T* Psi,
	const uniGrid2d< T >& grid)
{
if (balance.status == true) {
	int i, j, idx;
	T ens_s = (T)0;
	T en_s = (T)0;
	T *rhs;
	allocate(&rhs, grid.size);
	
	assign(rhs, (T)1.0, wim, -(T)1.0, balance.wim_p, grid.size);
	
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
			ens_s += w[idx] * rhs[idx];
			en_s -= Psi[idx] * rhs[idx];
		}
	}
	mpi_allreduce_comm(&ens_s, MPI_SUM, grid.mpi_com.comm);
	mpi_allreduce_comm(&en_s, MPI_SUM, grid.mpi_com.comm);
	
    balance.en_visc  = en_s * grid.dx * grid.dy / grid.mpi_width / grid.mpi_length;
    balance.ens_visc = ens_s * grid.dx * grid.dy / grid.mpi_width / grid.mpi_length;
	balance.m_en_visc += balance.en_visc;
	balance.m_ens_visc += balance.ens_visc;
        memcpy(balance.wim_p, wim, grid.size * sizeof(T));
	
	deallocate(rhs);
}
}

template< typename T >
void nse::sources_fric(
	s_balance< T >& balance,
	const T* w, const T* wim, const T* Psi,
	const uniGrid2d< T >& grid)
{
if (balance.status == true) {
	int i, j, idx;
	T ens_s = (T)0;
	T en_s = (T)0;
	T *rhs;
	allocate(&rhs, grid.size);
	
	assign(rhs, (T)1.0, wim, -(T)1.0, balance.wim_p, grid.size);
	
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
			ens_s += w[idx] * rhs[idx];
			en_s -= Psi[idx] * rhs[idx];
		}
	}
	mpi_allreduce_comm(&ens_s, MPI_SUM, grid.mpi_com.comm);
	mpi_allreduce_comm(&en_s, MPI_SUM, grid.mpi_com.comm);
	
	balance.m_en_fric += en_s * grid.dx * grid.dy / grid.mpi_width / grid.mpi_length;
	balance.m_ens_fric += ens_s * grid.dx * grid.dy / grid.mpi_width / grid.mpi_length;
        memcpy(balance.wim_p, wim, grid.size * sizeof(T));
	
	deallocate(rhs);
}
}

template< typename T >
void nse::sources_forcing(
	s_balance< T >& balance,
	const T* w, const T* wim, const T* Psi, const T dt,
	const uniGrid2d< T >& grid)
{
if (balance.status == true) {
	int i, j, idx;
	T ens_s = (T)0;
	T *rhs;
	allocate(&rhs, grid.size);
	
	assign(rhs, (T)1.0, wim, -(T)1.0, balance.wim_p, grid.size);
	
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
			ens_s += (T)0.5 * rhs[idx] * rhs[idx] * dt;
		}
	}
	mpi_allreduce_comm(&ens_s, MPI_SUM, grid.mpi_com.comm);
	
	balance.m_ens_forcing += ens_s * grid.dx * grid.dy / grid.mpi_width / grid.mpi_length;
        memcpy(balance.wim_p, wim, grid.size * sizeof(T));
	
	deallocate(rhs);
}
}

template< typename T >
void nse::invariant_level(
	s_balance< T >& balance,
	const T* w, const T* Psi,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;
	T ens = (T)0;
	T en = (T)0;
	
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
			ens += w[idx] * w[idx];
			en -= Psi[idx] * w[idx];
		}
	}
	mpi_allreduce_comm(&ens, MPI_SUM, grid.mpi_com.comm);
	mpi_allreduce_comm(&en, MPI_SUM, grid.mpi_com.comm);
	
	balance.en = (T)0.5 * en * grid.dx * grid.dy / grid.mpi_width / grid.mpi_length;
	balance.ens = (T)0.5 * ens * grid.dx * grid.dy / grid.mpi_width / grid.mpi_length;
	if (balance.status == true) {
		balance.m_en += balance.en;
		balance.m_ens += balance.ens;  
	}
}
// ------------------------------------------------------------------------ //
// * Gradient * //
// ------------------------------------------------------------------------ //
template< typename T >
void nse::u_sub_gradient(
	T* Uinterm, const T* X,
	const T c_gradient, const uniGrid2d< T >& grid)
{
	const T c_gradient_x = c_gradient * grid.dxi;

	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( X, Uinterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
			Uinterm[idx] -= c_gradient_x * (X[idx] - X[idx - grid.ny]);
		}
	}
}

template< typename T >
void nse::v_sub_gradient(
	T* Vinterm, const T* X,
	const T c_gradient, const uniGrid2d< T >& grid)
{
	const T c_gradient_y = c_gradient * grid.dyi;

	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( X, Vinterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
			Vinterm[idx] -= c_gradient_y * (X[idx] - X[idx - 1]);
		}
	}
}

template< typename T >
void nse::u_add_gradient(
	T* Uinterm, const T* X,
	const T c_gradient, const uniGrid2d< T >& grid)
{
	const T c_gradient_x = c_gradient * grid.dxi;

	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( X, Uinterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
			Uinterm[idx] += c_gradient_x * (X[idx] - X[idx - grid.ny]);
		}
	}
}

template< typename T >
void nse::v_add_gradient(
	T* Vinterm, const T* X,
	const T c_gradient, const uniGrid2d< T >& grid)
{
	const T c_gradient_y = c_gradient * grid.dyi;

	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( X, Vinterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
			Vinterm[idx] += c_gradient_y * (X[idx] - X[idx - 1]);
		}
	}
}
// ------------------------------------------------------------------------ //

// * Gradient (Var) * //
// ------------------------------------------------------------------------ //
template< typename T >
void nse::u_sub_gradient(
	T* Uinterm, const T* X,
	const T* c_gradient, const uniGrid2d< T >& grid)
{
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( X, Uinterm, c_gradient )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			Uinterm[idx] -= (c_gradient[idx] + c_gradient[idx - grid.ny]) *
				(X[idx] - X[idx - grid.ny]) * grid.dxih;
		}
	}
}

template< typename T >
void nse::v_sub_gradient(
	T* Vinterm, const T* X,
	const T* c_gradient, const uniGrid2d< T >& grid)
{
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( X, Vinterm, c_gradient )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			Vinterm[idx] -= (c_gradient[idx] + c_gradient[idx - 1]) *
				(X[idx] - X[idx - 1]) * grid.dyih;
		}
	}
}

template< typename T >
void nse::u_add_gradient(
	T* Uinterm, const T* X,
	const T* c_gradient, const uniGrid2d< T >& grid)
{
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( X, Uinterm, c_gradient )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			Uinterm[idx] += (c_gradient[idx] + c_gradient[idx - grid.ny]) *
				(X[idx] - X[idx - grid.ny]) * grid.dxih;
		}
	}
}

template< typename T >
void nse::v_add_gradient(
	T* Vinterm, const T* X,
	const T* c_gradient, const uniGrid2d< T >& grid)
{
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( X, Vinterm, c_gradient )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			Vinterm[idx] += (c_gradient[idx] + c_gradient[idx - 1]) *
				(X[idx] - X[idx - 1]) * grid.dyih;
		}
	}
}
// ------------------------------------------------------------------------ //

// * Gradient (-X4) * //
// ------------------------------------------------------------------------ //
template< typename T >
void nse::u_sub_gradient_x4(
	T* Uinterm, const T* X,
	const T c_gradient, const uniGrid2d< T >& grid)
{
	const T C1 = (T) 9.0 / (T) 8.0,
			C2 = (T) 1.0 / (T) 24.0;

	const T c_gradient_x = c_gradient * grid.dxi;

	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( X, Uinterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			Uinterm[idx] -= c_gradient_x * (
				C1 * (X[idx] - X[idx - grid.ny]) - C2 * (X[idx + grid.ny] - X[idx - 2 * grid.ny]));
		}
	}
}

template< typename T >
void nse::v_sub_gradient_x4(
	T* Vinterm, const T* X,
	const T c_gradient, const uniGrid2d< T >& grid)
{
	const T C1 = (T) 9.0 / (T) 8.0,
			C2 = (T) 1.0 / (T) 24.0;

	const T c_gradient_y = c_gradient * grid.dyi;

	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( X, Vinterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			Vinterm[idx] -= c_gradient_y * (
				C1 * (X[idx] - X[idx - 1]) - C2 * (X[idx + 1] - X[idx - 2]));
		}
	}
}

template< typename T >
void nse::u_add_gradient_x4(
	T* Uinterm, const T* X,
	const T c_gradient, const uniGrid2d< T >& grid)
{
	const T C1 = (T) 9.0 / (T) 8.0,
			C2 = (T) 1.0 / (T) 24.0;

	const T c_gradient_x = c_gradient * grid.dxi;

	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( X, Uinterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			Uinterm[idx] += c_gradient_x * (
				C1 * (X[idx] - X[idx - grid.ny]) - C2 * (X[idx + grid.ny] - X[idx - 2 * grid.ny]));
		}
	}
}

template< typename T >
void nse::v_add_gradient_x4(
	T* Vinterm, const T* X,
	const T c_gradient, const uniGrid2d< T >& grid)
{
	const T C1 = (T) 9.0 / (T) 8.0,
			C2 = (T) 1.0 / (T) 24.0;

	const T c_gradient_y = c_gradient * grid.dyi;

	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( X, Vinterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			Vinterm[idx] += c_gradient_y * (
				C1 * (X[idx] - X[idx - 1]) - C2 * (X[idx + 1] - X[idx - 2]));
		}
	}
}
// ------------------------------------------------------------------------ //

// * Poisson Eq. RHS * //
// ------------------------------------------------------------------------ //
template< typename T >
void nse::poisson_rhs(
	T* Rhs,
	const T* Div,
	const T* Uinterm, const T* Vinterm,
	const uniGrid2d< T >& grid, const T dt)
{
	const T idt = (T) 1.0 / dt;

	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) \
	shared(Div, Uinterm, Vinterm, Rhs)
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			Rhs[idx] = Div[idx] * idt +

				(Uinterm[idx + grid.ny] - Uinterm[idx]) * grid.dxi
				+ (Vinterm[idx + 1] - Vinterm[idx]) * grid.dyi;
		}
	}
}

template< typename T >
void nse::poisson_rhs(
	T* Rhs,
	const T* U, const T* V,
	const T* Uinterm, const T* Vinterm,
	const uniGrid2d< T >& grid, const T dt)
{
	const T idt = (T) 1.0 / dt;

	int i, j, idx;
	T divergence;

#pragma omp parallel for private( i, j, idx, divergence ) \
	shared(U, V, Uinterm, Vinterm, Rhs)
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			divergence =
				(U[idx + grid.ny] - U[idx]) * grid.dxi
				+ (V[idx + 1] - V[idx]) * grid.dyi;

			Rhs[idx] = divergence * idt +

				(Uinterm[idx + grid.ny] - Uinterm[idx]) * grid.dxi
				+ (Vinterm[idx + 1] - Vinterm[idx]) * grid.dyi;
		}
	}
}
// ------------------------------------------------------------------------ //

// * Poisson Eq. RHS (Mass) * //
// ------------------------------------------------------------------------ //
template< typename T >
void nse::poisson_rhs(
	T* Rhs,
	const T* U, const T* V,
	const T* Uinterm, const T* Vinterm,
	const T* u_mass, const T* v_mass,
	const uniGrid2d< T >& grid, const T dt)
{
	const T idt = (T) 1.0 / dt;

	int i, j, idx;
	T divergence;

#pragma omp parallel for private( i, j, idx, divergence ) \
	shared(U, V, Uinterm, Vinterm, Rhs)
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			divergence =
				(U[idx + grid.ny] * u_mass[idx + grid.ny] - U[idx] * u_mass[idx]) * grid.dxi
				+ (V[idx + 1] * v_mass[idx + 1] - V[idx] * v_mass[idx]) * grid.dyi;

			Rhs[idx] = divergence * idt +

				(Uinterm[idx + grid.ny] * u_mass[idx + grid.ny] - Uinterm[idx] * u_mass[idx]) * grid.dxi
				+ (Vinterm[idx + 1] * v_mass[idx + 1] - Vinterm[idx] * v_mass[idx]) * grid.dyi;
		}
	}
}
// ------------------------------------------------------------------------ //

// * Poisson Eq. RHS (-X4) * //
// ------------------------------------------------------------------------ //
template< typename T >
void nse::poisson_rhs_x4(
	T* Rhs,
	const T* Div,
	const T* Uinterm, const T* Vinterm,
	const uniGrid2d< T >& grid, const T dt)
{
	const T C1 = (T) 9.0 / (T) 8.0,
			C2 = (T) 1.0 / (T) 24.0;

	const T idt = (T) 1.0 / dt;

	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) \
	shared(Div, Uinterm, Vinterm, Rhs)
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			Rhs[idx] = Div[idx] * idt +

				(C1 * (Uinterm[idx + grid.ny] - Uinterm[idx]) - C2 * (Uinterm[idx + 2 * grid.ny] - Uinterm[idx - grid.ny])) * grid.dxi +
				(C1 * (Vinterm[idx + 1] - Vinterm[idx]) - C2 * (Vinterm[idx + 2] - Vinterm[idx - 1])) * grid.dyi;
		}
	}
}
// ------------------------------------------------------------------------ //
// * Adams-Bashforth Time Advancement * //
// ------------------------------------------------------------------------ //
template< typename T >
void nse::adams_bashforth_x2(
	T* Xn, const T* X, T* Xp,
	const uniGrid2d< T >& grid, const T dt)
{
	int i;
	T C;

#pragma omp parallel for private( i ) shared( Xp, X, Xn )
	for (i = 0; i < grid.size; i++) {
		C = dt * ((T) 1.5 * X[i] - (T) 0.5 * Xp[i]);

		Xp[i] = X[i];
		Xn[i] += C;
	
	}
}

template< typename T >
void nse::adams_bashforth_x2(
	T* X, T* Xp,
	const uniGrid2d< T >& grid)
{
	int i;
	T C;

#pragma omp parallel for private( i, C ) shared( Xp, X )
	for (i = 0; i < grid.size; i++) {
		C = (T) 1.5 * X[i] - (T) 0.5 * Xp[i];

		Xp[i] = X[i];
		X[i] = C;
	}
}

template< typename T >
void nse::adams_bashforth_x2(
	T* X, T* Xp,
	const uniGrid2d< T >& grid, const T dt, const T p_dt)
{
	const T c_dt = (T) 0.5 * dt / p_dt;

	int i;
	T C;

#pragma omp parallel for private( i, C ) shared( Xp, X )
	for (i = 0; i < grid.size; i++) {
		C = X[i] + c_dt * (X[i] - Xp[i]);

		Xp[i] = X[i];
		X[i] = C;
	}
}

template< typename T >
void nse::adams_bashforth_x2(
	T* X, T* Xp, const T eps,
	const uniGrid2d< T >& grid)
{
	int i;
	T C;

#pragma omp parallel for private( i, C ) shared( Xp, X )
	for (i = 0; i < grid.size; i++) {
		C = ((T) 1.5 + eps) * X[i] - ((T) 0.5 + eps) * Xp[i];
		Xp[i] = X[i]; X[i] = C;
	}
}

template< typename T >
void nse::adams_bashforth_x3(
	T* X, T* Xp, T* Xpp,
	const uniGrid2d< T >& grid)
{
	int i;
	T C;

#pragma omp parallel for private( i, C ) shared( Xpp, Xp, X )
	for (i = 0; i < grid.size; i++) {
		C = ((T) 23.0 / (T) 12.0) * X[i] -
			((T) 4.0 / (T) 3.0) * Xp[i] +
			((T) 5.0 / (T) 12.0) * Xpp[i];

		Xpp[i] = Xp[i];
		Xp[i] = X[i];
		X[i] = C;
	}
}

template< typename T >
void nse::adams_bashforth_x3(
	T* X, T* Xp, T* Xpp,
	const uniGrid2d< T >& grid,
	const T dt, const T p_dt, const T pp_dt)
{
	const T alpha = (T) 1.0 +
		((T) 1.0 / (T) 6.0) *
		(dt * ((T) 2.0 * dt + (T) 3.0 * pp_dt + (T) 6.0 * p_dt)) / (p_dt * (p_dt + pp_dt));
	const T beta = ((T) 1.0 / (T) 6.0) *
		(dt * ((T) 2.0 * dt + (T) 3.0 * pp_dt + (T) 3.0 * p_dt)) / (pp_dt * p_dt);
	const T gamma = ((T) 1.0 / (T) 6.0) *
		(dt * ((T) 2.0 * dt + (T) 3.0 * p_dt)) / (pp_dt * (p_dt + pp_dt));

	int i;
	T C;

#pragma omp parallel for private( i, C ) shared( Xpp, Xp, X )
	for (i = 0; i < grid.size; i++) {
		C = alpha * X[i] - beta * Xp[i] + gamma * Xpp[i];

		Xpp[i] = Xp[i];
		Xp[i] = X[i];
		X[i] = C;
	}
}
// ------------------------------------------------------------------------ //


// * Velocity Projection * //
// ------------------------------------------------------------------------ //
template< typename T >
void nse::u_projection(
	T* U, const T* Uinterm, const T* Phi,
	const uniGrid2d< T >& grid, const T dt)
{
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( Uinterm, Phi, U )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			U[idx] += dt * (Uinterm[idx] -
				(Phi[idx] - Phi[idx - grid.ny]) * grid.dxi);
		}
	}
}

template< typename T >
void nse::v_projection(
	T* V, const T* Vinterm, const T* Phi,
	const uniGrid2d< T >& grid, const T dt)
{
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( Vinterm, Phi, V )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			V[idx] += dt * (Vinterm[idx] -
				(Phi[idx] - Phi[idx - 1]) * grid.dyi);
		}
	}
}
// ------------------------------------------------------------------------ //

// * Velocity Projection (Variable Density) * //
// ------------------------------------------------------------------------ //
template< typename T >
void nse::u_projection(
	T* U, const T* Uinterm, const T* Phi, const T* i_density,
	const uniGrid2d< T >& grid, const T dt)
{
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( Uinterm, Phi, U, i_density )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			U[idx] += dt * (Uinterm[idx] -
				(i_density[idx] + i_density[idx - grid.ny]) *
				(Phi[idx] - Phi[idx - grid.ny]) * grid.dxih);
		}
	}
}

template< typename T >
void nse::v_projection(
	T* V, const T* Vinterm, const T* Phi, const T* i_density,
	const uniGrid2d< T >& grid, const T dt)
{
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( Vinterm, Phi, V, i_density )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			V[idx] += dt * (Vinterm[idx] -
				(i_density[idx] + i_density[idx - 1]) *
				(Phi[idx] - Phi[idx - 1]) * grid.dyih);
		}
	}
}
// ------------------------------------------------------------------------ //

// * Velocity Projection (-X4) * //
// ------------------------------------------------------------------------ //
template< typename T >
void nse::u_projection_x4(
	T* U, const T* Uinterm, const T* Phi,
	const uniGrid2d< T >& grid, const T dt)
{
	const T C1 = (T) 9.0 / (T) 8.0,
			C2 = (T) 1.0 / (T) 24.0;

	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( Uinterm, Phi, U )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			U[idx] += dt * (Uinterm[idx] -
				(C1 * (Phi[idx] - Phi[idx - grid.ny]) -
				C2 * (Phi[idx + grid.ny] - Phi[idx - 2 * grid.ny])) * grid.dxi);
		}
	}
}

template< typename T >
void nse::v_projection_x4(
	T* V, const T* Vinterm, const T* Phi,
	const uniGrid2d< T >& grid, const T dt)
{
	const T C1 = (T) 9.0 / (T) 8.0,
			C2 = (T) 1.0 / (T) 24.0;

	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( Vinterm, Phi, V )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {

			V[idx] += dt * (Vinterm[idx] -
				(C1 * (Phi[idx] - Phi[idx - 1]) -
				C2 * (Phi[idx + 1] - Phi[idx - 2])) * grid.dyi);
		}
	}
}
// ------------------------------------------------------------------------ //

// * Buoyancy * //
// ------------------------------------------------------------------------ //
template< typename T >
void nse::u_buoyancy(
	T* Uinterm, const T* X,
	const T c_expansion, const T c_gravity_x, const uniGrid2d< T >& grid)
{
	const T c_buoyancy = (T) 0.5 * c_gravity_x * c_expansion;

	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( X, Uinterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
			Uinterm[idx] -= c_buoyancy * (X[idx] + X[idx - grid.ny]);
		}
	}
}

template< typename T >
void nse::v_buoyancy(
	T* Vinterm, const T* X,
	const T c_expansion, const T c_gravity_y, const uniGrid2d< T >& grid)
{
	const T c_buoyancy = (T) 0.5 * c_gravity_y * c_expansion;

	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( X, Vinterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
			Vinterm[idx] -= c_buoyancy * (X[idx] + X[idx - 1]);
		}
	}
}
// ------------------------------------------------------------------------ //

// * Buoyancy * //
// ------------------------------------------------------------------------ //
template< typename T >
void nse::u_buoyancy_x4(
	T* Uinterm, const T* X,
	const T c_expansion, const T c_gravity_x, const uniGrid2d< T >& grid)
{
	const T C1 = (T) 9.0 / (T) 8.0,
		C2 = (T) 1.0 / (T) 8.0;
	const T c_expansion_x = (T) 0.5 * c_gravity_x * c_expansion;
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( X, Uinterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++)
		{
			Uinterm[idx] -= c_expansion_x * (
				C1 * (X[idx] + X[idx - grid.ny]) -
				C2 * (X[idx + grid.ny] + X[idx - 2 * grid.ny]));
		}
	}
}

template< typename T >
void nse::v_buoyancy_x4(
	T* Vinterm, const T* X,
	const T c_expansion, const T c_gravity_y, const uniGrid2d< T >& grid)
{
	const T C1 = (T) 9.0 / (T) 8.0,
		C2 = (T) 1.0 / (T) 8.0;
	const T c_expansion_y = (T) 0.5 * c_gravity_y * c_expansion;
	int i, j, idx;

#pragma omp parallel for private( i, j, idx ) shared( X, Vinterm )
	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++)
		{
			Vinterm[idx] -= c_expansion_y * (
				C1 * (X[idx] + X[idx - 1]) -
				C2 * (X[idx + 1] + X[idx - 2]));
		}
	}
}
// ------------------------------------------------------------------------ //

// * velocity abs max * //
template< typename T >
void nse::velocity_abs_max(T* umax, T* vmax,
	const T* U, const T* V,
	const uniGrid2d< T >& grid)
{
	int i, j, idx;
	T um = (T)0, vm = (T)0;

	for (i = grid.gcx; i < grid.nx - grid.gcx; i++)
	{
		idx = i * grid.ny + grid.gcy;
		for (j = grid.gcy; j < grid.ny - grid.gcy; j++, idx++) {
			if (fabs(U[idx]) > um) um = fabs(U[idx]);
			if (fabs(V[idx]) > vm) vm = fabs(V[idx]);
		}
	}

	mpi_allreduce(&um, &vm, MPI_MAX);

	(*umax) = um;
	(*vmax) = vm;
}

// * Energy spectrum * //
// spectrum of net energy
template< typename T >
void nse::energy_spectrum(T* E, T* k, T* Uin, T* Vin,
	const uniGrid2d< T >& grid)
{
	int i, j, idx, idxin;
	int nx = grid.mpi_nx - 2 * grid.gcx;
	int ny = grid.mpi_ny - 2 * grid.gcy;
	T *U, *V;
	
	if (grid.mpi_com.rank == 0) {
		U = new T[grid.mpi_size];
		V = new T[grid.mpi_size];
	}

	grid.mpi_com.gather(U, Uin, 0, grid.nx, grid.ny, grid.gcx, grid.gcy);
	grid.mpi_com.gather(V, Vin, 0, grid.nx, grid.ny, grid.gcx, grid.gcy);
	
	if (grid.mpi_com.rank == 0) {
		fftw_complex *Us, *Vs; // spectrum
		fftw_complex *in, *out;
		
		in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny);
		out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny);
		Us = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny);
		Vs = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny);
		
		fftw_plan plan=fftw_plan_dft_2d(nx, ny, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

		for (i = grid.gcx; i < grid.mpi_nx - grid.gcx; i++)
		{
			idxin = i * grid.mpi_ny + grid.gcy;
			idx = (i - grid.gcx) * (grid.mpi_ny - 2 * grid.gcy);
			for (j = grid.gcy; j < grid.mpi_ny - grid.gcy; j++, idx++, idxin++) {
				in[idx][0] = U[idxin];
				in[idx][1] = (T)0.0;
			}
		}
	
		fftw_execute(plan);
		for (idx = 0; idx < nx * ny; idx++)
		{
			Us[idx][0] = out[idx][0];
			Us[idx][1] = out[idx][1];
		}		
		
		for (i = grid.gcx; i < grid.mpi_nx - grid.gcx; i++)
		{
			idxin = i * grid.mpi_ny + grid.gcy;
			idx = (i - grid.gcx) * (grid.mpi_ny - 2 * grid.gcy);
			for (j = grid.gcy; j < grid.mpi_ny - grid.gcy; j++, idx++, idxin++) {
				in[idx][0] = V[idxin];
				in[idx][1] = (T)0.0;
			}
		}
		fftw_execute(plan);
		for (idx = 0; idx < nx * ny; idx++)
		{
			Vs[idx][0] = out[idx][0];
			Vs[idx][1] = out[idx][1];
		}	
		
		T kx, ky, k_, kmax;
		int l, lmax;
		kmax = (T)(nx / 2);
		lmax = (int)(kmax + 0.1);

		for (i = 0; i < lmax; i++)
			k[i] = (T)0.5 + (T)i;
		for (i = 0; i < lmax; i++)
			E[i] = (T)0.0;
		for (i = 0; i < nx; i++)
		{
			idx = i * ny;
			for (j = 0; j < ny; j++, idx++)
			{
				kx = (T)i;
				ky = (T)j;
				if (kx > kmax)
					kx = kx - 2 * kmax;
				if (ky > kmax)
					ky = ky - 2 * kmax;
				k_ = sqrt(kx * kx + ky * ky);
				l = (int)floor(k_);
				if (l < lmax)
					E[l] = E[l] + (T)(Us[idx][0] * Us[idx][0] + Us[idx][1] * Us[idx][1]) 
					+ (Vs[idx][0] * Vs[idx][0] + Vs[idx][1] * Vs[idx][1]);
			}
		}
		for (l = 0; l < lmax; l++)
			E[l] = E[l]  / ((T)2.0 * ((T)nx * (T)nx * (T)ny * (T)ny));
		fftw_destroy_plan(plan);
		
		fftw_free(in);
		fftw_free(out);
		fftw_free(Us);
		fftw_free(Vs);	
		delete[] U; delete[] V;
	}	
}

template< typename T >
void nse::fluxes(T* Flux, T* Flux_ens, T* E, T* k,
	T* U, T* V, T* U_n, T* V_n,
	const T dt, const uniGrid2d< T >& grid)
{
	T *E1, *E2;
	T *Flux_, *Flux_ens_;
	T idt = 1 / dt;
	int lmax = (grid.mpi_nx - 2 * grid.gcx) / 2;
	
	E1 = new T[lmax];
	E2 = new T[lmax];
	Flux_ = new T[lmax];
	Flux_ens_ = new T[lmax];
	
	energy_spectrum(E1, k, U, V, grid);
	energy_spectrum(E2, k, U_n, V_n, grid);
	
	if (grid.mpi_com.rank == 0) {
		printf("sum E1 = %.7f, sum E2 = %.7f\n", sum(E1, lmax), sum(E2, lmax));
		update(E, (T)1.0, E1, lmax);
		assign(Flux_, -idt, E2, idt, E1, lmax);
		
		for (int l = 0; l < lmax; l++)
			Flux_ens_[l] = Flux_[l] * (T)4.0 * pow(sin(k[l] * grid.dx / (T)2.0), (T)2.0) * grid.dxi * grid.dxi;
		for (int l =0; l < lmax; l++)
			Flux_[l] = Flux_[l - 1] + Flux_[l];
		for (int l = 0; l < lmax; l++)
			Flux_ens_[l] = Flux_ens_[l - 1] + Flux_ens_[l];
		update(Flux, (T)1.0, Flux_, lmax);
		update(Flux_ens, (T)1.0, Flux_ens_, lmax);
	}
}
// ------------------------------------------------------------------------ //

template void nse::check_const(const float* w, const char* message,
	const uniGrid2d< float >& grid);
template void nse::check_const(const double* w, const char* message,
	const uniGrid2d< double >& grid);

template void nse::remove_const(float* w,
	const uniGrid2d< float >& grid);
template void nse::remove_const(double* w,
	const uniGrid2d< double >& grid);

template void nse::noise(float* w,
	const float deviation, const uniGrid2d< float >& grid);
template void nse::noise(double* w,
	const double deviation, const uniGrid2d< double >& grid);

// * initialize: large friction * //
template void nse::u_friction(float* Uinterm,
	const float* U, const float mu,
	const uniGrid2d< float >& grid);
template void nse::u_friction(double* Uinterm,
	const double* U, const double mu,
	const uniGrid2d< double >& grid);

template void nse::v_friction(float* Vinterm,
	const float* V, const float mu,
	const uniGrid2d< float >& grid);
template void nse::v_friction(double* Vinterm,
	const double* V, const double mu,
	const uniGrid2d< double >& grid);

template void nse::w_friction(float* winterm,
        const float* w, const float mu,
        const uniGrid2d< float >& grid);
template void nse::w_friction(double* winterm,
        const double* w, const double mu,
        const uniGrid2d< double >& grid);

template void nse::w_beta_effect(float* winterm,
        const float* psi, const float beta,
        const uniGrid2d< float >& grid);
template void nse::w_beta_effect(double* winterm,
        const double* psi, const double beta,
        const uniGrid2d< double >& grid);

// * kabaret * //
template void nse::w_advection_kabaret(float* winterm,
        const float* wx, const float* wy,
	const float* U, const float* V,
        const uniGrid2d< float >& grid);
template void nse::w_advection_kabaret(double* winterm,
        const double* wx, const double* wy,
	const double* U, const double* V,
        const uniGrid2d< double >& grid);

template void nse::w_extrapolation_kabaret(float* wx_n,
        float* wy_n, const float* wx,
	const float* wy, const float* U,
	const float* V, const float* wim,
        const uniGrid2d< float >& grid);
template void nse::w_extrapolation_kabaret(double* wx_n,
        double* wy_n, const double* wx,
	const double* wy, const double* U,
	const double* V, const double* wim,
        const uniGrid2d< double >& grid);

template void nse::w_rhs_kabaret(float* rhs,
	const float* wx, const float* wy,
	const uniGrid2d< float >& grid);
template void nse::w_rhs_kabaret(double* rhs,
	const double* wx, const double* wy,
	const uniGrid2d< double >& grid);

template void nse::w_kabaret(float* wx,
	float* wy, const float* w,
	const uniGrid2d< float >& grid);
template void nse::w_kabaret(double* wx,
	double* wy, const double* w,
	const uniGrid2d< double >& grid);

template void nse::g2( float* wx_n, float* wy_n,
	const float* g1, const float* U,
	const float* V, const float* w, const float* wx, const float* wy, const float dt,
	const uniGrid2d< float >& grid);
template void nse::g2( double* wx_n, double* wy_n,
	const double* g1, const double* U,
	const double* V, const double* w, const double* wx, const double* wy, const double dt,
	const uniGrid2d< double >& grid);




// * initialize: forcing * //
template void nse::forcing(float* wim,
	const float k, const float kb, const float dt, const float E_in,
	const uniGrid2d< float >& grid);
template void nse::forcing(double* wim,
	const double k, const double kb, const double dt, const double E_in,
	const uniGrid2d< double >& grid);

template void nse::forcing_collocated(float* wim,
	const float k, const float kb, const float dt, const float E_in,
	const uniGrid2d< float >& grid);
template void nse::forcing_collocated(double* wim,
	const double k, const double kb, const double dt, const double E_in,
	const uniGrid2d< double >& grid);


// * initialize: advection * //
template void nse::u_advection(float* Uinterm,
	const float* U, const float* V,
	const uniGrid2d< float >& grid);
template void nse::u_advection(double* Uinterm,
	const double* U, const double* V,
	const uniGrid2d< double >& grid);

template void nse::v_advection(float* Vinterm,
	const float* U, const float* V,
	const uniGrid2d< float >& grid);
template void nse::v_advection(double* Vinterm,
	const double* U, const double* V,
	const uniGrid2d< double >& grid);

template void nse::w_mean_flow(float* winterm, 
	const float* w, const float Umean,
	const uniGrid2d< float >& grid);
template void nse::w_mean_flow(double* winterm, 
	const double* w, const double Umean,
	const uniGrid2d< double >& grid);

// * Arakawa jacobians +=
template void nse::w_J1(float* winterm, 
	const float* w, const float* Psi,
	const uniGrid2d< float >& grid);
template void nse::w_J1(double* winterm, 
	const double* w, const double* Psi,
	const uniGrid2d< double >& grid);

template void nse::w_J2(float* winterm, 
	const float* w, const float* Psi,
	const uniGrid2d< float >& grid);
template void nse::w_J2(double* winterm, 
	const double* w, const double* Psi,
	const uniGrid2d< double >& grid);

template void nse::w_J3(float* winterm, 
	const float* w, const float* Psi,
	const uniGrid2d< float >& grid);
template void nse::w_J3(double* winterm, 
	const double* w, const double* Psi,
	const uniGrid2d< double >& grid);

template void nse::J_EZ(float* winterm, 
	const float* w, const float* Psi,
	const uniGrid2d< float >& grid);
template void nse::J_EZ(
    double* winterm, 
	const double* w, const double* Psi,
	const uniGrid2d< double >& grid);


template void nse::add_vertical_shear(
	double* qim1, double* qim2, 
	const double* q1, const double* q2, 
	const double* V1, const double* V2, const double kd,
	const uniGrid2d< double >&grid);
template void nse::add_vertical_shear(
	float* qim1, float* qim2, 
	const float* q1, const float* q2, 
	const float* V1, const float* V2, const float kd,
	const uniGrid2d< float >&grid);

// * usual forms  =
template void nse::w_advection(float* winterm, 
	const float* w, const float *U, const float* V,
	const uniGrid2d< float >& grid);
template void nse::w_advection(double* winterm, 
	const double* w, const double* U, const double* V,
	const uniGrid2d< double >& grid);

template void nse::w_advection_div(float* winterm, 
	const float* w, const float *U, const float* V,
	const uniGrid2d< float >& grid);
template void nse::w_advection_div(double* winterm, 
	const double* w, const double* U, const double* V,
	const uniGrid2d< double >& grid);

template void nse::w_advection_div_x4(float* winterm, 
	const float* w, const float *U, const float* V,
	const uniGrid2d< float >& grid);
template void nse::w_advection_div_x4(double* winterm, 
	const double* w, const double* U, const double* V,
	const uniGrid2d< double >& grid);

template void nse::w_advection_div_stag(float* winterm, 
	const float* w, const float *U, const float* V,
	const uniGrid2d< float >& grid);
template void nse::w_advection_div_stag(double* winterm, 
	const double* w, const double* U, const double* V,
	const uniGrid2d< double >& grid);

template void nse::w_advection_div_stag_1(float* winterm, 
	const float* w, const float *U, const float* V,
	const uniGrid2d< float >& grid);
template void nse::w_advection_div_stag_1(double* winterm, 
	const double* w, const double* U, const double* V,
	const uniGrid2d< double >& grid);

template void nse::w_advection_en_ens(float* winterm, 
	const float* w, const float *U, const float* V, const float* Psi,
	const uniGrid2d< float >& grid);
template void nse::w_advection_en_ens(double* winterm, 
	const double* w, const double* U, const double* V, const double * Psi,
	const uniGrid2d< double >& grid);


template void nse::u_advection_div_x4(float* Uinterm,
	const float* U, const float* V,
	const uniGrid2d< float >& grid);
template void nse::u_advection_div_x4(double* Uinterm,
	const double* U, const double* V,
	const uniGrid2d< double >& grid);

template void nse::v_advection_div_x4(float* Vinterm,
	const float* U, const float* V,
	const uniGrid2d< float >& grid);
template void nse::v_advection_div_x4(double* Vinterm,
	const double* U, const double* V,
	const uniGrid2d< double >& grid);

template void nse::u_advection_skew_x4(float* Uinterm,
	const float* U, const float* V,
	const uniGrid2d< float >& grid);
template void nse::u_advection_skew_x4(double* Uinterm,
	const double* U, const double* V,
	const uniGrid2d< double >& grid);

template void nse::v_advection_skew_x4(float* Vinterm,
	const float* U, const float* V,
	const uniGrid2d< float >& grid);
template void nse::v_advection_skew_x4(double* Vinterm,
	const double* U, const double* V,
	const uniGrid2d< double >& grid);

template void nse::u_advection_weno(float* Uinterm,
	const float* U, const float* V, const uniGrid2d< float >& grid);
template void nse::u_advection_weno(double* Uinterm,
	const double* U, const double* V, const uniGrid2d< double >& grid);

template void nse::v_advection_weno(float* Vinterm,
	const float* U, const float* V, const uniGrid2d< float >& grid);
template void nse::v_advection_weno(double* Vinterm,
	const double* U, const double* V, const uniGrid2d< double >& grid);


template void nse::c_advection(float* Xinterm,
	const float* U, const float* V, const float* X,
	const uniGrid2d< float >& grid);
template void nse::c_advection(double* Xinterm,
	const double* U, const double* V, const double* X,
	const uniGrid2d< double >& grid);

template void nse::c_advection_div_x4(float* Xinterm,
	const float* U, const float* V, const float* X,
	const uniGrid2d< float >& grid);
template void nse::c_advection_div_x4(double* Xinterm,
	const double* U, const double* V, const double* X,
	const uniGrid2d< double >& grid);

template void nse::c_advection_skew_x4(float* Xinterm,
	const float* U, const float* V, const float* X,
	const uniGrid2d< float >& grid);
template void nse::c_advection_skew_x4(double* Xinterm,
	const double* U, const double* V, const double* X,
	const uniGrid2d< double >& grid);

template void nse::c_advection_upwind(float* Xinterm,
	const float* U, const float* V, const float* X,
	const uniGrid2d< float >& grid);
template void nse::c_advection_upwind(double* Xinterm,
	const double* U, const double* V, const double* X,
	const uniGrid2d< double >& grid);

template void nse::c_advection_upwind_x2(float* Xinterm,
	const float* U, const float* V, const float* X,
	const uniGrid2d< float >& grid);
template void nse::c_advection_upwind_x2(double* Xinterm,
	const double* U, const double* V, const double* X,
	const uniGrid2d< double >& grid);

template void nse::c_advection_upwind_x2_div(float* winterm,
	const float* U, const float* V, const float* w,
	const uniGrid2d< float >& grid);
template void nse::c_advection_upwind_x2_div(double* winterm,
	const double* U, const double* V, const double* w,
	const uniGrid2d< double >& grid);

template void nse::c_advection_upwind_x2_conserv(float* winterm,
	const float* U, const float* V, const float* w,
	const uniGrid2d< float >& grid);
template void nse::c_advection_upwind_x2_conserv(double* winterm,
	const double* U, const double* V, const double* w,
	const uniGrid2d< double >& grid);

template void nse::c_advection_upwind_x3(float* winterm,
	const float* U, const float* V, const float* w,
	const uniGrid2d< float >& grid);
template void nse::c_advection_upwind_x3(double* winterm,
	const double* U, const double* V, const double* w,
	const uniGrid2d< double >& grid);

template void nse::c_advection_upwind_x3_div(float* winterm,
	const float* U, const float* V, const float* w,
	const uniGrid2d< float >& grid);
template void nse::c_advection_upwind_x3_div(double* winterm,
	const double* U, const double* V, const double* w,
	const uniGrid2d< double >& grid);

template void nse::c_advection_upwind_x3_conserv(float* winterm,
	const float* U, const float* V, const float* w,
	const uniGrid2d< float >& grid);
template void nse::c_advection_upwind_x3_conserv(double* winterm,
	const double* U, const double* V, const double* w,
	const uniGrid2d< double >& grid);

template void nse::c_advection_tvd(float* Xinterm,
	const float* U, const float* V, const float* X,
	const uniGrid2d< float >& grid);
template void nse::c_advection_tvd(double* Xinterm,
	const double* U, const double* V, const double* X,
	const uniGrid2d< double >& grid);

template void nse::c_advection_weno(float* Xinterm,
	const float* U, const float* V, const float* X,
	const uniGrid2d< float >& grid);
template void nse::c_advection_weno(double* Xinterm,
	const double* U, const double* V, const double* X,
	const uniGrid2d< double >& grid);
// ------------------------------------------------------------------------ //

// * initialize: diffusion * //
template void nse::u_diffusion(float* Uinterm,
	const float* U, const float* V,
	const float c_viscosity, const uniGrid2d< float >& grid);
template void nse::u_diffusion(double* Uinterm,
	const double* U, const double* V,
	const double c_viscosity, const uniGrid2d< double >& grid);

template void nse::v_diffusion(float* Vinterm,
	const float* U, const float* V,
	const float c_viscosity, const uniGrid2d< float >& grid);
template void nse::v_diffusion(double* Vinterm,
	const double* U, const double* V,
	const double c_viscosity, const uniGrid2d< double >& grid);

template void nse::w_diffusion(float* winterm,
	const float* w, const float c_viscosity, 
	const uniGrid2d< float >& grid);
template void nse::w_diffusion(double* winterm,
	const double* w, const double c_viscosity, 
	const uniGrid2d< double >& grid);

template void nse::w_diffusion_2(float* winterm,
        const float* w, const float c_viscosity,
        const uniGrid2d< float >& grid);
template void nse::w_diffusion_2(double* winterm,
        const double* w, const double c_viscosity,
        const uniGrid2d< double >& grid);


template void nse::w_diffusion_x4(float* winterm,
	const float* w, const float c_viscosity, 
	const uniGrid2d< float >& grid);
template void nse::w_diffusion_x4(double* winterm,
	const double* w, const double c_viscosity, 
	const uniGrid2d< double >& grid);

template void nse::c_diffusion(float* Xinterm,
	const float* X,
	const float c_diffusivity, const uniGrid2d< float >& grid);
template void nse::c_diffusion(double* Xinterm,
	const double* X,
	const double c_diffusivity, const uniGrid2d< double >& grid);

template void nse::u_eddy_diffusion(float* Uinterm,
	const float* U, const float* V,
	const float* visc, const uniGrid2d< float >& grid);
template void nse::u_eddy_diffusion(double* Uinterm,
	const double* U, const double* V,
	const double* visc, const uniGrid2d< double >& grid);

template void nse::v_eddy_diffusion(float* Vinterm,
	const float* U, const float* V,
	const float* visc, const uniGrid2d< float >& grid);
template void nse::v_eddy_diffusion(double* Vinterm,
	const double* U, const double* V,
	const double* visc, const uniGrid2d< double >& grid);

template void nse::u_diffusion(float* Uinterm,
	const float* U, const float* V,
	const float* visc, const float* i_density, const uniGrid2d< float >& grid);
template void nse::u_diffusion(double* Uinterm,
	const double* U, const double* V,
	const double* visc, const double* i_density, const uniGrid2d< double >& grid);

template void nse::v_diffusion(float* Vinterm,
	const float* U, const float* V,
	const float* visc, const float* i_density, const uniGrid2d< float >& grid);
template void nse::v_diffusion(double* Vinterm,
	const double* U, const double* V,
	const double* visc, const double* i_density, const uniGrid2d< double >& grid);

template void nse::u_diffusion_x4(float* Uinterm,
	const float* U, const float* V,
	const float c_viscosity, const uniGrid2d< float >& grid);
template void nse::u_diffusion_x4(double* Uinterm,
	const double* U, const double* V,
	const double c_viscosity, const uniGrid2d< double >& grid);

template void nse::v_diffusion_x4(float* Vinterm,
	const float* U, const float* V,
	const float c_viscosity, const uniGrid2d< float >& grid);
template void nse::v_diffusion_x4(double* Vinterm,
	const double* U, const double* V,
	const double c_viscosity, const uniGrid2d< double >& grid);

template void nse::c_diffusion_x4(float* Xinterm,
	const float* X,
	const float c_diffusivity, const uniGrid2d< float >& grid);
template void nse::c_diffusion_x4(double* Xinterm,
	const double* X,
	const double c_diffusivity, const uniGrid2d< double >& grid);
// ------------------------------------------------------------------------ //
// * initialize: velocity * //
template void nse::velocity(float* U,
	float* V, const float* Psi,
	const uniGrid2d< float >& grid);
template void nse::velocity(double* U,
	double* V, const double* Psi,
	const uniGrid2d< double >& grid);

template void nse::velocity_x4(float* U,
	float* V, const float* Psi,
	const uniGrid2d< float >& grid);
template void nse::velocity_x4(double* U,
	double* V, const double* Psi,
	const uniGrid2d< double >& grid);


template void nse::velocity_stag(float* U,
	float* V, const float* Psi,
	const uniGrid2d< float >& grid);
template void nse::velocity_stag(double* U,
	double* V, const double* Psi,
	const uniGrid2d< double >& grid);
// ----------------------------------------------------------------------- //
// * initialize: divergence * //
template void nse::divergence(float* Div,
	const float* U, const float* V,
	const uniGrid2d< float >& grid);
template void nse::divergence(double* Div,
	const double* U, const double* V,
	const uniGrid2d< double >& grid);

template void nse::vorticity(float* Vort,
	const float* U, const float* V,
	const uniGrid2d< float >& grid);
template void nse::vorticity(double* Vort,
	const double* U, const double* V,
	const uniGrid2d< double >& grid);

template void nse::divergence(float* Div,
	const float* U, const float* V,
	const float* u_mass, const float* v_mass,
	const uniGrid2d< float >& grid);
template void nse::divergence(double* Div,
	const double* U, const double* V,
	const double* u_mass, const double *v_mass,
	const uniGrid2d< double >& grid);

template void nse::divergence_x4(float* Div,
	const float* U, const float* V,
	const uniGrid2d< float >& grid);
template void nse::divergence_x4(double* Div,
	const double* U, const double* V,
	const uniGrid2d< double >& grid);
// ------------------------------------------------------------------------ //

// * initialize: kinetic energy * //
template float nse::kinetic_energy(
	const float* U, const float* V, const uniGrid2d< float >& grid);
template double nse::kinetic_energy(
	const double* U, const double* V, const uniGrid2d< double >& grid);

template float nse::kinetic_energy_collocated(
	const float* U, const float* V, const uniGrid2d< float >& grid);
template double nse::kinetic_energy_collocated(
	const double* U, const double* V, const uniGrid2d< double >& grid);
// ------------------------------------------------------------------------ //

// * initialize: sources of enstrophy and energy
template void nse::sources(
	float* Ens, float* Ens_Source, float* En_source, const float* w,
	const float* wim, const float* Psi,
	const uniGrid2d< float >& grid);
template void nse::sources(
	double* Ens, double* Ens_Source, double* En_source, const double* w, 
	const double* wim, const double* Psi, 
	const uniGrid2d< double >& grid);

template void nse::sources_sh(
	s_balance< float >& balance, const float* w,
	const float* wim, const float* Psi,
	const uniGrid2d< float >& grid);
template void nse::sources_sh(
	s_balance< double >& balance, const double* w, 
	const double* wim, const double* Psi, 
	const uniGrid2d< double >& grid);

template void nse::sources_visc(
	s_balance< float >& balance, const float* w,
	const float* wim, const float* Psi,
	const uniGrid2d< float >& grid);
template void nse::sources_visc(
	s_balance< double >& balance, const double* w, 
	const double* wim, const double* Psi, 
	const uniGrid2d< double >& grid);

template void nse::sources_fric(
	s_balance< float >& balance, const float* w,
	const float* wim, const float* Psi,
	const uniGrid2d< float >& grid);
template void nse::sources_fric(
	s_balance< double >& balance, const double* w, 
	const double* wim, const double* Psi, 
	const uniGrid2d< double >& grid);

template void nse::sources_forcing(
	s_balance< float >& balance, const float* w,
	const float* wim, const float* Psi, const float dt, 
	const uniGrid2d< float >& grid);
template void nse::sources_forcing(
	s_balance< double >& balance, const double* w, 
	const double* wim, const double* Psi, const double dt,
	const uniGrid2d< double >& grid);

template void nse::invariant_level(
	s_balance< float >& balance, const float* w,
	const float* Psi, const uniGrid2d< float >& grid);
template void nse::invariant_level(
	s_balance< double >& balance, const double* w, 
	const double* Psi, const uniGrid2d< double >& grid);
// ------------------------------------------------------------------------ //
// * initialize: gradient * //
template void nse::u_sub_gradient(float* Uinterm,
	const float* X,
	const float c_gradient, const uniGrid2d< float >& grid);
template void nse::u_sub_gradient(double* Uinterm,
	const double* X,
	const double c_gradient, const uniGrid2d< double >& grid);

template void nse::v_sub_gradient(float* Vinterm,
	const float* X,
	const float c_gradient, const uniGrid2d< float >& grid);
template void nse::v_sub_gradient(double* Vinterm,
	const double* X,
	const double c_gradient, const uniGrid2d< double >& grid);

template void nse::u_add_gradient(float* Uinterm,
	const float* X,
	const float c_gradient, const uniGrid2d< float >& grid);
template void nse::u_add_gradient(double* Uinterm,
	const double* X,
	const double c_gradient, const uniGrid2d< double >& grid);

template void nse::v_add_gradient(float* Vinterm,
	const float* X,
	const float c_gradient, const uniGrid2d< float >& grid);
template void nse::v_add_gradient(double* Vinterm,
	const double* X,
	const double c_gradient, const uniGrid2d< double >& grid);

template void nse::u_sub_gradient(float* Uinterm,
	const float* X,
	const float* c_gradient, const uniGrid2d< float >& grid);
template void nse::u_sub_gradient(double* Uinterm,
	const double* X,
	const double* c_gradient, const uniGrid2d< double >& grid);

template void nse::v_sub_gradient(float* Vinterm,
	const float* X,
	const float* c_gradient, const uniGrid2d< float >& grid);
template void nse::v_sub_gradient(double* Vinterm,
	const double* X,
	const double* c_gradient, const uniGrid2d< double >& grid);

template void nse::u_add_gradient(float* Uinterm,
	const float* X,
	const float* c_gradient, const uniGrid2d< float >& grid);
template void nse::u_add_gradient(double* Uinterm,
	const double* X,
	const double* c_gradient, const uniGrid2d< double >& grid);

template void nse::v_add_gradient(float* Vinterm,
	const float* X,
	const float* c_gradient, const uniGrid2d< float >& grid);
template void nse::v_add_gradient(double* Vinterm,
	const double* X,
	const double* c_gradient, const uniGrid2d< double >& grid);

template void nse::u_sub_gradient_x4(float* Uinterm,
	const float* X,
	const float c_gradient, const uniGrid2d< float >& grid);
template void nse::u_sub_gradient_x4(double* Uinterm,
	const double* X,
	const double c_gradient, const uniGrid2d< double >& grid);

template void nse::v_sub_gradient_x4(float* Vinterm,
	const float* X,
	const float c_gradient, const uniGrid2d< float >& grid);
template void nse::v_sub_gradient_x4(double* Vinterm,
	const double* X,
	const double c_gradient, const uniGrid2d< double >& grid);

template void nse::u_add_gradient_x4(float* Uinterm,
	const float* X,
	const float c_gradient, const uniGrid2d< float >& grid);
template void nse::u_add_gradient_x4(double* Uinterm,
	const double* X,
	const double c_gradient, const uniGrid2d< double >& grid);

template void nse::v_add_gradient_x4(float* Vinterm,
	const float* X,
	const float c_gradient, const uniGrid2d< float >& grid);
template void nse::v_add_gradient_x4(double* Vinterm,
	const double* X,
	const double c_gradient, const uniGrid2d< double >& grid);
// ------------------------------------------------------------------------ //

// * initialize: poisson eq. rhs * //
template void nse::poisson_rhs(float* Rhs,
	const float* Div,
	const float* Uinterm, const float* Vinterm,
	const uniGrid2d< float >& grid, const float dt);
template void nse::poisson_rhs(double* Rhs,
	const double* Div,
	const double* Uinterm, const double* Vinterm,
	const uniGrid2d< double >& grid, const double dt);

template void nse::poisson_rhs(float* Rhs,
	const float* U, const float* V,
	const float* Uinterm, const float* Vinterm,
	const uniGrid2d< float >& grid, const float dt);
template void nse::poisson_rhs(double* Rhs,
	const double* U, const double* V,
	const double* Uinterm, const double* Vinterm,
	const uniGrid2d< double >& grid, const double dt);

template void nse::poisson_rhs_x4(float* Rhs,
	const float* Div,
	const float* Uinterm, const float* Vinterm,
	const uniGrid2d< float >& grid, const float dt);
template void nse::poisson_rhs_x4(double* Rhs,
	const double* Div,
	const double* Uinterm, const double* Vinterm,
	const uniGrid2d< double >& grid, const double dt);


template void nse::poisson_rhs(float* Rhs,
	const float* U, const float* V,
	const float* Uinterm, const float* Vinterm,
	const float* u_mass, const float* v_mass,
	const uniGrid2d< float >& grid, const float dt);
template void nse::poisson_rhs(double* Rhs,
	const double* U, const double* V,
	const double* Uinterm, const double* Vinterm,
	const double* u_mass, const double* v_mass,
	const uniGrid2d< double >& grid, const double dt);
// ------------------------------------------------------------------------ //
// * initialize: time advancement * //
template void nse::adams_bashforth_x2(float* Xn, const float* X, float* Xp,
	const uniGrid2d< float >& grid, const float dt);
template void nse::adams_bashforth_x2(double* Xn, const double* X, double* Xp,
	const uniGrid2d< double >& grid, const double dt);

template void nse::adams_bashforth_x2(float* X, float* Xp,
	const uniGrid2d< float >& grid);
template void nse::adams_bashforth_x2(double* X, double* Xp,
	const uniGrid2d< double >& grid);

template void nse::adams_bashforth_x2(float* X, float* Xp,
	const uniGrid2d< float >& grid, const float dt, const float p_dt);
template void nse::adams_bashforth_x2(double* X, double* Xp,
	const uniGrid2d< double >& grid, const double dt, const double p_dt);

template void nse::adams_bashforth_x2(float* X, float* Xp,
	const float eps,
	const uniGrid2d< float >& grid);
template void nse::adams_bashforth_x2(double* X, double* Xp,
	const double eps,
	const uniGrid2d< double >& grid);

template void nse::adams_bashforth_x3(float* X, float* Xp, float* Xpp,
	const uniGrid2d< float >& grid);
template void nse::adams_bashforth_x3(double* X, double* Xp, double* Xpp,
	const uniGrid2d< double >& grid);

template void nse::adams_bashforth_x3(float* X, float* Xp, float* Xpp,
	const uniGrid2d< float >& grid,
	const float dt, const float p_dt, const float pp_dt);
template void nse::adams_bashforth_x3(double* X, double* Xp, double* Xpp,
	const uniGrid2d< double >& grid,
	const double dt, const double p_dt, const double pp_dt);
// ------------------------------------------------------------------------ //


// * initialize: projection * //
template void nse::u_projection(float* U,
	const float* Uinterm, const float* Phi,
	const uniGrid2d< float >& grid, const float dt);
template void nse::u_projection(double* U,
	const double* Uinterm, const double* Phi,
	const uniGrid2d< double >& grid, const double dt);

template void nse::v_projection(float* V,
	const float* Vinterm, const float* Phi,
	const uniGrid2d< float >& grid, const float dt);
template void nse::v_projection(double* V,
	const double* Vinterm, const double* Phi,
	const uniGrid2d< double >& grid, const double dt);

template void nse::u_projection(float* U,
	const float* Uinterm, const float* Phi, const float* i_density,
	const uniGrid2d< float >& grid, const float dt);
template void nse::u_projection(double* U,
	const double* Uinterm, const double* Phi, const double* i_density,
	const uniGrid2d< double >& grid, const double dt);

template void nse::v_projection(float* V,
	const float* Vinterm, const float* Phi, const float* i_density,
	const uniGrid2d< float >& grid, const float dt);
template void nse::v_projection(double* V,
	const double* Vinterm, const double* Phi, const double* i_density,
	const uniGrid2d< double >& grid, const double dt);

template void nse::u_projection_x4(float* U,
	const float* Uinterm, const float* Phi,
	const uniGrid2d< float >& grid, const float dt);
template void nse::u_projection_x4(double* U,
	const double* Uinterm, const double* Phi,
	const uniGrid2d< double >& grid, const double dt);

template void nse::v_projection_x4(float* V,
	const float* Vinterm, const float* Phi,
	const uniGrid2d< float >& grid, const float dt);
template void nse::v_projection_x4(double* V,
	const double* Vinterm, const double* Phi,
	const uniGrid2d< double >& grid, const double dt);
// ------------------------------------------------------------------------ //

// * initialize: buoyancy * //
template void nse::u_buoyancy(float* Uinterm, const float* X,
	const float c_expansion, const float c_gravity_x, const uniGrid2d< float >& grid);
template void nse::u_buoyancy(double* Uinterm, const double* X,
	const double c_expansion, const double c_gravity_x, const uniGrid2d< double >& grid);

template void nse::v_buoyancy(float* Vinterm, const float* X,
	const float c_expansion, const float c_gravity_y, const uniGrid2d< float >& grid);
template void nse::v_buoyancy(double* Vinterm, const double* X,
	const double c_expansion, const double c_gravity_y, const uniGrid2d< double >& grid);

template void nse::u_buoyancy_x4(float* Uinterm, const float* X,
	const float c_expansion, const float c_gravity_x, const uniGrid2d< float >& grid);
template void nse::u_buoyancy_x4(double* Uinterm, const double* X,
	const double c_expansion, const double c_gravity_x, const uniGrid2d< double >& grid);

template void nse::v_buoyancy_x4(float* Vinterm, const float* X,
	const float c_expansion, const float c_gravity_y, const uniGrid2d< float >& grid);
template void nse::v_buoyancy_x4(double* Vinterm, const double* X,
	const double c_expansion, const double c_gravity_y, const uniGrid2d< double >& grid);
// ------------------------------------------------------------------------ //

// * initialize: velocity abs max * //
template void nse::velocity_abs_max(float* umax, float* vmax,
	const float* U, const float* V,
	const uniGrid2d< float >& grid);
template void nse::velocity_abs_max(double* umax, double* vmax,
	const double* U, const double* V,
	const uniGrid2d< double >& grid);
// ------------------------------------------------------------------------ //

// * Energy spectrum * //
template void nse::energy_spectrum( float* E, float* k, float* Uin, float* Vin,
	const uniGrid2d< float >& grid);
template void nse::energy_spectrum( double* E, double* k, double* Uin, double* Vin,
	const uniGrid2d< double >& grid);

template void nse::fluxes(float* Flux, float* Flux_ens, float* E, float* k,
	float* U, float* V, float* U_n, float* V_n,
	const float dt, const uniGrid2d< float >& grid);
template void nse::fluxes(double* Flux, double* Flux_ens, double* E, double* k,
	double* U, double* V, double* U_n, double* V_n,
	const double dt, const uniGrid2d< double >& grid);
