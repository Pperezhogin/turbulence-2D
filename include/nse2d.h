#pragma once

// *    2D Navier-Stokes module  * //

#include "unigrid2d.h"
#include <fftw3.h>
// *[nse2d.h]: base only //

namespace nse
{		
  	// balance of invariants
	template< typename T >
	struct s_balance {
		T *wim_p;
		int count; //number of records to get mean value
		bool status; // if true then 	  
		
		// all variables per unit square
		
		// mean values will be expressed in e_in and ens_in
		T m_en; // mean energy after reach equilibrium
		T m_ens; // mean enstrophy
		T m_en_sh; // energy source by advection scheme  dE/dt = m_en_sh
		T m_ens_sh; // enstrophy source by advection scheme
		T m_en_visc; // energy source by viscosity
		T m_ens_visc; // enstrophy source by viscosity
		T m_en_fric; // energy source by friction
		T m_ens_fric; // enstrophy source by friction
		T m_en_forcing; // energy input from forcing got from direct computation
		T m_ens_forcing; // enstrophy input from forcing
		
		// instantaneous values
		T en;
		T ens;
        T en_visc;
        T ens_visc;
		
		// deviation from balance to check if specified relation between energy and enstrophy is destroyed
		T dev_en;
		T dev_ens;
		
		public:
			void init(const T E_in, const uniGrid2d< T >& grid) {
				count = 0;
				allocate(&wim_p, grid.size);
				
				m_en = (T)0.0;
				m_ens = (T)0.0;
				m_en_sh = (T)0.0;
				m_ens_sh = (T)0.0;
				m_en_visc = (T)0.0;
				m_ens_visc = (T)0.0;
				m_en_fric = (T)0.0;
				m_ens_fric = (T)0.0;
				m_en_forcing = (T)0.0;
				m_ens_forcing = (T)0.0;
				m_en_forcing = E_in;
			}
			
			void processing() {
			  	m_ens_forcing /= (T)count;
			  
				m_en /= (T)count;
				m_ens /= (T)count;
				
				m_en_sh /= (T)count * m_en_forcing;
				m_ens_sh /= (T)count * m_ens_forcing;
				m_en_visc /= (T)count * m_en_forcing;
				m_ens_visc /= (T)count * m_ens_forcing;
				m_en_fric /= (T)count * m_en_forcing;
				m_ens_fric /= (T)count * m_ens_forcing;
				
				dev_en = m_en_sh + m_en_visc + m_en_fric + (T)1.0;
				dev_ens = m_ens_sh + m_ens_visc + m_ens_fric + (T)1.0;
			}
			
			void update_status(bool current_status, uniGrid2d< T >& grid)
			{
				status = current_status;
				if (status == true) {
					count++;
					null(wim_p, grid.size);
				}
			}
	}; 
	
	template < typename T >
	void remove_const(T* w,
	const uniGrid2d< T >& grid);
	
	template < typename T >
	void check_const(const T* w, const char* message,
	const uniGrid2d< T >& grid);

	template < typename T >
	void noise(T* w, const T deviation,
	const uniGrid2d< T >& grid);
	
	// * kabaret * //
	template <typename T>
	void w_advection_kabaret(T* winterm,
        const T* wx, const T* wy,
	const T* U, const T* V,
        const uniGrid2d< T >& grid);
	
	template <typename T>
	void w_extrapolation_kabaret(T* wx_n,
        T* wy_n, const T* wx,
	const T* wy, const T* U,
	const T* V, const T* wim,
        const uniGrid2d< T >& grid);
	
	template <typename T>
	void w_rhs_kabaret(T* rhs,
	const T* wx, const T* wy,
	const uniGrid2d< T >& grid);
	
	template <typename T>
	void w_kabaret(T* wx,
	T* wy, const T* w,
	const uniGrid2d< T >& grid);
	
	template< typename T>
	void g2(T* wx_n, T* wy_n,
	const T* g1, const T* U,
	const T* V, const T* w, const T* wx, const T* wy, const T dt,
	const uniGrid2d< T >& grid);
	
	// * friction * //
	template< typename T >
	void u_friction(T* Uinterm, const T* U, const T mu,
	const uniGrid2d< T >& grid);
	template< typename T >
	void v_friction(T* Vinterm, const T* V, const T mu,
	const uniGrid2d< T >& grid);

    template< typename T >
        void w_friction(T* winterm, const T* w, const T mu,
        const uniGrid2d< T >& grid);

	template < typename T >
		void w_beta_effect(T* winterm,
        const T* psi, const T beta,
        const uniGrid2d< T >& grid);

	// * forcing * //
	template< typename T >
	void forcing(T* wim, const T k, const T kb, const T dt,
		const T E_in, const uniGrid2d< T >& grid);
	template< typename T >
	void forcing_collocated(T* wim, const T k, const T kb, const T dt,
		const T E_in, const uniGrid2d< T >& grid);
      
	// * advection [ := - ] * //
	template< typename T >
	void u_advection(T* Uinterm, const T* U, const T* V,
		const uniGrid2d< T >& grid);
	template< typename T >
	void v_advection(T* Vinterm, const T* U, const T* V,
		const uniGrid2d< T >& grid);
	
	template < typename T>
	void w_mean_flow(T* winterm, 
	const T* w, const T Umean,
	const uniGrid2d< T >& grid);

	// * Arakawa jacobians [ += -]
	template < typename T>	
	void w_J1(T* winterm, const T* w, const T* Psi,
	const uniGrid2d< T >&grid);
	template < typename T>	
	void w_J2(T* winterm, const T* w, const T* Psi,
	const uniGrid2d< T >&grid);
	template < typename T>	
	void w_J3(T* winterm, const T* w, const T* Psi,
	const uniGrid2d< T >&grid);

	template < typename T>
    void J_EZ(T* winterm, const T* w, const T* Psi,
	const uniGrid2d< T >&grid);

	// * unit vertical shear [ += ]
	template < typename T>
	void add_vertical_shear(
	T* qim1, T* qim2, const T* q1, const T* q2, const T* V1, const T* V2, const T kd,
	const uniGrid2d< T >&grid);
	
	// * usual form [ := - ]
	template< typename T>
	void w_advection(T* winterm, const T* w, const T* U, const T* V,
		const uniGrid2d< T >& grid);
	template< typename T>
	void w_advection_div(T* winterm, const T* w, const T* U, const T* V,
		const uniGrid2d< T >& grid);
	template< typename T>
	void w_advection_div_x4(T* winterm, const T* w, const T* U, const T* V,
		const uniGrid2d< T >& grid);
	template< typename T>
	void w_advection_div_stag(T* winterm, const T* w, const T* U, const T* V,
		const uniGrid2d< T >& grid);
	template< typename T>
	void w_advection_div_stag_1(T* winterm, const T* w, const T* U, const T* V,
		const uniGrid2d< T >& grid);	
	template< typename T>
	void w_advection_en_ens(T* winterm, const T* w, const T* U, const T* V, const T* Psi,
		const uniGrid2d< T >& grid);
	
	template< typename T >
	void u_advection_div_x4(T* Uinterm, const T* U, const T* V,
		const uniGrid2d< T >& grid);
	template< typename T >
	void v_advection_div_x4(T* Vinterm, const T* U, const T* V,
		const uniGrid2d< T >& grid);

	template< typename T >
	void u_advection_skew_x4(T* Uinterm, const T* U, const T* V,
		const uniGrid2d< T >& grid);
	template< typename T >
	void v_advection_skew_x4(T* Vinterm, const T* U, const T* V,
		const uniGrid2d< T >& grid);


	template< typename T >
	void u_advection_weno(T* Uinterm, const T* U, const T* V,
		const uniGrid2d< T >& grid);
	template< typename T >
	void v_advection_weno(T* Vinterm, const T* U, const T* V,
		const uniGrid2d< T >& grid);


	template< typename T >
	void c_advection(T* Xinterm, const T* U, const T* V, const T* X,
		const uniGrid2d< T >& grid);

	template< typename T >
	void c_advection_div_x4(T* Xinterm, const T* U, const T* V, const T* X,
		const uniGrid2d< T >& grid);
	template< typename T >
	void c_advection_skew_x4(T* Xinterm, const T* U, const T* V, const T* X,
		const uniGrid2d< T >& grid);

	template< typename T >
	void c_advection_upwind(T* Xinterm, const T* U, const T* V, const T* X,
		const uniGrid2d< T >& grid);
	
	template< typename T >
	void c_advection_upwind_x2(T* Xinterm, const T* U, const T* V, const T* X,
		const uniGrid2d< T >& grid);
	
	template< typename T >
	void c_advection_upwind_x2_div(T* winterm, const T* U, const T* V, const T* w,
		const uniGrid2d< T >& grid);
	
	template< typename T >
	void c_advection_upwind_x2_conserv(T* winterm, const T* U, const T* V, const T* w,
		const uniGrid2d< T >& grid);
	
	template< typename T >
	void c_advection_upwind_x3(T* winterm, const T* U, const T* V, const T* w,
		const uniGrid2d< T >& grid);
	
	template< typename T >
	void c_advection_upwind_x3_div(T* winterm, const T* U, const T* V, const T* w,
		const uniGrid2d< T >& grid);

	template< typename T >
	void c_advection_upwind_x3_conserv(T* winterm, const T* U, const T* V, const T* w,
		const uniGrid2d< T >& grid);
	
	template< typename T >
	void c_advection_tvd(T* Xinterm, const T* U, const T* V, const T* X,
		const uniGrid2d< T >& grid);

	template< typename T >
	void c_advection_weno(T* Xinterm, const T* U, const T* V, const T* X,
		const uniGrid2d< T >& grid);
	// -------------------------------------------------------------------- //

	// * diffusion [ += ] * //
	template< typename T >
	void u_diffusion(T* Uinterm, const T* U, const T* V,
		const T c_viscosity, const uniGrid2d< T >& grid);
	template< typename T >
	void v_diffusion(T* Vinterm, const T* U, const T* V,
		const T c_viscosity, const uniGrid2d< T >& grid);

	template< typename T >
	void w_diffusion(T* winterm, const T* w,
		const T c_viscosity, const uniGrid2d< T >& grid);

	template< typename T >
        void w_diffusion_2(T* winterm, const T* w,
                const T c_viscosity, const uniGrid2d< T >& grid);
	
	template< typename T >
	void w_diffusion_x4(T* winterm, const T* w,
		const T c_viscosity, const uniGrid2d< T >& grid);
	
	template< typename T >
	void c_diffusion(T* Xinterm, const T* X,
		const T c_diffusivity, const uniGrid2d< T >& grid);

	template< typename T >
	void u_diffusion(T* Uinterm, const T* U, const T* V,
		const T* visc, const T* i_density, const uniGrid2d< T >& grid);
	template< typename T >
	void v_diffusion(T* Vinterm, const T* U, const T* V,
		const T* visc, const T* i_density, const uniGrid2d< T >& grid);

	template< typename T >
	void u_eddy_diffusion(T* Uinterm, const T* U, const T* V,
		const T* visc, const uniGrid2d< T >& grid);
	template< typename T >
	void v_eddy_diffusion(T* Vinterm, const T* U, const T* V,
		const T* visc, const uniGrid2d< T >& grid);

	template< typename T >
	void u_diffusion_x4(T* Uinterm, const T* U, const T* V,
		const T c_viscosity, const uniGrid2d< T >& grid);
	template< typename T >
	void v_diffusion_x4(T* Vinterm, const T* U, const T* V,
		const T c_viscosity, const uniGrid2d< T >& grid);
	template< typename T >
	void c_diffusion_x4(T* Xinterm, const T* X,
		const T c_diffusivity, const uniGrid2d< T >& grid);
	// -------------------------------------------------------------------- //
	// * velocity * //
	template< typename T >
	void velocity(T* U, T* V, const T* Psi,
		const uniGrid2d< T >& grid);
	template< typename T >
	void velocity_x4(T* U, T* V, const T* Psi,
		const uniGrid2d< T >& grid);
	template< typename T >
	void velocity_stag(T* U, T* V, const T* Psi,
		const uniGrid2d< T >& grid);
	// -------------------------------------------------------------------- //

	// * divergence [ := + ] * //
	template< typename T >
	void divergence(T* Div, const T* U, const T* V,
		const uniGrid2d< T >& grid);
    
    template< typename T >
	void vorticity(T* Vort, const T* U, const T* V,
		const uniGrid2d< T >& grid);

	template< typename T >
	void divergence(T* Div, const T* U, const T* V,
		const T* u_mass, const T* v_mass,
		const uniGrid2d< T >& grid);

	template< typename T >
	void divergence_x4(T* Div, const T* U, const T* V,
		const uniGrid2d< T >& grid);
	// -------------------------------------------------------------------- //

	// * kinetic energy [ := + ] * //
	template< typename T >
	T kinetic_energy(const T* U, const T* V,
		const uniGrid2d< T >& grid);
	
	template< typename T >
	T kinetic_energy_collocated(const T* U, const T* V,
		const uniGrid2d< T >& grid);
	// -------------------------------------------------------------------- //
	// * Enstrophy * //
	
	template< typename T >
	void sources(T* Ens, T* Ens_Source, T* En_Source, const T* w,
		const T* wim, const T* Psi, const uniGrid2d< T >& grid);
	
	template< typename T >
	void sources_sh(s_balance< T >& balance, const T* w,
		const T* wim, const T* Psi, const uniGrid2d< T >& grid);
	
	template< typename T >
	void sources_visc(s_balance< T >& balance, const T* w,
		const T* wim, const T* Psi, const uniGrid2d< T >& grid);
	
	template< typename T >
	void sources_fric(s_balance< T >& balance, const T* w,
		const T* wim, const T* Psi, const uniGrid2d< T >& grid);

	template< typename T >
	void sources_forcing(s_balance< T >& balance, const T* w,
		const T* wim, const T* Psi, const T dt, 
		const uniGrid2d< T >& grid);
	
	template< typename T >
	void invariant_level(s_balance< T >& balance, const T* w,
		const T* Psi, const uniGrid2d< T >& grid);
	
	
	// -------------------------------------------------------------------- //

	// * gradient [ -=, += ] * //
	template< typename T >
	void u_sub_gradient(T* Uinterm, const T* X,
		const T c_gradient, const uniGrid2d< T >& grid);
	template< typename T >
	void v_sub_gradient(T* Vinterm, const T* X,
		const T c_gradient, const uniGrid2d< T >& grid);

	template< typename T >
	void u_add_gradient(T* Uinterm, const T* X,
		const T c_gradient, const uniGrid2d< T >& grid);
	template< typename T >
	void v_add_gradient(T* Vinterm, const T* X,
		const T c_gradient, const uniGrid2d< T >& grid);

	template< typename T >
	void u_sub_gradient(T* Uinterm, const T* X,
		const T* c_gradient, const uniGrid2d< T >& grid);
	template< typename T >
	void v_sub_gradient(T* Vinterm, const T* X,
		const T* c_gradient, const uniGrid2d< T >& grid);

	template< typename T >
	void u_add_gradient(T* Uinterm, const T* X,
		const T* c_gradient, const uniGrid2d< T >& grid);
	template< typename T >
	void v_add_gradient(T* Vinterm, const T* X,
		const T* c_gradient, const uniGrid2d< T >& grid);


	template< typename T >
	void u_sub_gradient_x4(T* Uinterm, const T* X,
		const T c_gradient, const uniGrid2d< T >& grid);
	template< typename T >
	void v_sub_gradient_x4(T* Vinterm, const T* X,
		const T c_gradient, const uniGrid2d< T >& grid);

	template< typename T >
	void u_add_gradient_x4(T* Uinterm, const T* X,
		const T c_gradient, const uniGrid2d< T >& grid);
	template< typename T >
	void v_add_gradient_x4(T* Vinterm, const T* X,
		const T c_gradient, const uniGrid2d< T >& grid);
	// -------------------------------------------------------------------- //


	// * poisson equation rhs [ := ] * //
	template< typename T >
	void poisson_rhs(T* Rhs, 
		const T* Div, 
		const T* Uinterm, const T* Vinterm,
		const uniGrid2d< T >& grid, const T dt);
	template< typename T >
	void poisson_rhs(T* Rhs,
		const T* U, const T* V,
		const T* Uinterm, const T* Vinterm,
		const uniGrid2d< T >& grid, const T dt);

	template< typename T >
	void poisson_rhs(T* Rhs,
		const T* U, const T* V,
		const T* Uinterm, const T* Vinterm,
		const T* u_mass, const T* v_mass,
		const uniGrid2d< T >& grid, const T dt);

	template< typename T >
	void poisson_rhs_x4(T* Rhs,
		const T* Div,
		const T* Uinterm, const T* Vinterm,
		const uniGrid2d< T >& grid, const T dt);
	// -------------------------------------------------------------------- //

	// * adams-bashforth time advancement * //
	template< typename T >
	void adams_bashforth_x2(T* Xn, const T* X, T* Xp, const uniGrid2d< T >& grid, const T dt);
	
	template< typename T >
	void adams_bashforth_x2(T* X, T* Xp, const uniGrid2d< T >& grid);

	template< typename T >
	void adams_bashforth_x2(T* X, T* Xp, const uniGrid2d< T >& grid, const T dt, const T p_dt);

	template< typename T >
	void adams_bashforth_x2(T* X, T* Xp, const T eps, const uniGrid2d< T >& grid);

	template< typename T >
	void adams_bashforth_x3(T* X, T* Xp, T* Xpp, const uniGrid2d< T >& grid);

	template< typename T >
	void adams_bashforth_x3(T* X, T* Xp, T* Xpp, const uniGrid2d< T >& grid,
		const T dt, const T p_dt, const T pp_dt);
	// -------------------------------------------------------------------- //

	// * velocity projection * //
	template< typename T >
	void u_projection(T* U, const T* Uinterm, const T* Phi,
		const uniGrid2d< T >& grid, const T dt);
	template< typename T >
	void v_projection(T* V, const T* Vinterm, const T* Phi,
		const uniGrid2d< T >& grid, const T dt);

	template< typename T >
	void u_projection(T* U, const T* Uinterm, const T* Phi, const T* i_density,
		const uniGrid2d< T >& grid, const T dt);
	template< typename T >
	void v_projection(T* V, const T* Vinterm, const T* Phi, const T* i_density,
		const uniGrid2d< T >& grid, const T dt);

	template< typename T >
	void u_projection_x4(T* U, const T* Uinterm, const T* Phi,
		const uniGrid2d< T >& grid, const T dt);
	template< typename T >
	void v_projection_x4(T* V, const T* Vinterm, const T* Phi,
		const uniGrid2d< T >& grid, const T dt);
	// -------------------------------------------------------------------- //

	// * buoyancy  [ -= ] * //
	template< typename T >
	void u_buoyancy(T* Uinterm, const T* X,
		const T c_expansion, const T c_gravity_x, const uniGrid2d< T >& grid);
	template< typename T >
	void v_buoyancy(T* Vinterm, const T* X,
		const T c_expansion, const T c_gravity_y, const uniGrid2d< T >& grid);

	template< typename T >
	void u_buoyancy_x4(T* Uinterm, const T* X,
		const T c_expansion, const T c_gravity_x, const uniGrid2d< T >& grid);
	template< typename T >
	void v_buoyancy_x4(T* Vinterm, const T* X,
		const T c_expansion, const T c_gravity_y, const uniGrid2d< T >& grid);
	// -------------------------------------------------------------------- //

	// * velocity abs max * //
	template< typename T >
	void velocity_abs_max(T* umax, T* vmax,
		const T* U, const T* V,
		const uniGrid2d< T >& grid);
	// -------------------------------------------------------------------- //
	
	// * Energy spectrum * //
	template< typename T >
	void energy_spectrum( T* E, T* k, T* Uin, T* Vin,
	const uniGrid2d< T >& grid);
	
	template< typename T >
	void fluxes(T* Flux, T* Flux_ens, T* E, T* k,
		T* U, T* V, T* U_n, T* V_n, const T dt,
		const uniGrid2d< T >& grid);
	// -------------------------------------------------------------------- //
	//modified wavenumber
	template< typename T >
	inline T m_k(T k, T k_max) {
		T phi;
		phi = M_PI * k / k_max;
		return k * sin(phi / (T)2.0) * (T)2.0 / phi;
	}
}

