template < typename T >
void get_invaraints_fluxes_viscosity(T &Ec, T &Zc, T &Ediss, T &Zdiss,
T &Cs_lap_Zdiss, T &Cs_lap_MSE, T &Cs_bilap_Zdiss, T &Cs_bilap_MSE, 
T &Cs_ssm_bilap_Zdiss, T &Cs_ssm_bilap_MSE,
T &Cs_ssm_bilap_keb_ZEdiss, T &Cs_ssm_bilap_keb_MSE,
T &Cr_ZEdiss, T &Cr_MSE,
T &Cs_lap_dyn, T &Cs_bilap_dyn, T &Cs_ssm_bilap_dyn,
T &Cs_ssm_bilap_keb_dyn, T &Cr_dyn,
T &MSE_lap, T &MSE_bilap, 
T &MSE_ssm_bilap, T &MSE_ssm_bilap_keb,
T* psi, T* w, T* u, T* v, const T bf_width, const uniGrid2d< T >&grid,
dynamic_model< T >& dyn_model_lap, dynamic_model< T >& dyn_model_bilap, 
dynamic_model< T >& dyn_model_ssm_bilap, dynamic_model< T >& dyn_model_ssm_bilap_keb
);