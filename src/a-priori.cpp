#include "dynamic-model.h"
#include "memory-reader.h"

/*
    tau_j s_j = Cs^4 t_j s_j + Cr b_j s_j
    tau_j p_j = Cs^4 t_j p_j + Cr b_j p_j
    ->
    f = A * x,
    x = [Cs^4, Cr]^T
*/
template < typename T >
void Cs_Cr_estimate_ZEdiss(T &Cs, T &Cr, T* taux, T* tauy, T* tx, T* ty, T* bx, T* by, T* sx, T*sy, T* px, T* py, const uniGrid2d< T >&grid) {
    T detA;
    T ts, bs, tp, bp;
    T taus, taup;
    T Cs4;

    ts = integrate_xy(tx, ty, sx, sy, grid);
    bs = integrate_xy(bx, by, sx, sy, grid);
    tp = integrate_xy(tx, ty, px, py, grid);
    bp = integrate_xy(bx, by, px, py, grid);
    taus = integrate_xy(taux, tauy, sx, sy, grid);
    taup = integrate_xy(taux, tauy, px, py, grid);

    detA = ts * bp - bs * tp;

    Cs4 = (+bp*taus - bs*taup) / detA;
    Cr  = (-tp*taus + ts*taup) / detA;

    Cs = pow(Cs4, (T)0.25);
}

/*
    tau_j t_j = Cs^4 t_j t_j + Cr b_j t_j
    tau_j b_j = Cs^4 t_j b_j + Cr b_j b_j
    ->
    f = A * x,
    x = [Cs^4, Cr]^T
*/
template < typename T >
void Cs_Cr_estimate_MSE(T &Cs, T &Cr, T* taux, T* tauy, T* tx, T* ty, T* bx, T* by, const uniGrid2d< T >&grid) {
    T detA;
    T tt, bt, tb, bb;
    T taut, taub;
    T Cs4;

    tt = integrate_xy(tx, ty, tx, ty, grid);
    bt = integrate_xy(bx, by, tx, ty, grid);
    tb = bt;
    tb = integrate_xy(tx, ty, bx, by, grid);
    bb = integrate_xy(bx, by, bx, by, grid);
    taut = integrate_xy(taux, tauy, tx, ty, grid);
    taub = integrate_xy(taux, tauy, bx, by, grid);

    detA = tt * bb - bt * tb;

    Cs4 = (+bb*taut - bt*taub) / detA;
    Cr  = (-tb*taut + tt*taub) / detA;

    Cs = pow(Cs4, (T)0.25);
}

template < typename T >
T Cs_estimate_Zdiss(T* taux, T* tauy, T* tx, T* ty, T* sx, T*sy, const uniGrid2d< T >&grid) {
    T numerator, denominator;
    numerator   = integrate_xy(taux, tauy, sx, sy, grid);
    denominator = integrate_xy(tx, ty, sx, sy, grid);
    return numerator / denominator; 
}

template < typename T >
T Cs_estimate_MSE(T* taux, T* tauy, T* tx, T* ty, const uniGrid2d< T >&grid) {
    T numerator, denominator;
    numerator   = integrate_xy(taux, tauy, tx, ty, grid);
    denominator = integrate_xy(tx, ty, tx, ty, grid);
    return numerator / denominator; 
}

template < typename T >
T MSE_error(T* div_tau, T* div_model, const uniGrid2d< T >&grid) {
    T err[grid.size];
    T numerator, denominator;
    assign(err, (T)1.0, div_tau, -(T)1.0, div_model, grid.size);

    numerator   = integrate_xy(err, err, grid);
    denominator = integrate_xy(div_tau, div_tau, grid);

    return numerator / denominator;
}

//bf width in absolute value
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
)
{
    T psic[grid.size], wc[grid.size];
    T uc[grid.size], vc[grid.size];
    T taux[grid.size], tauy[grid.size]; // subgrid vorticity flux
    T lx[grid.size], ly[grid.size];
    T div_tau[grid.size]; // subgrid vorticity tendency
    T tx[grid.size], ty[grid.size]; // viscosity model
    T bx[grid.size], by[grid.size]; // reynolds backscatter model
    T sx[grid.size], sy[grid.size]; // nabla omega
    T px[grid.size], py[grid.size]; // nabla psi
    T _taux[grid.size], _tauy[grid.size]; //subgrid force minus leonard stress
    T div_model[grid.size]; // full model vector
    
    gauss_filter(uc, u, bf_width, grid);
    gauss_filter(vc, v, bf_width, grid);
    gauss_filter(wc, w, bf_width, grid);
    gauss_filter(psic, psi, bf_width, grid);

    // quadratic "invariants"
    Ec = - (T)0.5 * average_xy(wc, psic, grid);
    Zc =   (T)0.5 * average_xy(wc, wc, grid);

    // subgrid fluxes
    compute_leonard_vector(taux, tauy, w, u, v, bf_width / grid.dx, (T)0.0, (T)1.0, grid); // bf_width in grid units

    // dwc / dt = div_tau
    null(div_tau, grid.size);
    divergence_vector(div_tau, taux, tauy, grid);
    assign(div_tau, -(T)1.0, div_tau, grid.size);

    Ediss = - average_xy(psic, div_tau, grid);
    Zdiss =   average_xy(wc, div_tau, grid);

    // estimate Cs
    compute_leonard_vector(lx, ly, wc, uc, vc, bf_width / grid.dx, (T)0.0, (T)1.0, grid);
    assign(_taux, (T)1.0, taux, -(T)1.0, lx, grid.size);
    assign(_tauy, (T)1.0, tauy, -(T)1.0, ly, grid.size);
    nabla(sx, sy, wc, grid);

    // lap model
    lap_smagorinsky_model(tx, ty, sx, sy, uc, vc, bf_width, grid);
    Cs_lap_Zdiss = pow(Cs_estimate_Zdiss(taux, tauy, tx, ty, sx, sy, grid), (T)0.5);
    Cs_lap_MSE = pow(Cs_estimate_MSE(taux, tauy, tx, ty, grid), (T)0.5);

    // bilap model
    bilap_smagorinsky_model(tx, ty, sx, sy, uc, vc, bf_width, grid);
    Cs_bilap_Zdiss = pow(Cs_estimate_Zdiss(taux, tauy, tx, ty, sx, sy, grid), (T)0.25);
    Cs_bilap_MSE = pow(Cs_estimate_MSE(taux, tauy, tx, ty, grid), (T)0.25);

    // ssm_bilap model
    Cs_ssm_bilap_Zdiss = pow(Cs_estimate_Zdiss(_taux, _tauy, tx, ty, sx, sy, grid), (T)0.25);
    Cs_ssm_bilap_MSE = pow(Cs_estimate_MSE(_taux, _tauy, tx, ty, grid), (T)0.25);

    // ssm_bilap_keb model
    T _Csim_back = (T)1.0;
    backscatter_ssm(bx, by, wc, uc, vc, (T*)NULL, bf_width / grid.dx, (T)0.0, -(T)1.0, _Csim_back, grid);
    //  ----- warning. tx,ty assumed to be defined ------- //
    nabla(px, py, psic, grid);
    Cs_Cr_estimate_ZEdiss(Cs_ssm_bilap_keb_ZEdiss, Cr_ZEdiss, _taux, _tauy, tx, ty, bx, by, sx, sy, px, py, grid);
    Cs_Cr_estimate_MSE(Cs_ssm_bilap_keb_MSE, Cr_MSE, _taux, _tauy, tx, ty, bx, by, grid);

    // Dynamic model calls
    dyn_model_lap.allocate_memory(grid); if (grid.mpi_com.rank == 0) get_memory_size("Memory size in a priori:");
    dyn_model_lap.update_viscosity(wc, uc, vc, psic, (T)0.0, false, (T)0.0, grid);
    dyn_model_lap.statistics(psic, wc, uc, vc, (T)0.0, grid);
    Cs_lap_dyn = dyn_model_lap.Cs;
    null(div_model, grid.size);
    dyn_model_lap.apply(div_model, wc, uc, vc, grid);
    assign(div_model, -(T)1.0, div_model, grid.size);
    MSE_lap = MSE_error(div_tau, div_model, grid);
    dyn_model_lap.clear();

    dyn_model_bilap.allocate_memory(grid);
    dyn_model_bilap.update_viscosity(wc, uc, vc, psic, (T)0.0, false, (T)0.0, grid);
    dyn_model_bilap.statistics(psic, wc, uc, vc, (T)0.0, grid);
    Cs_bilap_dyn = dyn_model_bilap.Cs;
    null(div_model, grid.size);
    dyn_model_bilap.apply(div_model, wc, uc, vc, grid);
    assign(div_model, -(T)1.0, div_model, grid.size);
    MSE_bilap = MSE_error(div_tau, div_model, grid);
    dyn_model_bilap.clear();

    dyn_model_ssm_bilap.allocate_memory(grid);
    dyn_model_ssm_bilap.update_viscosity(wc, uc, vc, psic, (T)0.0, false, (T)0.0, grid);
    dyn_model_ssm_bilap.statistics(psic, wc, uc, vc, (T)0.0, grid);
    Cs_ssm_bilap_dyn = dyn_model_ssm_bilap.Cs;
    null(div_model, grid.size);
    dyn_model_ssm_bilap.apply(div_model, wc, uc, vc, grid);
    assign(div_model, -(T)1.0, div_model, grid.size);
    MSE_ssm_bilap = MSE_error(div_tau, div_model, grid);
    dyn_model_ssm_bilap.clear();

    dyn_model_ssm_bilap_keb.allocate_memory(grid);
    dyn_model_ssm_bilap_keb.update_viscosity(wc, uc, vc, psic, (T)0.0, false, (T)0.0, grid);
    dyn_model_ssm_bilap_keb.statistics(psic, wc, uc, vc, (T)0.0, grid);
    Cs_ssm_bilap_keb_dyn = dyn_model_ssm_bilap_keb.Cs;
    Cr_dyn = dyn_model_ssm_bilap_keb.Csim_back;
    null(div_model, grid.size);
    dyn_model_ssm_bilap_keb.apply(div_model, wc, uc, vc, grid);
    assign(div_model, -(T)1.0, div_model, grid.size);
    MSE_ssm_bilap_keb = MSE_error(div_tau, div_model, grid);
    dyn_model_ssm_bilap_keb.clear();
}

template void get_invaraints_fluxes_viscosity(float &Ec, float &Zc, float &Ediss, float &Zdiss,
float &Cs_lap_Zdiss, float &Cs_lap_MSE, float &Cs_bilap_Zdiss, float &Cs_bilap_MSE, 
float &Cs_ssm_bilap_Zdiss, float &Cs_ssm_bilap_MSE,
float &Cs_ssm_bilap_keb_ZEdiss, float &Cs_ssm_bilap_keb_MSE,
float &Cr_ZEdiss, float &Cr_MSE,
float &Cs_lap_dyn, float &Cs_bilap_dyn, float &Cs_ssm_bilap_dyn,
float &Cs_ssm_bilap_keb_dyn, float &Cr_dyn,
float &MSE_lap, float &MSE_bilap, 
float &MSE_ssm_bilap, float &MSE_ssm_bilap_keb,
float* psi, float* w, float* u, float* v, const float bf_width, const uniGrid2d< float >&grid,
dynamic_model< float >& dyn_model_lap, dynamic_model< float >& dyn_model_bilap, 
dynamic_model< float >& dyn_model_ssm_bilap, dynamic_model< float >& dyn_model_ssm_bilap_keb);

template void get_invaraints_fluxes_viscosity(double &Ec, double &Zc, double &Ediss, double &Zdiss,
double &Cs_lap_Zdiss, double &Cs_lap_MSE, double &Cs_bilap_Zdiss, double &Cs_bilap_MSE, 
double &Cs_ssm_bilap_Zdiss, double &Cs_ssm_bilap_MSE,
double &Cs_ssm_bilap_keb_ZEdiss, double &Cs_ssm_bilap_keb_MSE,
double &Cr_ZEdiss, double &Cr_MSE,
double &Cs_lap_dyn, double &Cs_bilap_dyn, double &Cs_ssm_bilap_dyn,
double &Cs_ssm_bilap_keb_dyn, double &Cr_dyn,
double &MSE_lap, double &MSE_bilap, 
double &MSE_ssm_bilap, double &MSE_ssm_bilap_keb,
double* psi, double* w, double* u, double* v, const double bf_width, const uniGrid2d< double >&grid,
dynamic_model< double >& dyn_model_lap, dynamic_model< double >& dyn_model_bilap, 
dynamic_model< double >& dyn_model_ssm_bilap, dynamic_model< double >& dyn_model_ssm_bilap_keb);