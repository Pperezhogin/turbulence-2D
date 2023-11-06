#define _CRT_SECURE_NO_DEPRECATE
#include "test2.h"

bool model_init()
{
	allocate(&U, &V, grid.size);
	allocate(&Psi, grid.size);
	allocate(&w, grid.size);
	allocate(&rhs, grid.size);

	allocate(&Psi_rk, grid.size);
	allocate(&w_rk, grid.size);

	allocate(&Usol, &Vsol, grid.size);
	allocate(&Uerr, &Verr, grid.size);
    allocate(&Psierr, &werr, grid.size);
	
	allocate(&Psisol, grid.size);
	allocate(&wsol, grid.size);
	
	allocate(&wim, grid.size);
    
    dyn_model.init(viscosity_model, averaging_method, mixed_model, mixed_type, negvisc_backscatter, reynolds_backscatter, 
	adm_model, adm_order, tf_width, bf_width, filter_iterations, Leonard_PV_Z_scheme, lagrangian_time, dt, grid);

	Reynolds_eq.init(grid);
	return true;
}

void model_clear()
{
	deallocate(U, V);
	deallocate(Psi);
	deallocate(w);
	deallocate(rhs);
	
	deallocate(Psi_rk);
	deallocate(w_rk);
	
	deallocate(Usol, Vsol);
	deallocate(Uerr, Verr);
	deallocate(Psierr, werr);
	
	deallocate(Psisol);
	deallocate(wsol);
	
	deallocate(wim);
   
    dyn_model.clear();
	Reynolds_eq.clear();
}

// --------------------------- //
// Init Navier-Stokes equation //
// --------------------------- //
void init_nse_eq()
{  
    dyn_model.init_lagrangian_eq(w, U, V, Psi, grid);
}

// ------------------------------- //
// Advance Navier-Stokes equation  //
// ------------------------------- //
bool advance_nse_eq_runge_kutta(bool dyn_model_on, bool Reynolds_eq_on)
{
        double begin_mark = omp_get_wtime();
        
		if (dyn_model_on) dyn_model.update_viscosity(w, U, V, Psi, dt, false, (Real)0.0, grid);
		if (dyn_model_on) dyn_model.statistics(Psi, w, U, V, dt, grid);
		if (Reynolds_eq_on) Reynolds_eq.RK_init(grid);

        memcpy(Psi_rk, Psi, grid.size * sizeof(Real));
        memcpy(w_rk, w, grid.size * sizeof(Real));
        
        Real q[3];
        int max_step;
        
        // 3 step RK parameters
        q[0] = (Real)1.0 / (Real)3.0;
        q[1] = (Real)1.0 / (Real)2.0;
        q[2] = (Real)1.0;
        max_step = 3;
        
        for (int step = 0; step < max_step; step++)
        {
            
            null(wim, grid.size);
                
			J_EZ(wim, w_rk, Psi_rk, grid);

            #ifdef DIFFUSION
            w_diffusion(wim, w_rk, (Real)1.0 / c_Reynolds, grid);
            #endif
            
            if (dyn_model_on) dyn_model.apply(wim, w_rk, U, V, grid);

			if (Reynolds_eq_on) {
				Reynolds_eq.apply(wim, grid);
				Reynolds_eq.RK_step(w_rk, U, V, q[step] * dt, grid);
			}
            
            assign(w_rk, (Real)1.0, w, q[step] * dt, wim, grid.size);
            w_bc(w_rk, grid);

            double pois_begin_mark = omp_get_wtime();
            poisson_status = poisson_fft(Psi_rk, w_rk, fft_data, grid);      
            cpu_pois_time += omp_get_wtime() - pois_begin_mark;
            
            psi_bc(Psi_rk, grid);
            velocity_stag(U, V, Psi_rk, grid); // velocity is diagnostic variable
            velocity_bc(U, V, grid);
        }
        memcpy(Psi, Psi_rk, grid.size * sizeof(Real));
        memcpy(w, w_rk, grid.size * sizeof(Real));        
	
        double end_mark = omp_get_wtime();
        cpu_nse_eq_time += end_mark - begin_mark;
        cpu_run_time += end_mark - begin_mark;

        return true;
}

// ------------------------------- //
// Time Advancement processing     //
// ------------------------------- //
bool advance_time()
{
	double begin_mark = omp_get_wtime();

	// advance time  //
	current_time += dt; time_index++;
	// --------------------------- //

	double end_mark = omp_get_wtime();
	cpu_run_time += end_mark - begin_mark;
	return true;
}

Real launch_model(bool dyn_model_on, bool Reynolds_eq_on = false)
{
	model_setup();
	model_init();
	unsigned seed = time(NULL);
    MPI_Bcast(&seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    seed += grid.mpi_com.rank;
	srand(seed);

	init_solution(Usol, Vsol, Psisol, wsol, grid);
    
	init_velocity(U, V, Usol, Vsol, grid);
	init_psi(Psi, Psisol, grid);
	init_w(w, wsol, grid);

	velocity_bc(U, V, grid);
	psi_bc(Psi, grid);
	w_bc(w,grid);
	
	// init nse:
	init_nse_eq();
    
	bool status = true;
	while (current_time < end_time) {
		if (!advance_nse_eq_runge_kutta(dyn_model_on, Reynolds_eq_on)) {status = false; break;}
		if (!advance_time()) { status = false; break; }
	}

    model_error(Uerr, Verr, Psierr, werr,
		U, V, Psi, w, Usol, Vsol, Psisol, wsol, c_Reynolds, current_time, grid);

	Real u_error_cnorm  , v_error_cnorm;
	Real psi_error_cnorm, w_error_cnorm;
	Real abs_error;

	u_error_cnorm = mpi_cnorm(Uerr, grid.size);
	v_error_cnorm = mpi_cnorm(Verr, grid.size);
	psi_error_cnorm = mpi_cnorm(Psierr, grid.size);
	w_error_cnorm   = mpi_cnorm(werr  , grid.size);

	// sum or relative C-errors 
	abs_error = u_error_cnorm + v_error_cnorm + psi_error_cnorm + w_error_cnorm / (Real)2.0;
	
	model_clear();

	return abs_error;
}

void launch_all_models()
{
	viscosity_model = lap;
	abs_error = launch_model(true);
	if (mpi_rank == 0) printf("Lap        = %.16e \n", abs_error);
	assert(abs_error < max_abs_error);

	viscosity_model = lap_smag;
	abs_error = launch_model(true);
	if (mpi_rank == 0) printf("Lap-Smag   = %.16e \n", abs_error);
	assert(abs_error < max_abs_error);

	viscosity_model = lap_leith;
	abs_error = launch_model(true);
	if (mpi_rank == 0) printf("Lap-Leith  = %.16e \n", abs_error);
	assert(abs_error < max_abs_error);

	viscosity_model = bilap;
	abs_error = launch_model(true);
	if (mpi_rank == 0) printf("Bilap      = %.16e \n", abs_error);
	assert(abs_error < max_abs_error);

	viscosity_model = bilap_smag;
	abs_error = launch_model(true);
	if (mpi_rank == 0) printf("Bilap-Smag = %.16e \n", abs_error);
	assert(abs_error < max_abs_error);

	/*
	viscosity_model = lap_w_smag;
	abs_error = launch_model(true);
	if (mpi_rank == 0) printf("Lap-w-Smag   = %.16e \n", abs_error);
	assert(abs_error < max_abs_error);

	viscosity_model = bilap_w_smag;
	abs_error = launch_model(true);
	if (mpi_rank == 0) printf("Bilap-w-Smag   = %.16e \n", abs_error);
	assert(abs_error < max_abs_error);
	*/

	if (mpi_rank == 0) printf("\n");
}

int main(int argc, char** argv)
{
	omp_set_num_threads(OPENMP_CORES);
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

	grid.set(
		(Real) 0.0, (Real) 0.0, domain_length, domain_width,
		domain_nx, domain_ny, domain_gcx, domain_gcy, 1);

	if (grid.mpi_com.rank == 0) printf("mpi communicator size = %i \n", grid.mpi_com.size);

	fft_data.init(grid);

	// DNS model
	abs_error = launch_model(false);
	if (mpi_rank == 0) printf("DNS        = %.16e \n", abs_error);
	assert(abs_error < max_abs_error);

	if (mpi_rank == 0) printf("\n");

	if (mpi_rank == 0) printf("Reynolds equation model: \n");
	abs_error = launch_model(false, true);
	if (mpi_rank == 0) printf("LES        = %.16e \n", abs_error);
	assert(abs_error < max_abs_error);
	if (mpi_rank == 0) printf("\n");

	// ----------- Dynamic models global ----------- //
	averaging_method = averaging_global;
	mixed_model = false;
	reynolds_backscatter = false;
    negvisc_backscatter = false;
	adm_model = false;
	
	if (mpi_rank == 0) printf("Dynamic global models: \n");
	launch_all_models();

	// ----------- Dynamic models global ----------- //
	averaging_method = Maulik2017;
	mixed_model = false;
	reynolds_backscatter = false;
    negvisc_backscatter = false;
	adm_model = false;
	
	if (mpi_rank == 0) printf("Dynamic global models, Maulik2017: \n");
	launch_all_models();

	// ----------- Mixed models global ----------- //
	averaging_method = averaging_global;
	mixed_model = true;
	mixed_type = mixed_ssm;
	reynolds_backscatter = false;
    negvisc_backscatter = false;
	adm_model = false;
	
	if (mpi_rank == 0) printf("Mixed global models: \n");
	launch_all_models();

	// ----------- Mixed models global ----------- //
	averaging_method = averaging_global;
	mixed_model = true;
	mixed_type = mixed_ssm;
	reynolds_backscatter = true;
    negvisc_backscatter = false;
	adm_model = false;
	
	if (mpi_rank == 0) printf("Mixed global models reynolds: \n");
	launch_all_models();

		// ----------- Mixed models global ----------- //
	averaging_method = dyn2;
	mixed_model = true;
	mixed_type = mixed_ssm;
	reynolds_backscatter = true;
    negvisc_backscatter = false;
	adm_model = false;
	
	if (mpi_rank == 0) printf("Mixed dyn2 models reynolds: \n");
	launch_all_models();

	// ----------- Mixed models global ----------- //
	averaging_method = averaging_global;
	mixed_model = true;
	mixed_type = mixed_ngm;
	reynolds_backscatter = false;
    negvisc_backscatter = false;
	adm_model = false;
	
	if (mpi_rank == 0) printf("Mixed NGM global models: \n");
	launch_all_models();

	// ----------- Mixed models global ----------- //
	averaging_method = averaging_global;
	mixed_model = true;
	mixed_type = mixed_ssm;
	reynolds_backscatter = false;
    negvisc_backscatter = false;
	adm_model = true;
	
	if (mpi_rank == 0) printf("Mixed ADM global models: \n");
	launch_all_models();

	// ----------- Mixed models local ----------- //
	averaging_method = clipping;
	mixed_model = true;
	mixed_type = mixed_ssm;
	filter_iterations = 1;
	reynolds_backscatter = false;
    negvisc_backscatter = false;
	adm_model = false;

	if (mpi_rank == 0) printf("Mixed clipped models (1 filter): \n");
	launch_all_models();

	// ----------- Mixed models local ----------- //
	averaging_method = clipping;
	mixed_model = true;
	mixed_type = mixed_ssm;
	filter_iterations = 1;
	reynolds_backscatter = true;
    negvisc_backscatter = false;
	adm_model = false;

	if (mpi_rank == 0) printf("Mixed clipped models reynolds (1 filter): \n");
	launch_all_models();

	
		// ----------- Mixed models local ----------- //
	averaging_method = clipping;
	mixed_model = true;
	mixed_type = mixed_ngm;
	filter_iterations = 1;
	reynolds_backscatter = false;
    negvisc_backscatter = false;
	adm_model = false;

	if (mpi_rank == 0) printf("Mixed NGM clipped models (1 filter): \n");
	launch_all_models();

	// ----------- Mixed models local ----------- //
	averaging_method = clipping;
	mixed_model = true;
	mixed_type = mixed_ssm;
	filter_iterations = 1;
	reynolds_backscatter = false;
    negvisc_backscatter = false;
	adm_model = true;

	if (mpi_rank == 0) printf("Mixed ADM clipped models (1 filter): \n");
	launch_all_models();
	
	
	// ----------- Mixed models lagrangian ----------- //
	averaging_method = lagrangian;
	lagrangian_time = (Real)2.0;
	mixed_model = true;
	mixed_type = mixed_ssm;
	reynolds_backscatter = false;
    negvisc_backscatter = false;
	adm_model = false;
	
	if (mpi_rank == 0) printf("Mixed Lagrangian models (T = 2*|S|^-1): \n");
	launch_all_models();

		// ----------- Mixed models lagrangian ----------- //
	averaging_method = lagrangian;
	lagrangian_time = (Real)2.0;
	mixed_model = true;
	mixed_type = mixed_ssm;
	reynolds_backscatter = true;
    negvisc_backscatter = false;
	adm_model = false;
	
	if (mpi_rank == 0) printf("Mixed Lagrangian models reynolds (T = 2*|S|^-1): \n");
	launch_all_models();

	// ----------- Mixed models lagrangian ----------- //
	averaging_method = lagrangian;
	lagrangian_time = (Real)2.0;
	mixed_model = true;
	mixed_type = mixed_ngm;
	reynolds_backscatter = false;
    negvisc_backscatter = false;
	adm_model = false;
	
	if (mpi_rank == 0) printf("Mixed NGM Lagrangian models (T = 2*|S|^-1): \n");
	launch_all_models();

	// ----------- Mixed models lagrangian ----------- //
	averaging_method = lagrangian;
	lagrangian_time = (Real)2.0;
	mixed_model = true;
	mixed_type = mixed_ssm;
	reynolds_backscatter = false;
    negvisc_backscatter = false;
	adm_model = true;
	
	if (mpi_rank == 0) printf("Mixed ADM Lagrangian models (T = 2*|S|^-1): \n");
	launch_all_models();

	// ----------- Mixed models lagrangian ----------- //
	averaging_method = lagrangian;
	lagrangian_time = (Real)10.0;
	mixed_model = true;
	mixed_type = mixed_ssm;
	reynolds_backscatter = false;
    negvisc_backscatter = false;
	
	if (mpi_rank == 0) printf("Mixed Lagrangian models (T = 10*|S|^-1): \n");
	launch_all_models();
    
    // ----------- Mixed models lagrangian backscatter ----------- //
	averaging_method = lagrangian;
	lagrangian_time = (Real)2.0;
	mixed_model = true;
	mixed_type = mixed_ssm;
	reynolds_backscatter = false;
    negvisc_backscatter = true;
	
	if (mpi_rank == 0) printf("Mixed Lagrangian models Wth backscatter: \n");
	viscosity_model = bilap_smag;
	abs_error = launch_model(true);
	if (mpi_rank == 0) printf("Bilap-Smag = %.16e \n", abs_error);
	assert(abs_error < max_abs_error);

	mpiCom2d::clear();
	// delete spatial communicators by hand //
	MPI_Comm_free(&grid.mpi_com.comm_x);
	MPI_Comm_free(&grid.mpi_com.comm_y);

	fft_data.clear();

	MPI_Finalize();
}
