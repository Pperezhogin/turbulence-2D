#define _CRT_SECURE_NO_DEPRECATE
#include "DNS.h"

void init_4models(dynamic_model< Real >& dyn_lap, dynamic_model< Real >& dyn_bilap,
dynamic_model< Real >& dyn_ssm_bilap, dynamic_model< Real >& dyn_ssm_bilap_keb, 
timeSeries &series, int npoints)
{
	series.set(23);
	series.name_variable(0, "Ec");
	series.name_variable(1, "Zc");
	series.name_variable(2, "Ediss");
	series.name_variable(3, "Zdiss");
	series.name_variable(4, "Cs lap Zdiss");
	series.name_variable(5, "Cs lap MSE");
	series.name_variable(6, "Cs lap dyn");
	series.name_variable(7, "Cs bilap Zdiss");
	series.name_variable(8, "Cs bilap MSE");
	series.name_variable(9, "Cs bilap dyn");
	series.name_variable(10, "Cs ssm_bilap Zdiss");
	series.name_variable(11, "Cs ssm_bilap MSE");
	series.name_variable(12, "Cs ssm_bilap dyn");
	series.name_variable(13, "Cs ssm_bilap_keb ZEdiss");
	series.name_variable(14, "Cs ssm_bilap_keb MSE");
	series.name_variable(15, "Cs ssm_bilap_keb dyn");
	series.name_variable(16, "Cr ssm_bilap_keb ZEdiss");
	series.name_variable(17, "Cr ssm_bilap_keb MSE");
	series.name_variable(18, "Cr ssm_bilap_keb dyn");
	series.name_variable(19, "MSE lap");
	series.name_variable(20, "MSE bilap");
	series.name_variable(21, "MSE ssm_bilap");
	series.name_variable(22, "MSE ssm_bilap_keb");

	Real filter_width;
	
	filter_width = 2. * M_PI / Real(npoints) * sqrt(6.0) / grid.dx;

	dyn_lap.init(lap_smag, averaging_global, false, mixed_ssm, false, false, 
	false, 2, filter_width, filter_width, 
		1, Leonard_PV_Z_scheme, (Real)1.0, dt, grid, false);

	dyn_bilap.init(bilap_smag, averaging_global, false, mixed_ssm, false, false, 
	false, 2, filter_width, filter_width, 
		1, Leonard_PV_Z_scheme, (Real)1.0, dt, grid, false);

	dyn_ssm_bilap.init(bilap_smag, averaging_global, true, mixed_ssm, false, false, 
	false, 2, filter_width, filter_width, 
		1, Leonard_PV_Z_scheme, (Real)1.0, dt, grid, false);

	dyn_ssm_bilap_keb.init(bilap_smag, averaging_global, true, mixed_ssm, false, true, 
	false, 2, filter_width, filter_width, 
		1, Leonard_PV_Z_scheme, (Real)1.0, dt, grid, false);

}

void analyze_4models(dynamic_model< Real >& dyn_lap, dynamic_model< Real >& dyn_bilap,
dynamic_model< Real >& dyn_ssm_bilap, dynamic_model< Real >& dyn_ssm_bilap_keb, 
timeSeries &series, int npoints)
{
	Real Ec, Zc, Ediss, Zdiss;
	Real Cs_lap_Zdiss, Cs_lap_MSE, Cs_bilap_Zdiss, Cs_bilap_MSE;
	Real Cs_ssm_bilap_Zdiss, Cs_ssm_bilap_MSE;
	Real Cs_ssm_bilap_keb_ZEdiss, Cs_ssm_bilap_keb_MSE;
	Real Cr_ZEdiss, Cr_MSE, Cr_dyn;
    Real Cs_lap_dyn, Cs_bilap_dyn;
	Real Cs_ssm_bilap_dyn, Cs_ssm_bilap_keb_dyn;
	Real MSE_lap, MSE_bilap;
	Real MSE_ssm_bilap, MSE_ssm_bilap_keb;
	Real filter_width;

	filter_width = 2. * M_PI / Real(npoints) * sqrt(6.0);

	get_invaraints_fluxes_viscosity(Ec, Zc, Ediss, Zdiss,
	Cs_lap_Zdiss, Cs_lap_MSE, Cs_bilap_Zdiss, Cs_bilap_MSE, 
	Cs_ssm_bilap_Zdiss, Cs_ssm_bilap_MSE,
	Cs_ssm_bilap_keb_ZEdiss, Cs_ssm_bilap_keb_MSE,
	Cr_ZEdiss, Cr_MSE,
	Cs_lap_dyn, Cs_bilap_dyn, Cs_ssm_bilap_dyn,
	Cs_ssm_bilap_keb_dyn, Cr_dyn,
	MSE_lap, MSE_bilap, 
	MSE_ssm_bilap, MSE_ssm_bilap_keb,
	Psi, w, U, V, filter_width, grid,
	dyn_lap, dyn_bilap, dyn_ssm_bilap, dyn_ssm_bilap_keb);

	series.push(0, Ec);
	series.push(1, Zc);
	series.push(2, Ediss);
	series.push(3, Zdiss);
	series.push(4, Cs_lap_Zdiss);
	series.push(5, Cs_lap_MSE);
	series.push(6, Cs_lap_dyn);
	series.push(7, Cs_bilap_Zdiss);
	series.push(8, Cs_bilap_MSE);
	series.push(9, Cs_bilap_dyn);
	series.push(10, Cs_ssm_bilap_Zdiss);
	series.push(11, Cs_ssm_bilap_MSE);
	series.push(12, Cs_ssm_bilap_dyn);
	series.push(13, Cs_ssm_bilap_keb_ZEdiss);
	series.push(14, Cs_ssm_bilap_keb_MSE);
	series.push(15, Cs_ssm_bilap_keb_dyn);
	series.push(16, Cr_ZEdiss);
	series.push(17, Cr_MSE);
	series.push(18, Cr_dyn);
	series.push(19, MSE_lap);
	series.push(20, MSE_bilap);
	series.push(21, MSE_ssm_bilap);
	series.push(22, MSE_ssm_bilap_keb);
	series.push_time((double)current_time);	
}

bool model_init()
{
	allocate(&U, &V, grid.size);
	allocate(&Psi, grid.size);
	allocate(&w, grid.size);
	allocate(&rhs, grid.size);

	allocate(&rhs_visc, grid.size);

	allocate(&Psi_rk, grid.size);
	allocate(&w_rk, grid.size);
	allocate(&U_rk, &V_rk, grid.size);

	allocate(&Usol, &Vsol, grid.size);
	allocate(&Uerr, &Verr, grid.size);
    allocate(&Psierr, &werr, grid.size);
	
	allocate(&Psisol, grid.size);
	allocate(&wsol, grid.size);
	
	allocate(&wim, grid.size);

	allocate(&phi, grid.size);
	allocate(&phi_rk, grid.size);
	allocate(&phim, grid.size);
	
	// poisson memory allocation //
#ifndef POISSON_FFT
    allocate(&memory, 4 * grid.size);
	// poisson multigrid allocation //
	if (pois_ngrid == 0) {

		pois_ngrid = 1;
		int np = (grid.nx - 2 * grid.gcx) * (grid.ny - 2 * grid.gcy);
		while (np > pois_mg_min) { pois_ngrid++; np /= 4; }
	}
	mg_data.init(grid, pois_ngrid);
#else
	fft_data.init(grid);
#endif
    
	balance.init((Real)1.0, grid);

	nse_series.set(4);
	nse_series.name_variable(0, "kinetic energy");
	nse_series.name_variable(1, "enstrophy");
    nse_series.name_variable(2, "kinetic energy viscous dissipation");
	nse_series.name_variable(3, "enstrophy viscous dissipation");

	init_4models(dyn_model_lap_128, dyn_model_bilap_128,
	dyn_model_ssm_bilap_128, dyn_model_ssm_bilap_keb_128, 
	series_128, 128);

	init_4models(dyn_model_lap_256, dyn_model_bilap_256,
	dyn_model_ssm_bilap_256, dyn_model_ssm_bilap_keb_256, 
	series_256, 256);

	init_4models(dyn_model_lap_512, dyn_model_bilap_512,
	dyn_model_ssm_bilap_512, dyn_model_ssm_bilap_keb_512, 
	series_512, 512);
	
#ifndef DUMP_CONTINUE
	if (grid.mpi_com.rank == 0) {
		nse_series.init(OUTPUT_DIR"nse.dsq");
		series_128.init(OUTPUT_DIR"series_128.dsq");
		series_256.init(OUTPUT_DIR"series_256.dsq");
		series_512.init(OUTPUT_DIR"series_512.dsq");
	}
#else
	if (grid.mpi_com.rank == 0) {
		copy_file(DUMP_NSE_SEQ_FILE, dump_continue_mark, NSE_SEQ_FILE);
		nse_series.init_append(NSE_SEQ_FILE);
	}
#endif

	return true;
}

void model_clear()
{
	deallocate(U, V);
	deallocate(Psi);
	deallocate(w);
	deallocate(rhs);

	deallocate(rhs_visc);
	
	deallocate(Psi_rk);
	deallocate(w_rk);
	deallocate(U_rk, V_rk);
	
	deallocate(Usol, Vsol);
	deallocate(Uerr, Verr);
	
	deallocate(Psisol);
	deallocate(wsol);
	
	deallocate(wim);

	deallocate(phi);
	deallocate(phi_rk);
	deallocate(phim);

	mpiCom2d::clear();
	// delete spatial communicators by hand //
	MPI_Comm_free(&grid.mpi_com.comm_x);
	MPI_Comm_free(&grid.mpi_com.comm_y);

#ifndef POISSON_FFT
    deallocate(memory); // poisson memory
	mg_data.clear();
#else
	fft_data.clear();
#endif
    
}

// --------------------------- //
// Init Navier-Stokes equation //
// --------------------------- //
void init_nse_eq()
{  
}

// ------------------------------- //
// Advance Navier-Stokes equation  //
// ------------------------------- //
bool advance_nse_eq_runge_kutta()
{
        double begin_mark = omp_get_wtime();
        
        memcpy(Psi_rk, Psi, grid.size * sizeof(Real));
        memcpy(w_rk, w, grid.size * sizeof(Real));
		memcpy(U_rk, U, grid.size * sizeof(Real));
		memcpy(V_rk, V, grid.size * sizeof(Real));
        
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

            if (time_index % ndebug == 0) {
                check_const(wim, "wim after adv and diff", grid);
            }
            
            assign(w_rk, (Real)1.0, w, q[step] * dt, wim, grid.size);
            w_bc(w_rk, grid);

            double pois_begin_mark = omp_get_wtime();
            poisson_status = poisson_fft(Psi_rk, w_rk, fft_data, grid);      
            cpu_pois_time += omp_get_wtime() - pois_begin_mark;
            
            psi_bc(Psi_rk, grid);
            velocity_stag(U_rk, V_rk, Psi_rk, grid); // velocity is diagnostic variable
            velocity_bc(U_rk, V_rk, grid);
        }
        memcpy(Psi, Psi_rk, grid.size * sizeof(Real));
        memcpy(w, w_rk, grid.size * sizeof(Real));        
		memcpy(U, U_rk, grid.size * sizeof(Real));        
		memcpy(V, V_rk, grid.size * sizeof(Real));        
	
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
    balance.update_status(true, grid);
    null(rhs_visc, grid.size);
	null(balance.wim_p, grid.size);
    w_diffusion(rhs_visc, w, (Real)1.0 / c_Reynolds, grid);
    sources_visc(balance, w, rhs_visc, Psi, grid);
    
	invariant_level(balance, w, Psi, grid);
	
	velocity_abs_max(&u_max, &v_max, U, V, grid);
	if (max(u_max, v_max) > (Real)15.0) {
		if (grid.mpi_com.rank == 0) printf("Model VZORVALAS'!\n");
		return false;
	}
	if (time_index % ndebug == 0) {
        
        model_error(Uerr, Verr, Psierr, werr,
		U, V, Psi, w, Usol, Vsol, Psisol, wsol, c_Reynolds, current_time, grid);

        u_error_cnorm = mpi_cnorm(Uerr, grid.size);
        v_error_cnorm = mpi_cnorm(Verr, grid.size);
        psi_error_cnorm = mpi_cnorm(Psierr, grid.size);
        w_error_cnorm   = mpi_cnorm(werr  , grid.size);
	}

	if ((grid.mpi_com.rank == 0) && (time_index % ndebug == 0)) {
		if (grid.mpi_com.rank == 0)
			get_memory_size("current memory size:");

		printf(" >> U(max) = %.4f, V(max) = %.4f, current CFL = %.4f \n", u_max, v_max, max(u_max * dt / grid.dx, v_max * dt / grid.dy));
		printf("nse-series length %i and series_128 length %i \n", nse_series.length(), series_128.length()); 

		int est_sec = (int)(cpu_run_time * ((double)
			((end_time - current_time) / (current_time - begin_time))));

		int est_min = est_sec / 60; est_sec %= 60;
		int est_hrs = est_min / 60; est_min %= 60;

		printf("\t >> time: %.7f [ETA: %i:%i:%i] [IC: %.4f s]\n\n", current_time,
			est_hrs, est_min, est_sec,
			cpu_run_time / time_index);
		printf("cpu_run_time = %E, a_priori_run_time = %E\n\n", cpu_run_time, a_priori_run_time);

	}

	if ((time_index % (32 * domain_nx / 2048) == 0) || (time_index == 1)) {
		nse_series.push(0,   (double)balance.en);
		nse_series.push(1,   (double)balance.ens);
		nse_series.push(2, - (double)balance.en_visc);
		nse_series.push(3, - (double)balance.ens_visc);
		nse_series.push_time((double)current_time);
	}

	double begin_mark_2 = omp_get_wtime();
	// 1 base filter analysis = 3.6 time steps of the model in run time
	if ((time_index % (128 * domain_nx / 2048) == 0) || (time_index == 1)) {
		if (grid.mpi_com.rank == 0)
			get_memory_size("before a priori memory size:");

		analyze_4models(dyn_model_lap_128, dyn_model_bilap_128,
		dyn_model_ssm_bilap_128, dyn_model_ssm_bilap_keb_128, 
		series_128, 128);

		analyze_4models(dyn_model_lap_256, dyn_model_bilap_256,
		dyn_model_ssm_bilap_256, dyn_model_ssm_bilap_keb_256, 
		series_256, 256);

		analyze_4models(dyn_model_lap_512, dyn_model_bilap_512,
		dyn_model_ssm_bilap_512, dyn_model_ssm_bilap_keb_512, 
		series_512, 512);

		if (grid.mpi_com.rank == 0)
			get_memory_size("after a priori memory size:");
	}
	
	double end_mark = omp_get_wtime();
	cpu_run_time += end_mark - begin_mark;
	a_priori_run_time += end_mark - begin_mark_2;
	return true;
}

int main(int argc, char** argv)
{
	omp_set_num_threads(OPENMP_CORES);

	MPI_Init(&argc, &argv);
	int file_index;
	if (argc > 1) 
		file_index = std::stoi(argv[1]);
	else
		file_index = 1;

	if (!model_setup()) // model parameters setup
	{
		int mpi_rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

		if (mpi_rank == 0)
			printf(" >> FAILURE! >> ** model setup **\n");

		model_print("FAILURE!: ** model setup **\n");

		MPI_Finalize();
		return 0;
	}

	#ifdef START_TIME_0
	if (grid.mpi_com.rank == 0)
	{
		printf("My initial data index is %i\n", file_index);
	}
	#endif

	#ifdef MESH_HAT_FILTER
		if (grid.mpi_com.rank == 0)
			printf("Mesh hat filter is on. Change to spectral \n");
		MPI_Barrier(grid.mpi_com.comm);
		return 0;
	#endif

	if (grid.mpi_com.rank == 0)
		get_memory_size("after model_setup()");

	if (!model_init()) {
		if (grid.mpi_com.rank == 0)
			printf(" >> FAILURE! >> ** model init **\n");

		model_print("FAILURE!: ** model init **\n");

		MPI_Finalize();
		return 0;
	}

	if (grid.mpi_com.rank == 0)
		get_memory_size("after model_init()");
	
	unsigned seed = time(NULL);
    MPI_Bcast(&seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    seed += grid.mpi_com.rank;
	srand(seed);
	for (int i = 0; i < grid.mpi_com.size; i++) {
		int rnk = grid.mpi_com.rank;
		if (rnk == i)
			printf("my id = %i, my seed = %i \n", rnk, seed);
		MPI_Barrier(grid.mpi_com.comm);
	}

	// init conditions: velocity, pressure
	init_solution(Usol, Vsol, Psisol, wsol, grid, file_index);
    
	init_velocity(U, V, Usol, Vsol, grid);
	init_psi(Psi, Psisol, grid);
	init_w(w, wsol, grid);

	velocity_bc(U, V, grid);
	psi_bc(Psi, grid);
	w_bc(w,grid);

	passive_tracer_field(phi, grid);
	
	// init nse:
	init_nse_eq();

	if (grid.mpi_com.rank == 0)
		get_memory_size("after init_nse_eq()");
    
	bool status = true;
	while (current_time < end_time) {

		// ------------------------------------------------- //
		if (!advance_nse_eq_runge_kutta()) {status = false; break;}
		if (!advance_time()) { status = false; break; }
		// ------------------------------------------------- //

		// ------------------------------------------------- //
		// write output & dump
		// ------------------------------------------------- //
		if (nse_series.length() >= c_seq_max_length) {
			if (grid.mpi_com.rank == 0) nse_series.write(OUTPUT_DIR"nse.dsq");
			nse_series.reset();
		}

		if (series_128.length() >= c_seq_max_length) {
			if (grid.mpi_com.rank == 0) series_128.write(OUTPUT_DIR"series_128.dsq");
			series_128.reset();
		}

		if (series_256.length() >= c_seq_max_length) {
			if (grid.mpi_com.rank == 0) series_256.write(OUTPUT_DIR"series_256.dsq");
			series_256.reset();
		}

		if (series_512.length() >= c_seq_max_length) {
			if (grid.mpi_com.rank == 0) series_512.write(OUTPUT_DIR"series_512.dsq");
			series_512.reset();
		}

		if (current_time >= print_mark) {       
            write_binary_przgn(PSI_BIN_FILE, Psi, grid, print_index);

			print_mark += print_dt;
			print_index++;
		}
	}

	if (grid.mpi_com.rank == 0) {
		nse_series.write(OUTPUT_DIR"nse.dsq");
		series_128.write(OUTPUT_DIR"series_128.dsq");
		series_256.write(OUTPUT_DIR"series_256.dsq");
		series_512.write(OUTPUT_DIR"series_512.dsq");
	}
	
	if (status)
		model_print("OK");
	else
		model_print(" FAILURE!: ** advance eq. [nse] **\n");
	
	model_clear();
	MPI_Finalize();
	return 0;
}

bool model_print(const char* msg_status)
{
	int status = 0;
	if (grid.mpi_com.rank == 0) {
		FILE* ptr = fopen(DATA_FILE, "w");
		if (ptr != NULL) {

			fprintf(ptr, " \t x = %.4f, y = %.4f\n",
				grid.mpi_x, grid.mpi_y);
			fprintf(ptr, " \t length = %.4f, width = %.4f\n",
				grid.mpi_length, grid.mpi_width);
			fprintf(ptr, " \t nx = %i, ny = %i, size = %i\n",
				grid.mpi_nx, grid.mpi_ny, grid.mpi_size);
			fprintf(ptr, " \t gcx = %i, gcy = %i\n", grid.gcx, grid.gcy);
			fprintf(ptr, " \t dx = %.7f, dy = %.7f\n\n", grid.dx, grid.dy);

			fprintf(ptr, " - time\n");
			fprintf(ptr, " \t begin = %.4f, end = %.4f\n", begin_time, end_time);
			fprintf(ptr, " \t CFL = %.7f\n", CFL);
			Real realCFL = max(u_max * dt / grid.dx, v_max * dt / grid.dy);
			fprintf(ptr, "\t realCFL = %.7f\n", realCFL);
			fprintf(ptr, " \t dt = %.7f\n\n", dt);

			#ifdef DIFFUSION
			fprintf(ptr, " \t Reynolds number = %.7f\n\n", c_Reynolds);
			#else
			fprintf(ptr, " \t diffusion is turned off \n\n");
			#endif

			
			fprintf(ptr, " \t Dynamic model is turned off \n");

			#ifdef SIMPLE_MODEL
				fprintf(ptr, " \t Non-dynamic biharmonic Smagorinsky with Cs2 = %E \n", Cs2);
			#endif


			fprintf(ptr, " - data type size: %i\n\n", (int) sizeof(Real));

			fprintf(ptr, " - openmp cores: %i [ of %i ]\n\n",
				omp_get_max_threads(), omp_get_num_procs());

			fprintf(ptr, " - mpi communicator: %i [%i - %i]\n\n",
				grid.mpi_com.size,
				grid.mpi_com.size_x, grid.mpi_com.size_y);

			fprintf(ptr, " - cpu run time = %.5f\n", cpu_run_time);
			fprintf(ptr, " \t nse equation = %.5f\n", cpu_nse_eq_time);
			fprintf(ptr, " \t nse poisson solver = %.5f\n", cpu_pois_time);

			fprintf(ptr, " \t mpi exchange = %.5f\n", grid.mpi_com.cpu_time_exch);
			fprintf(ptr, " \t\t mpi -x exchange = %.5f\n", grid.mpi_com.cpu_time_exch_x);
			fprintf(ptr, " \t\t mpi -y exchange = %.5f\n", grid.mpi_com.cpu_time_exch_y);


			status = 1;
			fclose(ptr);
		}
	}

	MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
	return (status == 1);
}
