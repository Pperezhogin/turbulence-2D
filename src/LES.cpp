#define _CRT_SECURE_NO_DEPRECATE
#include "LES.h"

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
    
    dyn_model.init(viscosity_model, averaging_method, mixed_model, mixed_type, negvisc_backscatter, reynolds_backscatter, 
		adm_model, adm_order, tf_width, bf_width, 
		filter_iterations, leonard_scheme,
		lagrangian_time, dt, grid);

	Reynolds_eq.init(grid);	
	Lagrangian_eq.init(grid);
    
	balance.init((Real)1.0, grid);

	nse_series.set(36);
	nse_series.name_variable(0, "kinetic energy");
	nse_series.name_variable(1, "enstrophy");
    nse_series.name_variable(2, "palinstrophy");
    nse_series.name_variable(3, "kinetic energy viscous dissipation");
	nse_series.name_variable(4, "enstrophy viscous dissipation");
    nse_series.name_variable(5, "Cs2");
    nse_series.name_variable(6, "Cs");
    nse_series.name_variable(7, "eddy visc P dissipation");
    nse_series.name_variable(8, "ssm P dissipation");
    nse_series.name_variable(9, "negvisc P dissipation");
    nse_series.name_variable(10, "model P dissipation");
    nse_series.name_variable(11, "eddy visc Z dissipation");
    nse_series.name_variable(12, "ssm Z dissipation");
    nse_series.name_variable(13, "negvisc Z dissipation");
    nse_series.name_variable(14, "model Z dissipation");
    nse_series.name_variable(15, "eddy visc E dissipation");
    nse_series.name_variable(16, "ssm E dissipation");
    nse_series.name_variable(17, "negvisc E dissipation");
    nse_series.name_variable(18, "model E dissipation");
    nse_series.name_variable(19, "negative Esub percent");
    nse_series.name_variable(20, "mean Esub");
	nse_series.name_variable(21, "LM backscatter percent");
	nse_series.name_variable(22, "LM diss to back");
	nse_series.name_variable(23, "MSE germano");
	nse_series.name_variable(24, "corr germano");
	nse_series.name_variable(25, "dissipation ratio germano");
	nse_series.name_variable(26, "lag time");
	nse_series.name_variable(27, "Decay time of subgrid energy");
	nse_series.name_variable(28, "eddy visc E diss galil");
	nse_series.name_variable(29, "negvisc E diss galil");
	nse_series.name_variable(30, "Csim Reynolds");
	nse_series.name_variable(31, "backscatter rate");
	nse_series.name_variable(32, "Reynolds: SGS KE");
	nse_series.name_variable(33, "Reynolds: total KE");
	nse_series.name_variable(34, "Reynolds: SGS KE production");
	nse_series.name_variable(35, "Reynolds: KE loss");

#ifndef DUMP_CONTINUE
	if (grid.mpi_com.rank == 0)
		nse_series.init(NSE_SEQ_FILE);
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
    
    dyn_model.clear();
	Reynolds_eq.clear();
	Lagrangian_eq.clear();
}

// --------------------------- //
// Init Navier-Stokes equation //
// --------------------------- //
void init_nse_eq()
{  
    #ifdef DYNAMIC_MODEL
		dyn_model.init_lagrangian_eq(w, U, V, Psi, grid);
	#endif
	#ifdef REYNOLDS_EQUATION
	Reynolds_eq.init_with_ZB(w, U, V, sqrt(6.0), grid);
	#endif
	#ifdef LAGRANGIAN_EQUATION
	Lagrangian_eq.init_with_ZB(w, U, V, sqrt(6.0), grid);
	#endif
}

void test_interpolate()
{
	#ifdef A_PRIORI_SMAG
	Real Cs;
	Cs = interpolate_1d(t_Cs, Cs_value, Cs_length, -(Real)1.0);
	printf("Test interpolate: t = -1, Cs = %E\n", Cs);

	Cs = interpolate_1d(t_Cs, Cs_value, Cs_length, (Real)0.00030679);
	printf("Test interpolate: t = 0.00030679, Cs = %E\n", Cs);

	Cs = interpolate_1d(t_Cs, Cs_value, Cs_length, (Real)0.000307);
	printf("Test interpolate: t = 0.000307, Cs = %E\n", Cs);

	Cs = interpolate_1d(t_Cs, Cs_value, Cs_length, (Real)3.5);
	printf("Test interpolate: t = 3.5, Cs = %E\n", Cs);

	Cs = interpolate_1d(t_Cs, Cs_value, Cs_length, (Real)10.0);
	printf("Test interpolate: t = 10.0, Cs = %E\n", Cs);

	Cs = interpolate_1d(t_Cs, Cs_value, Cs_length, (Real)9.9942);
	printf("Test interpolate: t = 9.9942, Cs = %E\n", Cs);

	Cs = interpolate_1d(t_Cs, Cs_value, Cs_length, (Real)9.9941);
	printf("Test interpolate: t = 9.9941, Cs = %E\n", Cs);
	#endif
}

// ------------------------------- //
// Advance Navier-Stokes equation  //
// ------------------------------- //
bool advance_nse_eq_runge_kutta()
{
        double begin_mark = omp_get_wtime();

		Real local_Cs = 0;
		#ifdef A_PRIORI_SMAG
			local_Cs = interpolate_1d(t_Cs, Cs_value, Cs_length, current_time);
		#endif

		#ifdef DYNAMIC_MODEL
		dyn_model.update_viscosity(w, U, V, Psi, dt, set_Smagorinsky_value, local_Cs, grid);
		#endif

		#ifdef SIMPLE_MODEL
		dyn_model.set_simple_model(bilap_smag, (Real)(0.06/36.), false, mixed_ssm, grid);
		#endif

		#ifdef DYNAMIC_MODEL
    	dyn_model.statistics(Psi, w, U, V, dt, grid);
    	#endif

		#ifdef DYNAMIC_MODEL_PAWAR
			dyn_model.Cs2_mean = DSM_Pawar(w, U, V, Pawar_test_width, Pawar_base_width, Pawar_clipping, Pawar_averaging, grid);
			dyn_model.Cs = sqrt(dyn_model.Cs2_mean);
		#endif

		#ifdef REYNOLDS_EQUATION
		Reynolds_eq.diagnostics(Psi, w, U, V, grid);
		Reynolds_eq.RK_init(grid);
		#endif

		#ifdef LAGRANGIAN_EQUATION
		Lagrangian_eq.RK_init(grid);
		#endif
        
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

			#ifdef MEAN_FLOW
				w_mean_flow(wim, w_rk, Umean, grid);
			#endif

            #ifdef DIFFUSION
            w_diffusion(wim, w_rk, (Real)1.0 / c_Reynolds, grid);
            #endif

			#if defined(DYNAMIC_MODEL) || defined(SIMPLE_MODEL)
			//dyn_model.apply(wim, w, U, V, grid); // Euler method
			dyn_model.apply(wim, w_rk, U_rk, V_rk, grid); // RK method
			#endif

			#ifdef AD_FILTERING_ON
				dyn_model.AD_filter(wim, w_rk, dt, 7, (Real)0.1, grid);
			#endif

			#ifdef DYNAMIC_MODEL_PAWAR
			Real mxx[grid.size], myy[grid.size], mxy[grid.size];
			lap_UV_smagorinsky_model(mxx, mxy, myy, U_rk, V_rk, dyn_model.Cs * grid.dx * Pawar_base_width, grid);
			Real mx[grid.size], my[grid.size];
			curl_tensor(mx, my, mxx, mxy, myy, grid);
			divergence_vector(wim, mx, my, grid);
			#endif

			#ifdef REYNOLDS_EQUATION
			Reynolds_eq.apply(wim, grid);
			Reynolds_eq.RK_step(w_rk, U_rk, V_rk, q[step] * dt, grid);
			#endif

			#ifdef LAGRANGIAN_EQUATION
			Lagrangian_eq.apply(wim, grid);
			Lagrangian_eq.RK_step(w_rk, U_rk, V_rk, q[step] * dt, grid);
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
            velocity_stag(U_rk, V_rk, Psi_rk, grid);
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
    
	//invariant_level(balance, w, Psi, grid);
    Real lap_w[grid.size];
    laplacian(lap_w, w, grid);
    Real energy, enstrophy, palinstrophy;
    energy = - (Real)0.5 * average_xy(w, Psi, grid);
    enstrophy = (Real)0.5 * average_xy(w, w, grid);
    palinstrophy = - (Real)0.5 * average_xy(lap_w,w,grid);
	
	velocity_abs_max(&u_max, &v_max, U, V, grid);
	if (time_index % ndebug == 0) {
        
        model_error(Uerr, Verr, Psierr, werr,
		U, V, Psi, w, Usol, Vsol, Psisol, wsol, c_Reynolds, current_time, grid);

        u_error_cnorm = mpi_cnorm(Uerr, grid.size);
        v_error_cnorm = mpi_cnorm(Verr, grid.size);
        psi_error_cnorm = mpi_cnorm(Psierr, grid.size);
        w_error_cnorm   = mpi_cnorm(werr  , grid.size);
	}

	if ((grid.mpi_com.rank == 0) && (time_index % ndebug == 0)) {
		printf(" >> U(max) = %.4f, V(max) = %.4f, current CFL = %.4f \n", u_max, v_max, max(u_max * dt / grid.dx, v_max * dt / grid.dy));
		printf("Eddy viscosity model max CFL = %1.4f, mean CFL = %1.4f (2 - stability, 1 - no oscillation) \n", dyn_model.CFL_EVM_max, dyn_model.CFL_EVM_mean);

		int est_sec = (int)(cpu_run_time * ((double)
			((end_time - current_time) / (current_time - begin_time))));

		int est_min = est_sec / 60; est_sec %= 60;
		int est_hrs = est_min / 60; est_min %= 60;

		printf("\t >> time: %.7f [ETA: %i:%i:%i] [IC: %.4f s]\n\n", current_time,
			est_hrs, est_min, est_sec,
			cpu_run_time / time_index);

	}

	nse_series.push(0,   (double)energy);
	nse_series.push(1,   (double)enstrophy);
	nse_series.push(2,   (double)palinstrophy);
	nse_series.push(3, - (double)balance.en_visc);
	nse_series.push(4, - (double)balance.ens_visc);
	nse_series.push(5,   (double)dyn_model.Cs2_mean);
	nse_series.push(6,   (double)dyn_model.Cs);
	nse_series.push(7,   (double)dyn_model.t_P_mean);
	nse_series.push(8,   (double)dyn_model.ssm_P_mean);
	nse_series.push(9,   (double)dyn_model.b_P_mean);
	nse_series.push(10,  (double)dyn_model.model_P_mean);
	nse_series.push(11,  (double)dyn_model.t_Z_mean);
	nse_series.push(12,  (double)dyn_model.ssm_Z_mean);
	nse_series.push(13,  (double)dyn_model.b_Z_mean);
	nse_series.push(14,  (double)dyn_model.model_Z_mean);
	nse_series.push(15,  (double)dyn_model.t_E_mean);
	nse_series.push(16,  (double)dyn_model.ssm_E_mean);
	nse_series.push(17,  (double)dyn_model.b_E_mean);
	nse_series.push(18,  (double)dyn_model.model_E_mean);
	nse_series.push(19,  (double)dyn_model.E_neg_p);
	nse_series.push(20,  (double)dyn_model.Esub_mean);   
	nse_series.push(21,  (double)dyn_model.LM_back_p);
	nse_series.push(22,  (double)dyn_model.LM_diss_to_back);
	nse_series.push(23,  (double)dyn_model.MSE_germano);
	nse_series.push(24,  (double)dyn_model.C_germano);
	nse_series.push(25,  (double)dyn_model.model_diss_to_l_diss_germano);
	nse_series.push(26,  (double)dyn_model.mean_lag_time);
	nse_series.push(27,  (double)dyn_model.Esub_time);
	nse_series.push(28,  (double)dyn_model.t_E_mean_flux);
	nse_series.push(29,  (double)dyn_model.b_E_mean_flux);
	nse_series.push(30,  (double)dyn_model.Csim_back);
	nse_series.push(31,  (double)dyn_model.backscatter_rate);
	nse_series.push(32,  (double)Reynolds_eq.SGS_KE);
	nse_series.push(33,  (double)Reynolds_eq.SGS_KE + (double)energy);
	nse_series.push(34,  (double)Reynolds_eq.SGS_KE_prod);
	nse_series.push(35,  (double)Reynolds_eq.KE_loss);
	nse_series.push_time((double)current_time);

	if (max(u_max, v_max) > (Real)15.0) {
		if (grid.mpi_com.rank == 0) printf("Model VZORVALAS'!\n");
		return false;
	}

	double end_mark = omp_get_wtime();
	cpu_run_time += end_mark - begin_mark;
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
	if (!model_init()) {
		if (grid.mpi_com.rank == 0)
			printf(" >> FAILURE! >> ** model init **\n");

		model_print("FAILURE!: ** model init **\n");

		MPI_Finalize();
		return 0;
	}
	srand (time(NULL) + grid.mpi_com.rank);
	//srand(grid.mpi_com.rank);
	// init conditions: velocity, pressure

	if (grid.mpi_com.rank == 0)
	{
		printf("My initial data index is %i\n", file_index);
	}

	#ifdef A_PRIORI_SMAG
	read_series("/data90t/users/perezhogin/decaying-turbulence/Cs_ssm_bilap_128.nsx", 
	Cs_length, t_Cs, Cs_value, grid);
	//test_interpolate();
	#endif

	init_solution(Usol, Vsol, Psisol, wsol, grid, file_index);
    
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

		// ------------------------------------------------- //
		if (!advance_nse_eq_runge_kutta()) {status = false; break;}
		if (!advance_time()) { status = false; break; }
		// ------------------------------------------------- //

		// ------------------------------------------------- //
		// write output & dump
		// ------------------------------------------------- //
		if (nse_series.length() >= c_seq_max_length) {
			if (grid.mpi_com.rank == 0) nse_series.write(NSE_SEQ_FILE);
			nse_series.reset();
		}

		if (current_time >= print_mark) {       
			
            write_tecplot(OUTPUT_DIR"-w-.plt", print_index,
				w, "w",
				print_xmin, print_xmax,
				print_ymin, print_ymax,
				grid, current_time);
            /*
			write_tecplot(OUTPUT_DIR"-Cs2-.plt", print_index,
				dyn_model.Cs2_local, "w",
				print_xmin, print_xmax,
				print_ymin, print_ymax,
				grid, current_time);
            */
			/*
			write_tecplot(OUTPUT_DIR"-damping_coef-.plt", print_index,
				dyn_model.damping_coef, "w",
				print_xmin, print_xmax,
				print_ymin, print_ymax,
				grid, current_time);

			write_tecplot(OUTPUT_DIR"-phi-.plt", print_index,
				phi, "w",
				print_xmin, print_xmax,
				print_ymin, print_ymax,
				grid, current_time);
			*/

            write_binary_przgn(PSI_BIN_FILE, Psi, grid, print_index);
			write_binary_przgn(OUTPUT_DIR"w.nsx", w, grid, print_index);
			
			// base level models
			#if defined(DYNAMIC_MODEL) || defined(SIMPLE_MODEL)
			write_binary_przgn(OUTPUT_DIR"Cs2.nsx", dyn_model.Cs2_local, grid, print_index);
			write_binary_przgn(OUTPUT_DIR"tx.nsx", dyn_model.tx, grid, print_index);
			write_binary_przgn(OUTPUT_DIR"ty.nsx", dyn_model.ty, grid, print_index);
			write_binary_przgn(OUTPUT_DIR"ssmx.nsx", dyn_model.ssmx, grid, print_index);
			write_binary_przgn(OUTPUT_DIR"ssmy.nsx", dyn_model.ssmy, grid, print_index);
			write_binary_przgn(OUTPUT_DIR"bx.nsx", dyn_model.bx, grid, print_index);
			write_binary_przgn(OUTPUT_DIR"by.nsx", dyn_model.by, grid, print_index);
			#endif

			// germano identity
			#ifdef DYNAMIC_MODEL
			// numerator denominator
            if (dyn_model.averaging_method == averaging_global)
            {
                write_binary_przgn(OUTPUT_DIR"LM.nsx", dyn_model.lm, grid, print_index);
			    write_binary_przgn(OUTPUT_DIR"MM.nsx", dyn_model.mm, grid, print_index);    
            } else {
			    write_binary_przgn(OUTPUT_DIR"LM.nsx", dyn_model.LM, grid, print_index);
			    write_binary_przgn(OUTPUT_DIR"MM.nsx", dyn_model.MM, grid, print_index);
            }
			#endif

			#ifdef REYNOLDS_EQUATION
			write_binary_przgn(OUTPUT_DIR"tau_xy.nsx", Reynolds_eq.tau_xy, grid, print_index);
			write_binary_przgn(OUTPUT_DIR"tau_dd.nsx", Reynolds_eq.tau_dd, grid, print_index);
			write_binary_przgn(OUTPUT_DIR"tau_tr.nsx", Reynolds_eq.tau_tr, grid, print_index);
			#endif

			#ifdef LAGRANGIAN_EQUATION
			write_binary_przgn(OUTPUT_DIR"fx.nsx", Lagrangian_eq.fx, grid, print_index);
			write_binary_przgn(OUTPUT_DIR"fy.nsx", Lagrangian_eq.fy, grid, print_index);
			#endif

			print_mark += print_dt;
			print_index++;
		}
		
		if (current_time >= dump_mark) {

			const int cpu_stamp_size = 4;
			double cpu_stamp[cpu_stamp_size];
			cpu_stamp[0] = cpu_run_time;
			cpu_stamp[1] = cpu_nse_eq_time;
			cpu_stamp[2] = cpu_pois_time;
			cpu_stamp[3] = grid.mpi_com.cpu_time_exch;

			write_binary_stamp(DUMP_NSE_STAMP_FILE, dump_index,
				cpu_stamp, cpu_stamp_size,
				grid, current_time, time_index);

			if (grid.mpi_com.rank == 0) {
				nse_series.write(NSE_SEQ_FILE);
				copy_file(NSE_SEQ_FILE, DUMP_NSE_SEQ_FILE, dump_index);
			}
			nse_series.reset();

			write_binary(DUMP_VELOCITY_FILE, dump_index,
				U, V, "U", "V",
				grid, current_time);
			write_binary(DUMP_VORTEX_FILE, dump_index,
				w, "w",
				grid, current_time);
			write_binary(DUMP_PSI_FILE, dump_index,
				Psi, "Psi",
				grid, current_time);

			dump_mark += dump_dt;
			dump_index++;
		}

	}

	if (grid.mpi_com.rank == 0) {
		nse_series.write(NSE_SEQ_FILE);
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

			#ifdef DYNAMIC_MODEL
			fprintf(ptr, " \t Dynamic model is turned on \n");
			fprintf(ptr, " \t viscosity model = 0, 1, 2, 3, 4 = lap, bilap, lap leith, lap smag, bilap smag \n");
			fprintf(ptr, " \t averaging method = 0, 1, 2 = averaging global, clipping, lagrangian \n");
			fprintf(ptr, " \t viscosity model = %i \n", dyn_model.viscosity_model);
			fprintf(ptr, " \t averaging method = %i \n", dyn_model.averaging_method);
			fprintf(ptr, " \t mixed model = %i \n", dyn_model.mixed_model);
			fprintf(ptr, " \t tf_width = %E \n", dyn_model.tf_width);
			fprintf(ptr, " \t bf_width = %E \n", dyn_model.bf_width);
			fprintf(ptr, " \t filter iterations = %i \n", dyn_model.filter_iterations);
			fprintf(ptr, " \t lagrangian time = %E \n", dyn_model.lagrangian_time);
			fprintf(ptr, " \t leonard scheme = %i \n", dyn_model.leonard_scheme);
			#else
			fprintf(ptr, " \t Dynamic model is turned off \n");
			#endif
			#ifdef DYNAMIC_MODEL_PAWAR
			fprintf(ptr, " \t Dynamic model in Pawar form is used \n");
			fprintf(ptr, " \t test width = %E \n", Pawar_test_width);
			fprintf(ptr, " \t base width = %E \n", Pawar_base_width);
			fprintf(ptr, " \t clipping = %i \n", int(Pawar_clipping));
			fprintf(ptr, " \t averaging = %i \n", Pawar_averaging);
			#endif

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
