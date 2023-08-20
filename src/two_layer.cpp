#define _CRT_SECURE_NO_DEPRECATE
#include "two_layer.h"

bool model_init()
{
	allocate(&U1, &V1, grid.size);
	allocate(&Psi1, grid.size);
	allocate(&w1, &q1, grid.size);
	
	allocate(&U2, &V2, grid.size);
	allocate(&Psi2, grid.size);
	allocate(&w2, &q2, grid.size);

	allocate(&rhs, grid.size);

	allocate(&rhs_visc, grid.size);

	allocate(&Psi1_rk, &Psi2_rk, grid.size);
	allocate(&q1_rk, &q2_rk, grid.size);

	allocate(&qim1, &qim2, grid.size);

	allocate(&eddy_rhs_1, &eddy_rhs_2, grid.size);
	allocate(&shear_rhs_1, &shear_rhs_2, grid.size);
	allocate(&fric_rhs_1, &fric_rhs_2, grid.size);
	
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
    
    dyn_model1.init(viscosity_model, averaging_method, mixed_model, mixed_ssm, negvisc_backscatter, reynolds_backscatter, 
	false, 0, tf_width, bf_width, 
		filter_iterations, Leonard_PV_Z_scheme, lagrangian_time, dt, grid);
	dyn_model2.init(viscosity_model, averaging_method, mixed_model, mixed_ssm, negvisc_backscatter, reynolds_backscatter, 
	false, 0, tf_width, bf_width, 
		filter_iterations, Leonard_PV_Z_scheme, lagrangian_time, dt, grid);
    
	balance.init((Real)1.0, grid);

	nse_series.set(20);
	nse_series.name_variable(0, "kinetic energy");
	nse_series.name_variable(1, "potential energy");
	nse_series.name_variable(2, "full energy");
	nse_series.name_variable(3, "potential enstrophy");
	nse_series.name_variable(4, "relative enstrophy");
	nse_series.name_variable(5, "energy dissipation by eddy parameterization");
	nse_series.name_variable(6, "potential enstrophy dissipation by eddy parameterization");
	nse_series.name_variable(7, "energy dissipation by Raleigh friction");
	nse_series.name_variable(8, "energy generation by prescribed vertical shear");
	nse_series.name_variable(9, "heat flux");
	nse_series.name_variable(10, "Cs2 1 layer");
	nse_series.name_variable(11, "Cs2 2 layer");
	nse_series.name_variable(12, "back percent 1 layer");
	nse_series.name_variable(13, "back percent 2 layer");
	nse_series.name_variable(14, "diss to back 1 layer");
	nse_series.name_variable(15, "diss to back 2 layer");
	nse_series.name_variable(16, "MSE germano 1 layer");
	nse_series.name_variable(17, "MSE germano 2 layer");
	nse_series.name_variable(18, "Cback 1 layer");
	nse_series.name_variable(19, "Cback 2 layer");


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
	deallocate(U1, V1);
	deallocate(Psi1);
	deallocate(w1, q1);
	
	deallocate(U2, V2);
	deallocate(Psi2);
	deallocate(w2, q2);

	deallocate(rhs);

	deallocate(rhs_visc);

	deallocate(Psi1_rk, Psi2_rk);
	deallocate(q1_rk, q2_rk);

	deallocate(qim1, qim2);

	deallocate(eddy_rhs_1, eddy_rhs_2);
	deallocate(shear_rhs_1, shear_rhs_2);
	deallocate(fric_rhs_1, fric_rhs_2);

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
    
    dyn_model1.clear();
	dyn_model2.clear();
}

// --------------------------- //
// Init Navier-Stokes equation //
// --------------------------- //
void init_nse_eq()
{  
    #ifdef DYNAMIC_MODEL
	#ifdef Q_PARAMETERIZATION
		dyn_model1.init_lagrangian_eq(q1, U1, V1, Psi1, grid);
		dyn_model2.init_lagrangian_eq(q2, U2, V2, Psi2, grid);
	#else
		dyn_model1.init_lagrangian_eq(w1, U1, V1, Psi1, grid);
		dyn_model2.init_lagrangian_eq(w2, U2, V2, Psi2, grid);
	#endif
	#endif
}

// ------------------------------- //
// Advance Navier-Stokes equation  //
// ------------------------------- //
bool advance_nse_eq_runge_kutta()
{
        double begin_mark = omp_get_wtime();

		#ifdef DYNAMIC_MODEL
		#ifdef Q_PARAMETERIZATION
			dyn_model1.update_viscosity(q1, U1, V1, Psi1, dt, false, (Real)0.0, grid);
			dyn_model2.update_viscosity(q2, U2, V2, Psi2, dt, false, (Real)0.0, grid);
		#else
			dyn_model1.update_viscosity(w1, U1, V1, Psi1, dt, false, (Real)0.0, grid);
			dyn_model2.update_viscosity(w2, U2, V2, Psi2, dt, false, (Real)0.0, grid);
		#endif
		#endif

		#ifdef SIMPLE_MODEL
		dyn_model1.set_simple_model(bilap_smag, Cs2, false, grid);
		dyn_model2.set_simple_model(bilap_smag, Cs2, false, grid);
		#endif

		#if defined(DYNAMIC_MODEL) || defined(SIMPLE_MODEL)
		#ifdef Q_PARAMETERIZATION
    		dyn_model1.statistics(Psi1, q1, U1, V1, dt, grid);
			dyn_model2.statistics(Psi2, q2, U2, V2, dt, grid);
		#else
			dyn_model1.statistics(Psi1, w1, U1, V1, dt, grid);
			dyn_model2.statistics(Psi2, w2, U2, V2, dt, grid);
		#endif
    	#endif
        
        memcpy(Psi1_rk, Psi1, grid.size * sizeof(Real));
        memcpy(q1_rk, q1, grid.size * sizeof(Real));
        memcpy(Psi2_rk, Psi2, grid.size * sizeof(Real));
        memcpy(q2_rk, q2, grid.size * sizeof(Real));

        Real q[3];
        int max_step;
        
        // 3 step RK parameters
        q[0] = (Real)1.0 / (Real)3.0;
        q[1] = (Real)1.0 / (Real)2.0;
        q[2] = (Real)1.0;
        max_step = 3;
        
        for (int step = 0; step < max_step; step++)
        {
            
            null(qim1, grid.size);
			null(qim2, grid.size);
                
            #ifdef J1
                w_J1(qim1, q1_rk, Psi1_rk, grid);
				w_J1(qim2, q2_rk, Psi2_rk, grid);
            #endif
                
            #ifdef J2
                w_J2(qim1, q1_rk, Psi1_rk, grid);
				w_J2(qim2, q2_rk, Psi2_rk, grid);
            #endif
                
            #ifdef J3
                w_J3(qim1, q1_rk, Psi1_rk, grid);
				w_J3(qim2, q2_rk, Psi2_rk, grid);
            #endif
                
            mul(qim1, (Real)1.0 / (Real)NUM, grid.size);
			mul(qim2, (Real)1.0 / (Real)NUM, grid.size);

			#ifdef ADM
				apply_filter_iter(qim1, qim1, 1, grid);
				apply_filter_iter(qim2, qim2, 1, grid);
			#endif

			// diffusion of relative vorticity
            #ifdef DIFFUSION
            w_diffusion(qim1, w1, (Real)1.0 / c_Reynolds, grid);
			w_diffusion(qim2, w2, (Real)1.0 / c_Reynolds, grid);
            #endif

			assign(fric_rhs_2, (Real)1.0, qim2, grid.size);
			w_friction(qim2, w2, Relaigh, grid);
			assign(fric_rhs_2, (Real)1.0, qim2, - (Real)1.0, fric_rhs_2, grid.size);

			#ifdef BETA_PLANE
			w_beta_effect(qim1, Psi1_rk, Beta_effect, grid);
			w_beta_effect(qim2, Psi2_rk, Beta_effect, grid);
			#endif

			assign(shear_rhs_1, (Real)1.0, qim1, grid.size);
			assign(shear_rhs_2, (Real)1.0, qim2, grid.size);
			add_vertical_shear(qim1, qim2, q1_rk, q2_rk, V1, V2, kd, grid);
			assign(shear_rhs_1, (Real)1.0, qim1, - (Real)1.0, shear_rhs_1, grid.size);
			assign(shear_rhs_2, (Real)1.0, qim2, - (Real)1.0, shear_rhs_2, grid.size);

			assign(eddy_rhs_1, (Real)1.0, qim1, grid.size);
			assign(eddy_rhs_2, (Real)1.0, qim2, grid.size);
			#if defined(DYNAMIC_MODEL) || defined(SIMPLE_MODEL)
			#ifdef Q_PARAMETERIZATION
				dyn_model1.apply(qim1, q1_rk, U1, V1, grid);
				dyn_model2.apply(qim2, q2_rk, U2, V2, grid);
			#else
				dyn_model1.apply(qim1, w1, U1, V1, grid);
				dyn_model2.apply(qim2, w2, U2, V2, grid);
			#endif
			#endif
			assign(eddy_rhs_1, (Real)1.0, qim1, - (Real)1.0, eddy_rhs_1, grid.size);
			assign(eddy_rhs_2, (Real)1.0, qim2, - (Real)1.0, eddy_rhs_2, grid.size);

            if (time_index % ndebug == 0) {
                check_const(qim1, "qim1 after adv and diff", grid);
				check_const(qim2, "qim2 after adv and diff", grid);
            }
            
            assign(q1_rk, (Real)1.0, q1, q[step] * dt, qim1, grid.size);
			assign(q2_rk, (Real)1.0, q2, q[step] * dt, qim2, grid.size);
            scalar_bc(q1_rk, q2_rk, grid);

            double pois_begin_mark = omp_get_wtime();
            two_layer_streamfunction(Psi1_rk, Psi2_rk, q1_rk, q2_rk, kd, fft_data, grid);
            cpu_pois_time += omp_get_wtime() - pois_begin_mark;
            
            scalar_bc(Psi1_rk, Psi2_rk, grid);
			velocity_stag(U1, V1, Psi1_rk, grid); 
			velocity_stag(U2, V2, Psi2_rk, grid); 
			
			velocity_bc(U1, V1, U2, V2, grid);
			vorticity(w1, U1, V1, grid);
			vorticity(w2, U2, V2, grid);
			scalar_bc(w1, w2, grid);
        }
        memcpy(Psi1, Psi1_rk, grid.size * sizeof(Real));
        memcpy(q1, q1_rk, grid.size * sizeof(Real));        
		memcpy(Psi2, Psi2_rk, grid.size * sizeof(Real));
        memcpy(q2, q2_rk, grid.size * sizeof(Real));        
	
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
    Real u1_max, v1_max, u2_max, v2_max;
	Real err;
	if (time_index % ndebug == 0) {
		velocity_abs_max(&u1_max, &v1_max, U1, V1, grid);
		velocity_abs_max(&u2_max, &v2_max, U2, V2, grid);
		err = check_solver(q1, q2, Psi1, Psi2, w1, w2);			
	}

	if ((grid.mpi_com.rank == 0) && (time_index % ndebug == 0)) {
		printf(" >> U1(max) = %.4f, V1(max) = %.4f, current CFL = %.4f \n", u1_max, v1_max, max(u1_max * dt / grid.dx, v1_max * dt / grid.dy));
		printf(" >> U2(max) = %.4f, V2(max) = %.4f, current CFL = %.4f \n", u2_max, v2_max, max(u2_max * dt / grid.dx, v2_max * dt / grid.dy));
		printf(" >> error in solver = %E \n", err);
		printf("C_back 1 and 2 layers = %.4f, %.4f\n", dyn_model1.Csim_back, dyn_model2.Csim_back);
		int est_sec = (int)(cpu_run_time * ((double)
			((end_time - current_time) / (current_time - begin_time))));

		int est_min = est_sec / 60; est_sec %= 60;
		int est_hrs = est_min / 60; est_min %= 60;

		printf("\t >> time: %.7f [ETA: %i:%i:%i] [IC: %.4f s]\n\n", current_time,
			est_hrs, est_min, est_sec,
			cpu_run_time / time_index);

	}
	
	//invariant_level(balance, w, Psi, grid);
	Real KE = two_layer_KE(U1, V1, U2, V2);
	Real PE = two_layer_PE(Psi1, Psi2);
	Real Zpot = two_layer_enstrophy(q1, q2);
	Real Zrel = two_layer_enstrophy(w1, w2);
	Real E_diss = E_dissipation(eddy_rhs_1, eddy_rhs_2, Psi1, Psi2);
	Real Z_diss = Z_dissipation(eddy_rhs_1, eddy_rhs_2, q1, q2);
	Real E_fric = E_dissipation(fric_rhs_1, fric_rhs_2, Psi1, Psi2);
	Real E_input = - E_dissipation(shear_rhs_1, shear_rhs_2, Psi1, Psi2);
	Real Heat = heat_flux(V1, V2, Psi1, Psi2);
	nse_series.push(0, KE);
	nse_series.push(1, PE);
	nse_series.push(2, KE+PE);
	nse_series.push(3, Zpot);
	nse_series.push(4, Zrel);
	nse_series.push(5, E_diss);
	nse_series.push(6, Z_diss);
	nse_series.push(7, E_fric);
	nse_series.push(8, E_input);
	nse_series.push(9, Heat);
	nse_series.push(10, (double)dyn_model1.Cs2_mean);
	nse_series.push(11, (double)dyn_model2.Cs2_mean);
	nse_series.push(12, (double)dyn_model1.LM_back_p);
	nse_series.push(13, (double)dyn_model2.LM_back_p);
	nse_series.push(14, (double)dyn_model1.LM_diss_to_back);
	nse_series.push(15, (double)dyn_model2.LM_diss_to_back);
    nse_series.push(16, (double)dyn_model1.MSE_germano);
	nse_series.push(17, (double)dyn_model2.MSE_germano);
	nse_series.push(18, (double)dyn_model1.Csim_back);
	nse_series.push(19, (double)dyn_model2.Csim_back);

	nse_series.push_time((double)current_time);

	double end_mark = omp_get_wtime();
	cpu_run_time += end_mark - begin_mark;
	return true;
}

int main(int argc, char** argv)
{
	omp_set_num_threads(OPENMP_CORES);

	MPI_Init(&argc, &argv);
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
	initial_condition(U1, V1, U2, V2, q1, q2, w1, w2, Psi1, Psi2, grid);
	
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
			      
			write_tecplot(OUTPUT_DIR"-q1-.plt", print_index,
				q1, "w",
				print_xmin, print_xmax,
				print_ymin, print_ymax,
				grid, current_time);
			/*
			write_tecplot(OUTPUT_DIR"-w2-.plt", print_index,
				w2, "w",
				print_xmin, print_xmax,
				print_ymin, print_ymax,
				grid, current_time);
			*/
            write_binary_przgn(OUTPUT_DIR"psi1-bin.nsx", Psi1, grid, print_index);
			write_binary_przgn(OUTPUT_DIR"psi2-bin.nsx", Psi2, grid, print_index);
			//write_binary_przgn(OUTPUT_DIR"q1-bin.nsx", q1, grid, print_index);
			//write_binary_przgn(OUTPUT_DIR"q2-bin.nsx", q2, grid, print_index);

			#if defined(DYNAMIC_MODEL) || defined(SIMPLE_MODEL)
			write_binary_przgn(OUTPUT_DIR"Cs2-1-bin.nsx", dyn_model1.Cs2_local, grid, print_index);
			write_binary_przgn(OUTPUT_DIR"tx1-bin.nsx", dyn_model1.tx, grid, print_index);
			write_binary_przgn(OUTPUT_DIR"ty1-bin.nsx", dyn_model1.ty, grid, print_index);
			write_binary_przgn(OUTPUT_DIR"ssmx1-bin.nsx", dyn_model1.ssmx, grid, print_index);
			write_binary_przgn(OUTPUT_DIR"ssmy1-bin.nsx", dyn_model1.ssmy, grid, print_index);

			write_binary_przgn(OUTPUT_DIR"Cs2-2-bin.nsx", dyn_model2.Cs2_local, grid, print_index);
			write_binary_przgn(OUTPUT_DIR"tx2-bin.nsx", dyn_model2.tx, grid, print_index);
			write_binary_przgn(OUTPUT_DIR"ty2-bin.nsx", dyn_model2.ty, grid, print_index);
			write_binary_przgn(OUTPUT_DIR"ssmx2-bin.nsx", dyn_model2.ssmx, grid, print_index);
			write_binary_przgn(OUTPUT_DIR"ssmy2-bin.nsx", dyn_model2.ssmy, grid, print_index);
			#endif

			print_mark += print_dt;
			print_index++;
		}
		
		if (current_time >= dump_mark) {
			/*
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
			dump_index++;*/
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
			fprintf(ptr, " \t viscosity model = %i \n", dyn_model1.viscosity_model);
			fprintf(ptr, " \t averaging method = %i \n", dyn_model1.averaging_method);
			fprintf(ptr, " \t mixed model = %i \n", dyn_model1.mixed_model);
			fprintf(ptr, " \t tf_width = %E \n", dyn_model1.tf_width);
			fprintf(ptr, " \t bf_width = %E \n", dyn_model1.bf_width);
			fprintf(ptr, " \t filter iterations = %i \n", dyn_model1.filter_iterations);
			fprintf(ptr, " \t lagrangian time = %E \n", dyn_model1.lagrangian_time);
			#else
			fprintf(ptr, " \t Dynamic model is turned off \n");
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
