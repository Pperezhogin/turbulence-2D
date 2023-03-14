#define _CRT_SECURE_NO_DEPRECATE
#include "SGS_KE_LES.h"

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
    
	balance.init((Real)1.0, grid);

	#ifdef ON_BACKSCATTER
	nse_series.set(15);
	#else
	nse_series.set(5);
	#endif
	nse_series.name_variable(0, "kinetic energy");
	nse_series.name_variable(1, "enstrophy");
    nse_series.name_variable(2, "palinstrophy");
    nse_series.name_variable(3, "kinetic energy viscous dissipation");
	nse_series.name_variable(4, "enstrophy viscous dissipation");
	#ifdef ON_BACKSCATTER
    nse_series.name_variable(5, "Ediss eq");
	nse_series.name_variable(6, "Eback eq");
	nse_series.name_variable(7, "Ediss ten");
	nse_series.name_variable(8, "Eback ten");
	nse_series.name_variable(9, "SGS KE");
	nse_series.name_variable(10, "SFS KE");
	nse_series.name_variable(11, "nu2");
	nse_series.name_variable(12, "neg SGS KE");
	nse_series.name_variable(13, "min SGS KE");
	nse_series.name_variable(14, "KE sum");
	#endif

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

#ifdef ON_BACKSCATTER 
    sgs_ke.clear();
#endif
}

// --------------------------- //
// Init Navier-Stokes equation //
// --------------------------- //
void init_nse_eq()
{  
	#ifdef ON_BACKSCATTER
	sgs_ke.init(w, U, V, nu2_method, initial_cond, dt, grid);
	#endif
}

// ------------------------------- //
// Advance Navier-Stokes equation  //
// ------------------------------- //
bool advance_nse_eq_runge_kutta()
{
        double begin_mark = omp_get_wtime();

		#ifdef ON_BACKSCATTER
		sgs_ke.update_KE(w, U, V, grid); 
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

            #ifdef DIFFUSION
            w_diffusion(wim, w_rk, (Real)1.0 / c_Reynolds, grid);
            #endif

			#ifdef ON_BACKSCATTER
			sgs_ke.apply(wim, U_rk, V_rk, grid);
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
		printf(" >> U(max) = %.4f, V(max) = %.4f, current CFL = %.4f \n", u_max, v_max, max(u_max * dt / grid.dx, v_max * dt / grid.dy));

		int est_sec = (int)(cpu_run_time * ((double)
			((end_time - current_time) / (current_time - begin_time))));

		int est_min = est_sec / 60; est_sec %= 60;
		int est_hrs = est_min / 60; est_min %= 60;

		printf("\t >> time: %.7f [ETA: %i:%i:%i] [IC: %.4f s]\n\n", current_time,
			est_hrs, est_min, est_sec,
			cpu_run_time / time_index);

	}

	nse_series.push(0,    (double)energy);
	nse_series.push(1,    (double)enstrophy);
	nse_series.push(2,    (double)palinstrophy);
	nse_series.push(3,  - (double)balance.en_visc);
	nse_series.push(4,  - (double)balance.ens_visc);
	#ifdef ON_BACKSCATTER
	nse_series.push(5,    (double)sgs_ke.m_Ediss_eq);
	nse_series.push(6,    (double)sgs_ke.m_Eback_eq);
	nse_series.push(7,    (double)sgs_ke.m_Ediss_ten);
	nse_series.push(8,    (double)sgs_ke.m_Eback_ten);
	nse_series.push(9,    (double)sgs_ke.m_SGS_KE);
	nse_series.push(10,   (double)sgs_ke.m_SFS_KE);
	nse_series.push(11,   (double)sgs_ke.m_nu2);
	nse_series.push(12,   (double)sgs_ke.neg_SGS_KE);
	nse_series.push(13,   (double)sgs_ke.min_SGS_KE);
	nse_series.push(14,   (double)sgs_ke.m_SGS_KE+(double)energy);
	#endif
	
	nse_series.push_time((double)current_time);

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

	if (grid.mpi_com.rank == 0)
	{
		printf("My initial data index is %i\n", file_index);
	}

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
        
			#ifdef ON_BACKSCATTER
			write_tecplot(OUTPUT_DIR"-nu2-.plt", print_index,
				sgs_ke.nu2, "w",
				print_xmin, print_xmax,
				print_ymin, print_ymax,
				grid, current_time);

			write_tecplot(OUTPUT_DIR"-Ediss-.plt", print_index,
				sgs_ke.Ediss, "w",
				print_xmin, print_xmax,
				print_ymin, print_ymax,
				grid, current_time);

			write_tecplot(OUTPUT_DIR"-Eback-.plt", print_index,
				sgs_ke.Eback, "w",
				print_xmin, print_xmax,
				print_ymin, print_ymax,
				grid, current_time);

			write_tecplot(OUTPUT_DIR"-SGS_KE-.plt", print_index,
				sgs_ke.sgs_ke, "w",
				print_xmin, print_xmax,
				print_ymin, print_ymax,
				grid, current_time);
            
			write_tecplot(OUTPUT_DIR"-SFS_KE-.plt", print_index,
				sgs_ke.sfs_ke, "w",
				print_xmin, print_xmax,
				print_ymin, print_ymax,
				grid, current_time);

			write_tecplot(OUTPUT_DIR"-nu_eddy-.plt", print_index,
				sgs_ke.nu_eddy, "w",
				print_xmin, print_xmax,
				print_ymin, print_ymax,
				grid, current_time);

			write_binary_przgn(OUTPUT_DIR"-wim_diss-.nsx", sgs_ke.wim_diss, grid, print_index);
			write_binary_przgn(OUTPUT_DIR"-wim_back-.nsx", sgs_ke.wim_back, grid, print_index);
			#endif
            write_binary_przgn(PSI_BIN_FILE, Psi, grid, print_index);

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
