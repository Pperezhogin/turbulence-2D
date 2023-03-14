#include <cassert>
#include <string>
#include "DNS_a_priori.h"

#define LAP_FOLDER           lap
#define BILAP_FOLDER         bilap
#define SSM_BILAP_FOLDER     ssm_bilap
#define SSM_BILAP_KEB_FOLDER ssm_bilap_keb

void init_4models(dynamic_model< Real >& dyn_lap, dynamic_model< Real >& dyn_bilap,
dynamic_model< Real >& dyn_ssm_bilap, dynamic_model< Real >& dyn_ssm_bilap_keb, int npoints)
{
	Real filter_width;
    Real dt = 0;
	
	filter_width = 2. * M_PI / Real(npoints) * sqrt(6.0) / grid.dx;

	dyn_lap.init(lap_smag, averaging_global, false, mixed_ssm, false, false, 
	false, 2, filter_width, filter_width, 
		1, (Real)1.0, dt, grid, false);

	dyn_bilap.init(bilap_smag, averaging_global, false, mixed_ssm, false, false, 
	false, 2, filter_width, filter_width, 
		1, (Real)1.0, dt, grid, false);

	dyn_ssm_bilap.init(bilap_smag, averaging_global, true, mixed_ssm, false, false, 
	false, 2, filter_width, filter_width, 
		1, (Real)1.0, dt, grid, false);

	dyn_ssm_bilap_keb.init(bilap_smag, averaging_global, true, mixed_ssm, false, true, 
	false, 2, filter_width, filter_width, 
		1, (Real)1.0, dt, grid, false);
}

void model_init() {
    allocate(&Psi, grid.size);
    
    init_4models(dyn_model_lap_128, dyn_model_bilap_128,
	dyn_model_ssm_bilap_128, dyn_model_ssm_bilap_keb_128, 128);

	init_4models(dyn_model_lap_256, dyn_model_bilap_256,
	dyn_model_ssm_bilap_256, dyn_model_ssm_bilap_keb_256, 256);

	init_4models(dyn_model_lap_512, dyn_model_bilap_512,
	dyn_model_ssm_bilap_512, dyn_model_ssm_bilap_keb_512, 512);
}

void write_fields(dynamic_model< Real >& dyn_model, Real *psic, int N_coarse, int print_index, string folder_) {
    write_binary_przgn_filter(folder_ + "tx.nsx", dyn_model.tx, N_coarse, grid, print_index);
    write_binary_przgn_filter(folder_ + "ty.nsx", dyn_model.ty, N_coarse, grid, print_index);
    write_binary_przgn_filter(folder_ + "ssmx.nsx", dyn_model.ssmx, N_coarse, grid, print_index);
    write_binary_przgn_filter(folder_ + "ssmy.nsx", dyn_model.ssmy, N_coarse, grid, print_index);
    write_binary_przgn_filter(folder_ + "bx.nsx", dyn_model.bx, N_coarse, grid, print_index);
    write_binary_przgn_filter(folder_ + "by.nsx", dyn_model.by, N_coarse, grid, print_index);
    write_binary_przgn_filter(folder_ + "psi-bin.nsx", psic, N_coarse, grid, print_index);
}

void apply_4_models(dynamic_model< Real >& dyn_lap, dynamic_model< Real >& dyn_bilap,
dynamic_model< Real >& dyn_ssm_bilap, dynamic_model< Real >& dyn_ssm_bilap_keb, 
int npoints, string folder, int print_index, int N_coarse) {
    Real psic[grid.size], wc[grid.size];
    Real uc[grid.size], vc[grid.size];

    Real bf_width;
    string folder_;

    bf_width = 2. * M_PI / Real(npoints) * sqrt(6.0);

    gauss_filter(psic, Psi, bf_width, grid);
    psi_bc(psic, grid);
    velocity_stag(uc, vc, psic, grid); 
    velocity_bc(uc, vc, grid);
    vorticity(wc, uc, vc, grid);
    w_bc(wc, grid);

    folder_ = folder + "/lap/";
    dyn_lap.allocate_memory(grid);
    dyn_lap.update_viscosity(wc, uc, vc, psic, (Real)0.0, false, (Real)0.0, grid);
    write_fields(dyn_lap, psic, N_coarse, print_index, folder_);
    dyn_lap.clear();

    folder_ = folder + "/bilap/";
    dyn_bilap.allocate_memory(grid);
    dyn_bilap.update_viscosity(wc, uc, vc, psic, (Real)0.0, false, (Real)0.0, grid);
    write_fields(dyn_bilap, psic, N_coarse, print_index, folder_);
    dyn_bilap.clear();

    folder_ = folder + "/ssm_bilap/";
    dyn_ssm_bilap.allocate_memory(grid);
    dyn_ssm_bilap.update_viscosity(wc, uc, vc, psic, (Real)0.0, false, (Real)0.0, grid);
    write_fields(dyn_ssm_bilap, psic, N_coarse, print_index, folder_);
    dyn_ssm_bilap.clear();

    folder_ = folder + "/ssm_bilap_keb/";
    dyn_ssm_bilap_keb.allocate_memory(grid);
    dyn_ssm_bilap_keb.update_viscosity(wc, uc, vc, psic, (Real)0.0, false, (Real)0.0, grid);
    write_fields(dyn_ssm_bilap_keb, psic, N_coarse, print_index, folder_);
    dyn_ssm_bilap_keb.clear();
    
}

int main(int argc, char** argv) 
{
    omp_set_num_threads(OPENMP_CORES);

	MPI_Init(&argc, &argv);

    grid.set(
    (Real) 0.0, (Real) 0.0, domain_length, domain_width,
    domain_nx, domain_ny, domain_gcx, domain_gcy, 1);

    int N_coarse = 512;

    model_init();

    printf("mpicom comm = %i \n", grid.mpi_com.comm);
    printf("mpicom size = %i \n", grid.mpi_com.size);

    for (int file_index = 1; file_index <= 50; file_index++) {
        read_binary_przgn("/data90t/users/perezhogin/decaying-turbulence/a_priori_2/DNS_4096/psi-bin.nsx", Psi, grid, file_index);
        
        apply_4_models(dyn_model_lap_128, dyn_model_bilap_128,
	    dyn_model_ssm_bilap_128, dyn_model_ssm_bilap_keb_128,
        128, "128_filter", file_index, N_coarse);

        apply_4_models(dyn_model_lap_256, dyn_model_bilap_256,
	    dyn_model_ssm_bilap_256, dyn_model_ssm_bilap_keb_256,
        256, "256_filter", file_index, N_coarse);

        apply_4_models(dyn_model_lap_512, dyn_model_bilap_512,
	    dyn_model_ssm_bilap_512, dyn_model_ssm_bilap_keb_512,
        512, "512_filter", file_index, N_coarse);
        if (grid.mpi_com.rank == 0)
            printf("ensemble member %i done \n", file_index);
    }
    
    MPI_Finalize();

    return 0;
}