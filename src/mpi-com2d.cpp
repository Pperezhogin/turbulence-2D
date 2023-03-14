#include "mpi-com2d.h"

// initialize: mpiCom2d constants
size_t nse::mpiCom2d::exch_size = 0;
size_t nse::mpiCom2d::exch_size_x = 0;
size_t nse::mpiCom2d::exch_size_y = 0;
size_t nse::mpiCom2d::exch_size_xy = 0;

void* nse::mpiCom2d::exch = NULL;
void* nse::mpiCom2d::exch_x = NULL;
void* nse::mpiCom2d::exch_y = NULL;
void* nse::mpiCom2d::exch_xy = NULL;

double nse::mpiCom2d::cpu_time_exch = (double)0;
double nse::mpiCom2d::cpu_time_exch_x = (double)0;
double nse::mpiCom2d::cpu_time_exch_y = (double) 0.0;
