#pragma once

#include <omp.h>
#include <cmath>

#include "unigrid2d.h"
#include "vecmath.h"
#include "dynamic-model-supplementary.h"
#include "fourier-methods.h"
#include "nse2d.h"
#include "nse-out2d.h"
#include "pois2d-fft.h"

using namespace nse;

// --- main definitions --- //
#define OPENMP_CORES		1		// number of OpenMP cores to use

// - data type precision: float[double] //
#define Real double

// --- computational domain --- //
const Real domain_length = (Real) 2.0 * (Real)M_PI;        // - [x]
const Real domain_width = (Real) 2.0 * (Real)M_PI;         // - [y]

const int domain_nx = 8192, domain_ny = 8192;		// # of grid cells in -[x,y] directions

const int domain_gcx = 2, domain_gcy = 2, domain_gcz = 2;

const Real print_xmin = (Real) 0.0, print_xmax = domain_length;
const Real print_ymin = (Real) 0.0, print_ymax = domain_width;

uniGrid2d< Real > grid;
