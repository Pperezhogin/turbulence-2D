#pragma once

// * boundary condition constants * //

// *[pois-const2d.h]: full //

namespace nse
{
	//#define _POIS2D_BC_DIRICHLET_POINT    // dirichlet point with periodic boundary conditions //

	const int c_pois_miniters = 1;		// minimum number of poisson solver iterations

	const int c_pois_bc_west_ext = 0;
	const int c_pois_bc_east_ext = 1;
	const int c_pois_bc_south_ext = 2;
	const int c_pois_bc_north_ext = 3;

	const int c_pois_bc_periodic_x = 4;
	const int c_pois_bc_periodic_y = 5;

	const int c_pois_bc_neumann = 6;

	const int c_pois_bc_periodic_xy = 7;

	
}
