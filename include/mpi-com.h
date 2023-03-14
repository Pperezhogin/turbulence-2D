 #pragma once

#include <mpi.h>

// *[mpi-com.h]: full //

namespace nse
{
	// * MPI reduction operators * //
	template< typename T >
	T mpi_allreduce(const T in, MPI_Op operation);
	template< typename T >
	T mpi_allreduce_comm(const T in, MPI_Op operation, MPI_Comm comm);

	template< typename T >
	void mpi_allreduce(T* x, MPI_Op operation);
	template< typename T >
	void mpi_allreduce_comm(T* x, MPI_Op operation, MPI_Comm comm);

	template< typename T >
	void mpi_allreduce(T* x, T* y, MPI_Op operation);
	template< typename T >
	void mpi_allreduce_comm(T* x, T* y, MPI_Op operation, MPI_Comm comm);

	template< typename T >
	void mpi_allreduce(T* x, T* y, T* z, MPI_Op operation);
	template< typename T >
	void mpi_allreduce_comm(T* x, T* y, T* z, MPI_Op operation, MPI_Comm comm);

	template< typename T >
	void mpi_allreduce(T* in, T* out, const int n, MPI_Op operation);
	template< typename T >
	void mpi_allreduce_comm(T* in, T* out, const int n, MPI_Op operation, MPI_Comm comm);

	// * MPI data type wrapper  * //
	template< typename T >
	MPI_Datatype mpi_type();

	// * MPI sub-array constructor (no commision) * //
	template< typename T >
	void mpi_subarray_type(MPI_Datatype* subarray,
		const int nx, const int ny, const int nz,
		const int ib, const int ie,
		const int jb, const int je,
		const int kb, const int ke);

	// * MPI topology factorization * //
	// size = mx * my factorization (mx >= my)
	void mpi_com_dims(int size, int* mx, int* my);
	// size = mx * my * mz factorization (mx >= my >= mz)
	void mpi_com_dims(int size, int* mx, int* my, int* mz);

	// * MPI decomposition: size (including ghost cells) * //
	int mpi_local_size(const int mpi_nx, const int gcx, const int rank_x, const int size_x);
	// * MPI decomposition: offset (including overlapping ghost cells) * //
	int mpi_local_offset(const int mpi_nx, const int gcx, const int rank_x, const int size_x);
}


namespace nse
{
	// * MPI reduction operators * //
	// --------------------------- //
	template< typename T > 
	inline T mpi_allreduce( // single element reduction
		const T in, MPI_Op operation)
	{
		T mpi_in = in, mpi_out;
		MPI_Allreduce(&mpi_in, &mpi_out, 1,
			mpi_type< T >(), operation, MPI_COMM_WORLD);

		return mpi_out;
	}

	template< typename T > 
	inline T mpi_allreduce_comm( // single element reduction
		const T in, MPI_Op operation, MPI_Comm comm)
	{
		T mpi_in = in, mpi_out;
		MPI_Allreduce(&mpi_in, &mpi_out, 1,
			mpi_type< T >(), operation, comm);

		return mpi_out;
	}

	template< typename T > 
	inline void mpi_allreduce( // single element reduction
		T* x, MPI_Op operation)
	{
		T in = (*x);
		MPI_Allreduce(&in, x, 1,
			mpi_type< T >(), operation, MPI_COMM_WORLD);
	}

	template< typename T > 
	inline void mpi_allreduce_comm( // single element reduction
		T* x, MPI_Op operation, MPI_Comm comm)
	{
		T in = (*x);
		MPI_Allreduce(&in, x, 1,
			mpi_type< T >(), operation, comm);
	}

	template< typename T > 
	inline void mpi_allreduce( // double element reduction
		T* x, T* y, MPI_Op operation)
	{
		T in[2], out[2];

		in[0] = (*x);
		in[1] = (*y);

		MPI_Allreduce(in, out, 2,
			mpi_type< T >(), operation, MPI_COMM_WORLD);

		(*x) = out[0];
		(*y) = out[1];
	}

	template< typename T > 
	inline void mpi_allreduce_comm( // double element reduction
		T* x, T* y, MPI_Op operation, MPI_Comm comm)
	{
		T in[2], out[2];

		in[0] = (*x);
		in[1] = (*y);

		MPI_Allreduce(in, out, 2,
			mpi_type< T >(), operation, comm);

		(*x) = out[0];
		(*y) = out[1];
	}

	template< typename T > 
	inline void mpi_allreduce( // triple element reduction
		T* x, T* y, T* z, MPI_Op operation)
	{
		T in[3], out[3];

		in[0] = (*x);
		in[1] = (*y);
		in[2] = (*z);

		MPI_Allreduce(in, out, 3,
			mpi_type< T >(), operation, MPI_COMM_WORLD);

		(*x) = out[0];
		(*y) = out[1];
		(*z) = out[2];
	}

	template< typename T > 
	inline void mpi_allreduce_comm( // triple element reduction
		T* x, T* y, T* z, MPI_Op operation, MPI_Comm comm)
	{
		T in[3], out[3];

		in[0] = (*x);
		in[1] = (*y);
		in[2] = (*z);

		MPI_Allreduce(in, out, 3,
			mpi_type< T >(), operation, comm);

		(*x) = out[0];
		(*y) = out[1];
		(*z) = out[2];
	}

	template< typename T >
	inline void mpi_allreduce(T* in, T* out, const int n, MPI_Op operation)
	{
		MPI_Allreduce(in, out, n,
			mpi_type< T >(), operation, MPI_COMM_WORLD);
	}

	template< typename T >
	inline void mpi_allreduce_comm(T* in, T* out, const int n, MPI_Op operation, MPI_Comm comm)
	{
		MPI_Allreduce(in, out, n,
			mpi_type< T >(), operation, comm);
	}

	// * MPI data type wrapper * //
	// ------------------------- //
	template< > inline MPI_Datatype mpi_type< float >() { return MPI_FLOAT; }
	template< > inline MPI_Datatype mpi_type< double >() { return MPI_DOUBLE; }
	template< > inline MPI_Datatype mpi_type< int >() { return MPI_INT; }
	template< > inline MPI_Datatype mpi_type< char >() { return MPI_CHAR; }
	template< > inline MPI_Datatype mpi_type< unsigned char >() { return MPI_UNSIGNED_CHAR; }


	// * MPI sub-array constructor (no commision) * //
	// -------------------------------------------- //
	template< typename T >
	inline void mpi_subarray_type(MPI_Datatype* subarray,
		const int nx, const int ny, const int nz,
		const int ib, const int ie,
		const int jb, const int je,
		const int kb, const int ke)
	{
		int size[3], subsize[3], pos[3];

		size[0] = nx; 
		size[1] = ny; 
		size[2] = nz;
		
		subsize[0] = ie - ib + 1;
		subsize[1] = je - jb + 1;
		subsize[2] = ke - kb + 1;

		pos[0] = ib;
		pos[1] = jb;
		pos[2] = kb;

		MPI_Type_create_subarray(3, size, subsize, pos,
			MPI_ORDER_C, mpi_type< T >(), subarray);
	}

	// * MPI topology factorization * //
	// ------------------------------ //
	inline void mpi_com_dims(const int size, int* mx, int* my)
	{
		int dims[2];
		dims[0] = 0; dims[1] = 0;
		
		MPI_Dims_create(size, 2, dims);
		(*mx) = dims[0]; 
		(*my) = dims[1];
	}

	inline void mpi_com_dims(const int size, int* mx, int* my, int* mz)
	{
		int dims[3];
		dims[0] = 0; dims[1] = 0; dims[2] = 0;

		MPI_Dims_create(size, 3, dims);
		(*mx) = dims[0];
		(*my) = dims[1];
		(*mz) = dims[2];
	}

	// * MPI decomposition: size (including ghost cells) * //
	// --------------------------------------------------- //
	inline int mpi_local_size(
		const int mpi_nx, const int gcx, const int rank_x, const int size_x)
	{
		const int mpi_cx = mpi_nx - 2 * gcx;

		int cx = mpi_cx / size_x;
		if (rank_x < (mpi_cx % size_x)) cx++;

		return cx + 2 * gcx;
	}

	// * MPI decomposition: offset (including overlapping ghost cells) * //
	// ----------------------------------------------------------------- //
	inline int mpi_local_offset(
		const int mpi_nx, const int gcx, const int rank_x, const int size_x)
	{
		const int mpi_cx = mpi_nx - 2 * gcx;
		const int cx = mpi_cx / size_x;

		int offset = 0;
		for (int i = 0; i < rank_x; i++) {
			offset += cx;
			if (i < (mpi_cx % size_x)) offset++;
		}

		return offset;
	}
}
