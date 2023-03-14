#pragma once

// * time series container * //

// *[time-series.h]: statistics removed //

namespace nse
{
	class timeSeries {
	public:
		bool set(const int nvar);
		void reset();	            // reset sequence to beginning

		long int length() const;   // get sequence length

		// push variable referenced by index or name //
		bool push(const int idx, const double value);
		bool push(const char* name, const double value);
		// push sequence time [ step to next element ] //
		bool push_time(const double value);

		// name variable referenced by index
		bool name_variable(const int idx, const char* name);

		bool init(const char* filename) const;			// init: new output time series file
		bool init_append(const char* filename) const;	// init: append to output time series file
		bool write(const char* filename) const;			// write data to file

		timeSeries();
		timeSeries(const timeSeries& seq);
		~timeSeries();

	private:
		static const int c_max_name_length = 32;
		static const int c_seq_init_length = 16 * 1024;

		int nvar;				// number of variables in sequence
		long int size, ptr;		// sequence size and current length

		double* data;	        // series data
		char** varname;		    // variable names
	};


	class timeSlice {
	public:

		bool set(const long int slice_size);

		template< typename T >
		bool push(const T* X);

		template< typename T >
		void average(T* X) const;

		bool write_dump(const char* filename) const;
		bool write_dump(const char* filename, const int idx) const;
		bool write_dump(const char* filename, const int idx, const int sub_idx) const;

		bool read_dump(const char* filename);
		bool read_dump(const char* filename, const int idx);
		bool read_dump(const char* filename, const int idx, const int sub_idx);

		void clear();

		timeSlice();
		timeSlice(const timeSlice& t_cont);
		~timeSlice();

	private:

		// Control parameters //
		static const long int psum_size = 512;		// each psum contains [psum_size] elements
		static const long int psum_num_init = 64;	// number of psums at init
		static const long int psum_num_add = 64;	// number of psums to add on demand
		// ---------------------------------------------------------------------------- //

	public:
		// Partial sums data //
		long int psum_num_max;				// maximum number of availabe psums
		long int psum_num;					// number of current psums
		double **psum_list;					// list of psums 

		// Current working list of slices //
		long int slice_ptr;					// current number of elements in working list
		double **slice_list;				// current list of elements
		double *slice_sum;					// sum of current list

		// General slice parameters //
		long int slice_num;				// full number of elements(slices) = psum_num * psum_size + slice_ptr	 
		long int slice_size;			// size of element(slice)
	};
}
