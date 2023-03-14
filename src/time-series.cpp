#define _CRT_SECURE_NO_DEPRECATE
#include "time-series.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

namespace nse
{
	timeSeries::timeSeries() : nvar(0), size(0), ptr(0) { }
	timeSeries::timeSeries(const timeSeries& series)
	{
		if ((series.nvar > 0) && (series.size > 0)) {
			nvar = series.nvar;
			size = series.size; ptr = series.ptr;

			data = new double[(nvar + 1) * size];
			varname = new char*[nvar];
			for (int k = 0; k < nvar; k++) {
				varname[k] = new char[c_max_name_length + 1];
				strcpy(varname[k], series.varname[k]);
			}
			memcpy(data, series.data, (nvar + 1) * size * sizeof(double));
		}
	}
	timeSeries :: ~timeSeries()
	{
		if (nvar > 0) {
			for (int k = 0; k < nvar; k++)
				delete[] varname[k];
			delete[] varname;
			delete[] data;

			nvar = 0;
			size = 0; ptr = 0;
		}
	}

	bool timeSeries::set(const int n)
	{
		if ((n <= 0) || (size > 0)) return false;

		nvar = n;
		size = c_seq_init_length; ptr = 0;

		data = new double[(nvar + 1) * size];
		varname = new char*[nvar];
		for (int k = 0; k < nvar; k++) {
			varname[k] = new char[c_max_name_length + 1];
			strncpy(varname[k], "\0", c_max_name_length + 1);
		}

		return true;
	}

	bool timeSeries::init(const char* filename) const
	{
		FILE* file_ptr;
		file_ptr = fopen(filename, "wb");
		if (file_ptr == NULL) return false;

		fwrite(&nvar, sizeof(int), 1, file_ptr);
		int name_length = c_max_name_length;
		fwrite(&name_length, sizeof(int), 1, file_ptr);

		for (int k = 0; k < nvar; k++)
			fwrite(varname[k], sizeof(char), c_max_name_length + 1, file_ptr);

		fclose(file_ptr);
		return true;
	}

	bool timeSeries::init_append(const char* filename) const
	{
		FILE* file_ptr;
		file_ptr = fopen(filename, "rb");
		if (file_ptr == NULL) return false;

		int check_nvar, check_name_length;
		fread(&check_nvar, sizeof(int), 1, file_ptr);
		fread(&check_name_length, sizeof(int), 1, file_ptr);

		fclose(file_ptr);

		if ((check_nvar != nvar) || 
			(check_name_length != c_max_name_length)) return false;

		return true;
	}

	void timeSeries::reset() { ptr = 0; }

	long int timeSeries::length() const { return ptr; }

	bool timeSeries::push(const int idx, const double value)
	{
		if ((idx < 0) || (idx > nvar - 1)) return false;

		data[ptr * (nvar + 1) + idx + 1] = value;
		return true;
	}

	bool timeSeries::push(const char* name, const double value)
	{
		for (int k = 0; k < nvar; k++)
		if (!strcmp(varname[k], name)) {
			push(k, value);
			return true;
		}
		return false;
	}

	bool timeSeries::push_time(const double value)
	{
		if ((nvar == 0) || (size == 0)) return false;

		data[ptr * (nvar + 1)] = value;
		if (ptr == size - 1) {
			double* save = new double[size * (nvar + 1)];
			memcpy(save, data, size * (nvar + 1) * sizeof(double));

			delete[] data;

			data = new double[(size + c_seq_init_length) * (nvar + 1)];
			memcpy(data, save, size * (nvar + 1) * sizeof(double));
			size += c_seq_init_length;

			delete[] save;
		}

		ptr++;
		return true;
	}

	bool timeSeries::name_variable(const int idx, const char* name)
	{
		if ((idx < 0) || (idx > nvar - 1)) return false;

		if (strlen(name) < c_max_name_length + 1)
			strcpy(varname[idx], name);
		else
		{
			strncpy(varname[idx], name, c_max_name_length);
			varname[idx][c_max_name_length] = '\0';
		}

		return true;
	}

	bool timeSeries::write(const char* filename) const
	{
		if ((nvar == 0) || (size == 0)) return false;
		if (ptr == 0) return true;

		FILE* file_ptr;

		file_ptr = fopen(filename, "rb");
		if (file_ptr == NULL) return false;

		int check_nvar;
		fread(&check_nvar, sizeof(int), 1, file_ptr);
		fclose(file_ptr);

		if (check_nvar != nvar) return false;

		file_ptr = fopen(filename, "ab");
		if (file_ptr == NULL) return false;

		fwrite(data, sizeof(double), (nvar + 1) * ptr, file_ptr);

		fclose(file_ptr);
		return true;
	}
}

// Time-Slice -- (Averaging on the fly) //
namespace nse
{
	timeSlice::timeSlice() : psum_num(0), slice_ptr(0), slice_num(0),
		slice_size(0), psum_num_max(0) { }

	timeSlice::timeSlice(const timeSlice& slice)
	{
	}
	timeSlice :: ~timeSlice()
	{
		clear();
	}

	bool timeSlice::set(const long int _slice_size)
	{
		if (_slice_size <= 0) return false;

		clear();
		slice_size = _slice_size;

		slice_list = new double*[psum_size];
		for (int k = 0; k < psum_size; k++)
			slice_list[k] = new double[slice_size];

		slice_sum = new double[slice_size];
		for (int i = 0; i < slice_size; i++)
			slice_sum[i] = (double)0;

		psum_num_max = psum_num_init;
		psum_list = new double*[psum_num_max];
		for (int k = 0; k < psum_num_max; k++)
			psum_list[k] = new double[slice_size];

		return true;
	}

	void timeSlice::clear()
	{
		if (slice_size > 0) {

			for (int k = 0; k < psum_size; k++)
				delete[] slice_list[k];
			delete[] slice_list;
			delete[] slice_sum;

			for (int k = 0; k < psum_num_max; k++)
				delete[] psum_list[k];
			delete[] psum_list;
		}

		psum_num = 0;
		psum_num_max = 0;
		slice_ptr = 0;
		slice_num = 0;
		slice_size = 0;
	}

	template< typename T >
	bool timeSlice::push(const T* X)
	{
		if (slice_size <= 0) return false;

		// add "X" to the list of elements 
		int i;
		double x_value;
#pragma omp parallel for private(i, x_value) shared(X)
		for (i = 0; i < slice_size; i++) {
			x_value = (double)X[i];

			slice_list[slice_ptr][i] = x_value;
			slice_sum[i] += x_value;
		}

		slice_ptr++;
		slice_num++;

		// check if list is full
		if (slice_ptr >= psum_size) {

			// add partial sum of list at position [psum_num]
			memcpy(psum_list[psum_num], slice_sum, slice_size * sizeof(double));

			int i;
			double div_value = (double) 1.0 / psum_size;
#pragma omp parallel for private(i) shared(div_value)
			for (i = 0; i < slice_size; i++)
				psum_list[psum_num][i] *= div_value;


			psum_num++;
			// add additional memory to partial sums vector
			if (psum_num >= psum_num_max) {

				double **ptr_psum;
				ptr_psum = new double*[psum_num_max + psum_num_add];
				for (int k = 0; k < psum_num_max; k++) {
					ptr_psum[k] = new double[slice_size];
					memcpy(ptr_psum[k], psum_list[k], slice_size * sizeof(double));
					delete[] psum_list[k];
				}
				delete[] psum_list;
				for (int k = psum_num_max; k < psum_num_max + psum_num_add; k++)
					ptr_psum[k] = new double[slice_size];

				psum_num_max += psum_num_add;
				psum_list = ptr_psum;
			}

			slice_ptr = 0;
			for (int i = 0; i < slice_size; i++)
				slice_sum[i] = (double)0;
		}

		return true;
	}
	template bool timeSlice::push(const float* X);
	template bool timeSlice::push(const double* X);

	template< typename T >
	void timeSlice::average(T* X) const
	{
		double sum_i;
		double psum_div = (double)psum_size / (double)slice_num;

		int i, k;
#pragma omp parallel for private(i, k, sum_i) shared(X, psum_div)
		for (i = 0; i < slice_size; i++) {

			sum_i = (double)0;
			for (k = 0; k < psum_num; k++)
				sum_i += psum_list[k][i];
			sum_i *= psum_div;

			if (slice_ptr > 0)
				sum_i += (slice_sum[i] / ((double)slice_ptr)) *
				((double)slice_ptr / (double)slice_num);


			X[i] = (T)sum_i;
		}
	}
	template void timeSlice::average(float* X) const;
	template void timeSlice::average(double* X) const;

	bool timeSlice::write_dump(const char* filename) const
	{
		FILE *file_ptr = fopen(filename, "wb");
		if (file_ptr == NULL) return false;

		const int long header_idx = 'n' + 's' + 'e' + 'u';

		long int header[7];
		header[0] = header_idx;				// header checker //
		header[1] = psum_size;				// size of partial sum //
		header[2] = psum_num_max;			// max 'available' number of p-sums //
		header[3] = psum_num;				// num of 'filled' p-sums //
		header[4] = slice_ptr;				// num of elements in list //
		header[5] = slice_num;				// overall number of elements //
		header[6] = slice_size;				// dim size of each element //

		// write header //
		fwrite(header, sizeof(long int), 7, file_ptr);

		// write partial sums //
		for (int k = 0; k < psum_num; k++)
			fwrite(psum_list[k], sizeof(double), slice_size, file_ptr);

		// write current elements in list //
		for (int k = 0; k < slice_ptr; k++)
			fwrite(slice_list[k], sizeof(double), slice_size, file_ptr);

		// write current partial sum //
		fwrite(slice_sum, sizeof(double), slice_size, file_ptr);

		fclose(file_ptr);
		return true;
	}

	bool timeSlice::read_dump(const char* filename)
	{
		FILE *file_ptr = fopen(filename, "rb");
		if (file_ptr == NULL) return false;

		const int long header_idx = 'n' + 's' + 'e' + 'u';
		int ncount;

		long int header[7];

		// read header //
		ncount = fread(header, sizeof(long int), 7, file_ptr);
		if ((ncount != 7) || (header[0] != header_idx) || (header[1] != psum_size))
			return false;

		// clear container data ->
		clear();

		psum_num_max = header[2];
		psum_num = header[3];
		slice_ptr = header[4];
		slice_num = header[5];
		slice_size = header[6];


		ncount = 0;
		// read partial sums //
		psum_list = new double*[psum_num_max];
		for (int k = 0; k < psum_num_max; k++)
			psum_list[k] = new double[slice_size];
		for (int k = 0; k < psum_num; k++)
			ncount += fread(psum_list[k], sizeof(double), slice_size, file_ptr);

		// read current elements in list //
		slice_list = new double*[psum_size];
		for (int k = 0; k < psum_size; k++)
			slice_list[k] = new double[slice_size];
		for (int k = 0; k < slice_ptr; k++)
			ncount += fread(slice_list[k], sizeof(double), slice_size, file_ptr);

		// read current partial sum //
		ncount += fread(slice_sum, sizeof(double), slice_size, file_ptr);

		fclose(file_ptr);
		return (ncount == (psum_num + slice_ptr + 1) * slice_size);
	}
}
