#include "str-com.h"

#define _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#include <string.h> 

// implementation //
namespace nse
{
	char* app_index(
		const char* name, const int index)
	{
		char* app_name;
		sprintf(app_name, "%s%i", name, index);
		return app_name;
	}
	char* append_index( // append index to string
		const char* name, const int index)
	{
		char* copy;
		copy = new char[strlen(name) + 1]; strcpy(copy, name);

		// find last occurence of '.' signaling file name extension
		char* ptr; ptr = strrchr(copy, '.');
		if (ptr == NULL) { delete[] copy; return NULL; }

		(*ptr) = '\0'; ptr++; // set pointer to extension part of string

		// deterimine index digits amount
		int sign = (index >= 0) ? 1 : -1;
		int indexSize = (sign == -1);
		int indexCopy = sign * index;
		do
		{
			indexCopy /= 10; indexSize++;
		} while (indexCopy > 0);

		int size = strlen(copy) + 1 + strlen(ptr) + indexSize;
		char* app_name = new char[size + 1];
		sprintf(app_name, "%s%i.%s", copy, index, ptr);

		delete[] copy;
		return app_name;
	}

	bool copy_file( // copy file
		const char* filename_src, const char* filename_dest)
	{
		const int chunk_size = 10 * 1024;
		char chunk[chunk_size];


		FILE *file_src, *file_dest;
		
		file_src = fopen(filename_src, "rb");
		if (file_src == NULL) return false;

		file_dest = fopen(filename_dest, "wb");
		if (file_dest == NULL) {
			fclose(file_src); 
			return false;
		}

		int nsize;
		while (!feof(file_src)) {
			nsize = fread(chunk, sizeof(char), chunk_size, file_src);
			if ((nsize > 0) && (nsize <= chunk_size)) {
				fwrite(chunk, sizeof(char), nsize, file_dest);
			}
		}

		fclose(file_src);
		fclose(file_dest);

		return true;
	}

	bool copy_file( // copy file
		const char* filename_src,
		const char* filename_dest, const int dest_index)
	{
		char* ch_filename_dest = append_index(filename_dest, dest_index);
		bool status = copy_file(filename_src, ch_filename_dest);

		delete[] ch_filename_dest;
		return status;
	}

	bool copy_file( // copy file
		const char* filename_src, const int src_index,
		const char* filename_dest)
	{
		char* ch_filename_src = append_index(filename_src, src_index);
		bool status = copy_file(ch_filename_src, filename_dest);

		delete[] ch_filename_src;
		return status;
	}

	bool copy_file( // copy file
		const char* filename_src, const int src_index,
		const char* filename_dest, const int dest_index)
	{
		char* ch_filename_src = append_index(filename_src, src_index);
		char* ch_filename_dest = append_index(filename_dest, dest_index);

		bool status = copy_file(ch_filename_src, ch_filename_dest);

		delete[] ch_filename_src;
		delete[] ch_filename_dest;
		return status;
	}
}
