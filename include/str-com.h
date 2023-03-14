#pragma once

// * general nse routines for string operations * //

// *[str-com.h]: full //

namespace nse
{
	char* append_index(const char* name, const int index);
	char* app_index(const char* name, const int index);

	bool copy_file(const char* filename_src, const char* filename_dest);
	bool copy_file(const char* filename_src, 
		const char* filename_dest, const int dest_index);
	bool copy_file(const char* filename_src, const int src_index, 
		const char* filename_dest);
	bool copy_file(const char* filename_src, const int src_index,
		const char* filename_dest, const int dest_index);
}