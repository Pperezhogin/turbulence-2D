set (lib_src Reynolds-equation.cpp a-priori.cpp dynamic-model.cpp fourier-methods.cpp mg-data2d.cpp mpi-com2d.cpp mtrand.cpp nse-out2d.cpp nse2d.cpp pois-base2d.cpp pois2d-fft.cpp pois2d.cpp str-com.cpp time-series.cpp memory-reader.cpp)

add_library(my_lib STATIC ${lib_src})

add_executable(DNS DNS.cpp)
add_executable(LES LES.cpp)
add_executable(two_layer two_layer.cpp)
add_executable(DNS_a_priori DNS_a_priori.cpp)

# link static libraries
target_link_libraries(DNS           my_lib fftw3_mpi fftw3)
target_link_libraries(LES           my_lib fftw3_mpi fftw3)
target_link_libraries(two_layer     my_lib fftw3_mpi fftw3)
target_link_libraries(DNS_a_priori  my_lib fftw3_mpi fftw3)