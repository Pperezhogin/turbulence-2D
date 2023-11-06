add_executable(test1 test1.cpp)
add_executable(test2 test2.cpp)
add_executable(test3 test3.cpp)
add_executable(test_proc test_proc.cpp)

# link static libraries
target_link_libraries(test1 my_lib fftw3_mpi fftw3)
target_link_libraries(test2 my_lib fftw3_mpi fftw3)
target_link_libraries(test3 my_lib fftw3_mpi fftw3)
target_link_libraries(test_proc my_lib)

add_test(NAME Test_convolution  COMMAND "mpirun" -n 2 ./test1)
add_test(NAME Test_Taylor_Green_1core COMMAND "mpirun" -n 1 ./test2)
add_test(NAME Test_Taylor_Green_2core COMMAND "mpirun" -n 2 ./test2)
add_test(NAME Test_proc COMMAND "mpirun" -n 2 ./test_proc)
