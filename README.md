# FLASH_RelatedWork
This repo is to house benchmark implementations on various portability frameworks. Directories must be modified per individual environment.

# Building KOKKOS projects:
Run Makefile in the directory, but make sure you have kokkos installed and in the systems path variables. Also, set the KOKKOS_DEVICES variable accordingly.

# Building RAJA projects :

/usr/local/cuda-10.2/bin/nvcc nbody.cpp -o nbody.bin -x cu -arch=sm_52 -I. -I/archive-t2/Design/fpga_computing/wip/RAJA_install/include -I/usr/local/cuda-11.2/include -I ../tpl/camp/include/ -I ../tpl/cub/ -L../../RAJA_install/lib/ --extended-lambda  -Xcompiler -fopenmp -lpthread -lcuda -lRAJA

/usr/local/cuda-10.2/bin/nvcc particle-diffusion.cpp -o particle-diffusion.bin -x cu -arch=sm_52 -I. -I/archive-t2/Design/fpga_computing/wip/RAJA_install/include -I/usr/local/cuda-11.2/include -I ../tpl/camp/include/ -I ../tpl/cub/ -L../../RAJA_install/lib/ --extended-lambda  -Xcompiler -fopenmp -lpthread -lcuda -lRAJA


# Building OpenMP projects:

(CPU)
/archive-t2/Design/fpga_computing/wip/depends/llvm-clang-cuda/bin/clang *.cpp -I. -I/archive-t2/Design/fpga_computing~/wip/depends/llvm-clang-cuda/include -lstdc++ -lm -o particle-diffusion.cpu -lgomp -fopenmp

(CPU)
/archive-t2/Design/fpga_computing/wip/depends/llvm-clang-cuda/bin/clang *.cpp -I. -I/archive-t2/Design/fpga_computing~/wip/depends/llvm-clang-cuda/include -lstdc++ -lm -o nbody.cpu -lgomp -fopenmp

(GPU)
nvcc -ccbin=/archive-t2/Design/fpga_computing/wip/depends/llvm-clang-cuda/bin/clang *.cpp -I. -I/archive-t2/Design/fpga_computing~/wip/depends/llvm-clang-cuda/include -I/usr/local/cuda-10.1/include -Xcompiler -fopenmp -Xcompiler -fopenmp-targets=nvptx64-nvidia-cuda  -arch=sm_52 -lcuda -lcudart -lcublas -o particle-diffusion.gpu

(GPU)
nvcc -ccbin=/archive-t2/Design/fpga_computing/wip/depends/llvm-clang-cuda/bin/clang *.cpp -I. -I/archive-t2/Design/fpga_computing~/wip/depends/llvm-clang-cuda/include -I/usr/local/cuda-10.1/include -Xcompiler -fopenmp -Xcompiler -fopenmp-targets=nvptx64-nvidia-cuda  -arch=sm_52 -lcuda -lcudart -lcublas -o particle-diffusion.gpu


# Building SyCL projects:

clang++ -fsycl *.cpp -o particle-diffusion.cpu -I. -I/home/user/mriera1/sycl_examples/include/ -I/home/user/mriera1/sycl_ws/llvm/sycl/include -I/home/user/mriera1/dpcpp_compiler/include/sycl -L/home/user/mriera1/dpcpp_compiler/lib/ -L/archive-t2/Design/fpga_computing/wip/mkl/lib/intel64 -I/archive-t2/Design/fpga_computing/wip/mkl/include -DCL_TARGET_OPENCL_VERSION=220  -lmkl_rt -lgomp

clang++ -fsycl *.cpp -o nbody.cpu -I. -I/home/user/mriera1/sycl_examples/include/ -I/home/user/mriera1/sycl_ws/llvm/sycl/include -I/home/user/mriera1/dpcpp_compiler/include/sycl -L/home/user/mriera1/dpcpp_compiler/lib/ -L/archive-t2/Design/fpga_computing/wip/mkl/lib/intel64 -I/archive-t2/Design/fpga_computing/wip/mkl/include -DCL_TARGET_OPENCL_VERSION=220  -lmkl_rt -lgomp
