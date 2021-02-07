echo "Compiling LSQR_CPU"
g++ -o LSQR_CPU -I lsqr_cpu/eigen-3.3.7 lsqr_cpu/lsqr_cpu.cpp -std=c++11
echo "Compiling LSQR_GPU"
nvcc -o LSQR_GPU lsqr_gpu/matmul/GPUVector.cu lsqr_gpu/matmul/SpMat.cu lsqr_gpu/main.cu -std=c++11 -lcublas