obj_files = GPUVector.o SpMat.o
all: lsqr

lsqr: $(obj_files)
	nvcc -o lsqr $^ main.cu -std=c++11 -lcublas

debug: $(obj_files)
	nvcc -o lsqr $^ main.cu -g -G -std=c++11 -lcublas

GPUVector.o: %.o: matmul/%.cu
	nvcc -c $^

SpMat.o: %.o: matmul/%.cu
	nvcc -c $^

clean:
	rm -f *.c *.o lsqr
