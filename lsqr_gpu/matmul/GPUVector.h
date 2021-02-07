#pragma once

#include "utils.h"
#include <cublas.h>

struct GPUVector {
	int n;
	double *elements;
	cublasHandle_t handle;
	GPUVector() {}
	GPUVector(cublasHandle_t handle, int N, double *data) : n(N), handle(handle) {
		// printf("Initializing Vector\n");
		cudaMallocManaged(&elements, sizeof(double)*N);
		cudaMemcpy(elements, data, sizeof(double)*N, cudaMemcpyDefault);
	}
	GPUVector(cublasHandle_t handle, int N) : n(N), handle(handle) {
		cudaMallocManaged(&elements, sizeof(double)*N);
		cudaMemset(elements,0,sizeof(double)*N);
	}
	double norm();
	GPUVector operator+(const GPUVector b);
	GPUVector operator-(const GPUVector b);
	int size() {return n;}
};

GPUVector operator*(const GPUVector v, const double s);
GPUVector operator*(const double s, const GPUVector v);