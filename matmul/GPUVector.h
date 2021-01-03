#pragma once

#include "utils.h"


struct GPUVector {
	int n;
	double *elements;
	GPUVector(int N, double *data) : n(N) {
		cudaMallocManaged(&elements, sizeof(double)*N);
		cudaMemcpy(elements, data, sizeof(double)*N, cudaMemcpyDefault);
	}
	/*
	__global__ norm_kernel() {
		
	}
	norm(double &a) {
		
	}
	*/
};
