#pragma once

#include "utils.h"


struct GPUMatrix {
	int M, N;
	double *elements;
	GPUMatrix(int n, int m, double *data) : M(m), N(n) {
		cudaMalloc(&elements, sizeof(double)*M*N);
		cudaMemcpy(elements, data, sizeof(double)*M*N, cudaMemcpyHostToDevice);
	}
	~GPUMatrix() {
		CUDAFREE(elements);
	}
};
