#pragma once

#include "utils.h"


struct GPUMatrix {
	int rows, cols;
	double *elements;
	GPUMatrix(int n, int m, double *data) : rows(m), cols(n) {
		cudaMalloc(&elements, sizeof(double)*rows*cols);
		cudaMemcpy(elements, data, sizeof(double)*rows*cols, cudaMemcpyHostToDevice);
	}
	~GPUMatrix() {
		CUDAFREE(elements);
	}
};
