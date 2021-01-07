#pragma once

#include "utils.h"


struct GPUVector {
	int n;
	double *elements;
	GPUVector(int N, double *data) : n(N) {
		cudaMallocManaged(&elements, sizeof(double)*N);
		cudaMemcpy(elements, data, sizeof(double)*N, cudaMemcpyDefault);
	}
	double norm();
	void scale(GPUVector v, const double s);
	void add(const GPUVector b, GPUVector out);
	void sub(const GPUVector b, GPUVector out);
};
