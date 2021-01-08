#pragma once

#include "utils.h"


struct GPUVector {
	int n;
	double *elements;
	GPUVector(int N, double *data) : n(N) {
		cudaMallocManaged(&elements, sizeof(double)*N);
		cudaMemcpy(elements, data, sizeof(double)*N, cudaMemcpyDefault);
	}
	GPUVector(int N) : n(N) {
		cudaMallocManaged(&elements, sizeof(double)*N);
	}
	double norm();
	GPUVector operator+(const GPUVector b);
	GPUVector operator-(const GPUVector b);
};

GPUVector operator*(const GPUVector v, const double s);
GPUVector operator*(const double s, const GPUVector v);