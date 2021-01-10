#include "GPUVector.h"
#include "utils.h"


__global__ void norm_kernel(const double *data, const int n, double *output) {

	__shared__ double partialSum[2*BLOCK_SIZE];

	int i = threadIdx.x;

	int start = 2 * blockIdx.x * blockDim.x;

	// Each thread loads two elements
	partialSum[i] = data[start + i] * data[start + i];
	partialSum[blockDim.x + i] = data[start + blockDim.x + i] * data[start + blockDim.x + i];

	for (int stride = blockDim.x; stride > 0; stride /= 2) {
		__syncthreads();
		if (i < stride) partialSum[i] += partialSum[i + stride];
	}

	if (i == 0) *output = sqrt(partialSum[0]);
}

double GPUVector::norm() {
	double h_result;
	double *d_result;

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);

	cudaMalloc(&d_result, sizeof(double));

	norm_kernel<<<dimGrid, dimBlock>>>(elements, n, d_result);

	cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
	CUDAFREE(d_result);

	return h_result;
}

__global__ void scale_kernel(const double *input, double *output, const int n, const double s) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	output[i] = input[i] * s;
}

GPUVector operator*(const GPUVector v, const double s) {
	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid((v.n + dimBlock.x - 1) / dimBlock.x);

	GPUVector c(v.n);
	
	scale_kernel<<<dimGrid, dimBlock>>>(v.elements, c.elements, v.n, s);

	return c;
}
GPUVector operator*(const double s, const GPUVector v) {
	return v * s;
}

__global__ void add_kernel(const double *a, const double *b, double *c) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	c[i] = a[i] + b[i];
}

__global__ void sub_kernel(const double *a, const double *b, double *c) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	c[i] = a[i] - b[i];
}

GPUVector GPUVector::operator+(const GPUVector b) {
	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);

	GPUVector c(n);
	
	add_kernel<<<dimGrid, dimBlock>>>(elements, b.elements, c.elements);

	return c;
}

GPUVector GPUVector::operator-(const GPUVector b) {
	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);

	GPUVector c(n);

	sub_kernel<<<dimGrid, dimBlock>>>(elements, b.elements, c.elements);

	return c;
}