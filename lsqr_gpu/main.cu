#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <chrono>
//#include <cusparse_v2.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "matmul/SpMat.h"
#include "matmul/GPUVector.h"
#include "lsqr_gpu.h"
#include <cublas_v2.h>

double norm(GPUVector &v) {
    double norm = v.norm();
	return norm;
}
// reads vector from file
void read_vector(char* file_name, double** data, int &n) {
	FILE *file = fopen(file_name, "rb");
	if (file==NULL) {fputs ("File error\n",stderr); exit (1);}
	
	char *token = strtok(file_name, "_");
	token = strtok(NULL, "_");
	n = std::stoi( token );
	
	// printf("Vector size: %d\n",n);
	
	*data = (double*) malloc (sizeof(double) * n);
	if (*data == NULL) {fputs ("Memory error",stderr); exit (2);}
	fread(*data,sizeof(double),n,file);
	fclose(file);
}

// reads matrix from file and parses it to csr format
void read_sparse_matrix(char* file_name, int** rowPtr, int** colInd, double** val, int& n, int& m, int& totalNnz) {
	// printf("file_name = %s\n",file_name);
	
	FILE *file = fopen(file_name, "rb");
	if (file==NULL) {fputs ("File error\n",stderr); exit (1);}
	
	char *token = strtok(file_name, "_");
	token = strtok(NULL, "_");
	m = std::stoi( token );
	token = strtok(NULL, "_");
	n = std::stoi( token );
	
	// printf("Matrix size: %dx%d\n",m,n);
	
	double *data = (double*) malloc (sizeof(double) * n);
	int * rowNnz = (int*) malloc(sizeof(int)*m);
	*rowPtr = (int*) malloc(sizeof(int)*(m+1)) ;
	//rowPtr = new int[m];
	// (*rowPtr)[0] = 0;
	totalNnz = 0;
	int rowCounter = 0;
	(*rowPtr)[0] = 0;
	if (data == NULL) {fputs ("Memory error",stderr); exit (2);}
	while(fread(data,sizeof(double),n,file)) {
		rowNnz[rowCounter] = 0;
		for(int i = 0; i < n; i++)
			if(std::abs(data[i]) > ZERO)
				rowNnz[rowCounter]++;
		totalNnz += rowNnz[rowCounter];
		rowCounter++;
		(*rowPtr)[rowCounter] = totalNnz;
	}
	
	// printf("Total Non-Zero Elements: %d\n",totalNnz);	
	rewind(file);

	*val = (double*) malloc(sizeof(double)*totalNnz);
	*colInd = (int*) malloc(sizeof(int)*totalNnz);
	int counter = 0;
	
	while(fread(data,sizeof(double),n,file)) {
		for(int i = 0; i < n; i++) 
			if(std::abs(data[i]) > ZERO){
				(*val)[counter] = data[i];
				(*colInd)[counter] = i;
				counter++;
			}
	}
	fclose(file);
	// printf("Read Data\n");
	FREE(data);
	FREE(rowNnz);
}


// expected input: mxn matrix binary file named "matrix_m_n", m vector binary file named "vector_m"
int main(int argc, char *argv[])
{
	cublasHandle_t handle;
	cublasStatus_t status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf("Error creating handle\n");
		exit(-1);
	}
	if(argc < 3) {
		printf("Matrix and vector file required\n");
		return 0;
	}
	char* matrix_file_name = argv[1];
	char* vector_file_name = argv[2];
	int* rowPtr = NULL; 
	int* colInd = NULL; 
	double* val = NULL;
	int n; 
	int m;
	int totalNnz;
	// reads matrix in csr format from file
	read_sparse_matrix(matrix_file_name, &rowPtr, &colInd, &val, n, m, totalNnz);
	double *vec_data = NULL;
	int vec_dim;
	// reads vector from file
	read_vector(vector_file_name, &vec_data, vec_dim);
	if(vec_dim != m) {
		printf("Vector dimension (%d) must agree with number of rows (%d) in matrix",vec_dim,m);
		return 0;
	}
	GPUVector b(handle, vec_dim,vec_data);
	GPUVector x(handle, n);
	SpMat A(rowPtr, colInd, val, m, n, totalNnz);

	printf("Starting Calculation (n = %d,m = %d)\n",n,m);
    printf("initial residual = %f\n",norm(b));
	// Start GPU timing
    cudaEvent_t evStart, evStop;
	cudaEventCreate(&evStart);
	cudaEventCreate(&evStop);
	cudaEventRecord(evStart, 0);

	lsqr(A,b,x);

	// Stop GPU timing
	cudaEventRecord(evStop, 0);
	cudaEventSynchronize(evStop);
	float elapsedTime_ms;
	cudaEventElapsedTime(&elapsedTime_ms, evStart, evStop);
	cudaEventDestroy(evStart);
	cudaEventDestroy(evStop);
	
	double *x_cpu = new double[n];
	cudaMemcpy(x_cpu, x.elements, sizeof(double) * n, cudaMemcpyDeviceToHost);

	printf("elapsed time [s]: %f\n",elapsedTime_ms/1000);
	GPUVector residual_vec = dot(A,x) - b;
    printf("final residual = %f\n",norm(residual_vec));
	
	printf("x = (");
	for(int i = 0; i < n; i++)
		printf("%f ",x_cpu[i]);
	printf(")\n");

	FREE(x_cpu);	
	FREE(rowPtr);
	FREE(colInd);
	FREE(val);
	FREE(vec_data);
    return 0;
}
