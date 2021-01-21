#pragma once

#include <cuda.h>
#include <assert.h>
#include <stdlib.h>
#include <string>
#include <string.h> 
#include <stdio.h>

#define BLOCK_SIZE 16
#define ZERO 0.000000001

// some garbage collection
#define FREE(ptr) if(ptr) free(ptr)
#define CUDAFREE(ptr) if(ptr) cudaFree(ptr)
#define CUDASUCCESS(stat) assert(stat == cudaSuccess)
#define CUSPARSESUCCESS(stat) assert(stat == CUSPARSE_STATUS_SUCCESS)

// status (only for testing)
extern cudaError_t cudaStat; //= cudaSuccess;

// loads template array to gpu
template <typename datatype>
void to_gpu(datatype *array_cpu, datatype *array_gpu, const int n) {
	cudaMemcpy(array_gpu,array_cpu,sizeof(datatype)*n,cudaMemcpyHostToDevice);
}

// loads template array from gpu
template <typename datatype>
void from_gpu(datatype *array_cpu, datatype *array_gpu, const int n) {
	cudaMemcpy(array_cpu,array_gpu,sizeof(datatype)*n,cudaMemcpyDeviceToHost);
}
/*
void read_vector(char* file_name, double* data, int &n) {
	FILE *file = fopen(file_name, "rb");
	if (file==NULL) {fputs ("File error",stderr); exit (1);}
	
	char *token = strtok(file_name, "_");
	token = strtok(NULL, "_");
	n = std::stoi( token );
	
	printf("Vector size: %d\n",n);
	
	data = (double*) malloc (sizeof(double) * n);
	if (data == NULL) {fputs ("Memory error",stderr); exit (2);}
	fread(data,sizeof(double),n,file);
	fclose(file);
}

void read_sparse_matrix(char* file_name, int* rowPtr, int* colInd, double* val, int& n, int& m, int& totalNnz) {
	printf("file_name = %s\n",file_name);
	
	FILE *file = fopen(file_name, "rb");
	if (file==NULL) {fputs ("File error",stderr); exit (1);}
	
	char *token = strtok(file_name, "_");
	token = strtok(NULL, "_");
	m = std::stoi( token );
	token = strtok(NULL, "_");
	n = std::stoi( token );
	
	printf("Matrix size: %dx%d\n",m,n);
	
	double *data = (double*) malloc (sizeof(double) * n);
	int * rowNnz = (int*) malloc(sizeof(int)*m);
	rowPtr = (int*) malloc(sizeof(int)*m) ;
	totalNnz = 0;
	int rowCounter = 0;
	if (data == NULL) {fputs ("Memory error",stderr); exit (2);}
	while(fread(data,sizeof(double),n,file)) {
		rowNnz[rowCounter] = 0;
		for(int i = 0; i < n; i++)
			if(std::abs(data[i]) > ZERO)
				rowNnz[rowCounter]++;
		totalNnz += rowNnz[rowCounter];
		rowPtr[rowCounter + 1] = totalNnz;
		rowCounter++;
	}
	
	printf("Total Non-Zero Elements: %d\n",totalNnz);	
	rewind(file);

	val = (double*) malloc(sizeof(double)*totalNnz);
	colInd = (int*) malloc(sizeof(int)*totalNnz);
	int counter = 0;
	
	while(fread(data,sizeof(double),n,file)) {
		for(int i = 0; i < n; i++) 
			if(std::abs(data[i]) > ZERO){
				val[counter] = data[i];
				colInd[counter] = i;
				counter++;
			}
	}
	fclose(file);
	printf("Read Data\n");
	FREE(data);
	FREE(rowNnz);
}
*/