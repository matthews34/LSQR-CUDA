#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h> 
#include <stdio.h> 

#define ZERO 0.0000001
#define FREE(ptr) if(ptr) free(ptr);

int main(int argc, char ** argv) {
	if(argc < 2)
		return 0;
	std::string file_name = argv[1];
	printf("file_name = %s\n",file_name.c_str());
	
	FILE *file = fopen(file_name.c_str(), "rb");
	if (file==NULL) {fputs ("File error",stderr); exit (1);}
	
	char * file_name_char = argv[1];
	char *token = strtok(file_name_char, "_");
	token = strtok(NULL, "_");
	int m = std::stoi( token );
	token = strtok(NULL, "_");
	int n = std::stoi( token );
	
	printf("Matrix size: %dx%d\n",m,n);
	
	double * data = (double*) malloc (sizeof(double) * n);
	int * rowNnz = (int*) malloc(sizeof(int)*m);
	int * rowPtr = (int*) malloc(sizeof(int)*m) ;
	int totalNnz = 0;
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


	double * val = (double*) malloc(sizeof(double)*totalNnz);
	int * colInd = (int*) malloc(sizeof(int)*totalNnz);
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
	/*
	printf("RowPtr = ");
	for(int i = 0; i < m+1;i++)
		printf("%d ",rowPtr[i]);
	printf("\n");
	printf("ColInd = ");
	for(int i = 0; i < totalNnz;i++)
		printf("%d ",colInd[i]);
	printf("\n");
	printf("Val = ");
	for(int i = 0; i < totalNnz;i++)
		printf("%f ",val[i]);
	printf("\n");
	*/
	
	FREE(data)
	FREE(rowNnz)
	FREE(rowPtr)
	FREE(colInd)
	FREE(val)
}