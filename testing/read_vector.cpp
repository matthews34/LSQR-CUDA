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
	int n = std::stoi( token );
	
	printf("Vector size: %d\n",n);
	
	double * data = (double*) malloc (sizeof(double) * n);
	if (data == NULL) {fputs ("Memory error",stderr); exit (2);}
	fread(data,sizeof(double),n,file);
	fclose(file);
	
	FREE(data)
}