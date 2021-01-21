#include "stdio.h"
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <string>

#define LO 1
#define HI 20

int main(int argc, char** argv) {
	// call with parameters 
	int n = 10;
	if(argc > 1) {
		n = std::stoi( argv[1] );
	}
	double sparsity = 0.8;
	if(argc > 2) {
		sparsity = std::stod( argv[2] );
	}
		
	std::srand(std::time(nullptr));
	double data;
	std::string file_name = "vector_" + std::to_string(n);
	printf("file_name = %s\n",file_name.c_str());
	FILE *file = fopen(file_name.c_str(), "wb");
	
	int count = 0;
	bool is_data;
	// roll 6-sided dice 20 times
    for (int i = 0; i < n; i++) {
		is_data =  std::rand() > RAND_MAX * sparsity;
		if(is_data)
			count++;
		data =  is_data ? LO + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(HI-LO))) : 0;
		fwrite(&data, sizeof(double), 1, file);
    }
	fclose(file);
	printf("Counted %d of %d values\n",count,n);
}