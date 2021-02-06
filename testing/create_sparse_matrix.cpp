#include "stdio.h"
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <string>

#define LO 1
#define HI 20

int main(int argc, char** argv) {
	// call with parameters 
	int n = 10, m = 10;
	if(argc > 2) {
		m = std::stoi( argv[1] );
		n = std::stoi( argv[2] );
	}
	double sparsity = 0.8;
	if(argc > 3) {
		sparsity = std::stod( argv[3] );
	}
	bool random = true;
	if(argc > 4)
		random = false;
	std::srand(std::time(nullptr));
	double data;
	std::string file_name = "matrix_" + std::to_string(m);
	file_name += "_" + std::to_string(n);
	printf("file_name = %s\n",file_name.c_str());
	FILE *file = fopen(file_name.c_str(), "wb");
	int count = 0;
	bool is_data;
	bool is_diag;
	if(random) {
		for (int i = 0; i < m*n; i++) {
			is_data =  std::rand() > RAND_MAX * sparsity;
			is_diag = i/m == i%n;
			data =  (is_data || is_diag) ? LO + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(HI-LO))) : 0;
			if(is_data || is_diag)
				count++;
			fwrite(&data, sizeof(double), 1, file);
		}
	} else {
		for (int i = 0; i < m*n; i++) {
			data = count++;
			fwrite(&data, sizeof(double), 1, file);
		}
	}
	fclose(file);
	printf("Counted %d of %d values\n",count,n*m);
}