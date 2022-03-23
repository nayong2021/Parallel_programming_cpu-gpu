#include <immintrin.h>
#include <x86intrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <unistd.h>
#define ARRAY_LENGTH 8

float a[5][7] = {
	{4,9,2,0,3,6,1},
	{2,0,3,7,4,9,8},
	{4,1,7,9,8,1,8},
	{1,3,7,5,8,4,7},
	{1,2,3,0,9,4,8}
};
float b[7][15] = {
	{0,9,5,2,4,3,8,2,8,3,5,9,2,3,5},
	{9,5,4,2,7,9,5,9,5,2,5,1,2,5,7},
	{8,3,5,1,7,4,9,3,7,2,7,9,7,9,4},
	{1,6,2,4,4,6,5,8,6,5,3,0,1,5,3},
	{5,9,4,7,2,6,3,8,9,6,0,5,9,2,1},
	{1,0,3,2,7,8,5,3,7,8,3,1,3,4,6},
	{6,9,2,9,3,4,3,0,8,1,2,7,2,3,5}
};
float answer[5][15] = {
	{124,123,98,70,144,171,137,137,168,101,99,91,87,108,135},
	{108,177,98,153,152,188,159,128,242,151,95,130,111,136,141},
	{163,260,128,183,155,191,198,177,283,137,120,197,159,169,136},
	{174,210,120,162,159,196,176,166,258,135,110,168,160,159,136},
	{139,181,92,152,109,151,116,113,212,107,64,143,136,98,104},
};

void check_answer(float **c, int a_m, int b_p){
	for(int i = 0; i < a_m; i++)
		for(int j = 0; j < b_p; j++){
			if(answer[i][j] != c[i][j]){
				printf("wrong!\n");
				return;
			}
		}
	printf("correct!\n");
}

void clear_c(float **c, int a_m, int b_p){
	for(int i = 0; i < a_m; i++)
		for(int j = 0; j < b_p; j++)
			c[i][j] = 0;
}

int main(int argc, char *argv[])
{
	int a_m = sizeof(a)/sizeof(a[0]), b_p = sizeof(b[0])/sizeof(float), n = sizeof(a[0])/sizeof(float);
	float** b_extend = (float**)malloc(sizeof(float*) * n);
	unsigned long start, elapsed;
	float* temp_array = aligned_alloc(32, 32);
	float** c = (float**)malloc(sizeof(float*)*a_m);
	__m256**a_vector, ** b_vector, ** c_vector;
	int opt, mode;
	while((opt = getopt(argc, argv, "v:")) != -1){
		if(opt == 'v')
			mode = atoi(optarg);
	}
	if(mode == 1)
	{
		int b_vector_p = (b_p/8 + (b_p%8 ? 1 : 0));
		for(int i = 0; i < n; i++){
			b_extend[i] = aligned_alloc(32, sizeof(float) * 8 * b_vector_p);
			for(int j = 0; j < 8 * b_vector_p; j++){
				if(j < b_p)
					b_extend[i][j]=b[i][j];
				else
					b_extend[i][j]=0;
			}
		}
		a_vector = (__m256**)malloc(sizeof(__m256*)*a_m);
		for(int i = 0; i < a_m; i++){
			a_vector[i] = aligned_alloc(sizeof(__m256), sizeof(__m256) * n);
			for(int j = 0; j < n; j++)
				a_vector[i][j] = _mm256_broadcast_ss(&a[i][j]);
		}
		b_vector = (__m256**)malloc(sizeof(__m256*)*n);
		for(int i = 0; i < n; i++){
			b_vector[i] = aligned_alloc(sizeof(__m256), sizeof(__m256) * b_vector_p);
			for(int j = 0; j < b_vector_p; j++)
				b_vector[i][j] = _mm256_load_ps(&b_extend[i][j*8]);
		}
		c_vector = (__m256**)malloc(sizeof(__m256*)*a_m);
		for(int i = 0; i < a_m; i++){
			c_vector[i] = aligned_alloc(sizeof(__m256), sizeof(__m256) * b_vector_p);
			for(int j = 0; j < b_vector_p; j++)
				c_vector[i][j] = _mm256_setzero_ps();
		}
		
		
		start = __rdtsc();
		for(int i = 0; i < a_m; i++)
			for(int j = 0; j < n; j++){
				for(int k = 0; k < b_vector_p; k++)
					c_vector[i][k] = _mm256_add_ps (c_vector[i][k], _mm256_mul_ps(a_vector[i][j], b_vector[j][k]));
			}
		elapsed = __rdtsc() - start;
		printf("Elapsed time with AVX : %" PRIu64 "\n", elapsed);
		
		for(int i = 0; i < a_m; i++){
			c[i] = (float*)malloc(sizeof(float)*b_p);
			for(int j = 0; j < b_vector_p; j++){
				_mm256_store_ps(temp_array, c_vector[i][j]);
				for(int k = 0; k < 8; k++){
					if(j*8+k >= b_p)
						break;
					else
					c[i][j*8+k] = temp_array[k];
				}
			}
		}
		check_answer(c, a_m, b_p);
	}
	else if(mode == 2){
		for(int i = 0; i < a_m; i++)
			c[i] = (float*)malloc(sizeof(float)*b_p);
		start = __rdtsc();
		for(int i = 0; i < a_m; i++)
			for(int j = 0; j < b_p; j++)
				for(int k = 0; k < n; k++)
					c[i][j] += a[i][k] * b[k][j];
		elapsed = __rdtsc() - start;
		
		printf("Elapsed time with Non-AVX : %" PRIu64 "\n", elapsed);
		check_answer(c, a_m, b_p);
	}	
	return(0);
}	
