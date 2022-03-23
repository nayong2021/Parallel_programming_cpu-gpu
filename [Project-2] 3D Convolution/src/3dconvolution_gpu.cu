#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define MAX_KERNEL 1000
#define TILE_SIZE 3

__constant__ float Mc[MAX_KERNEL];

__global__ void Convolution(float *N, float *P, int N_depth, int N_height, int N_width, int k_size, int P_depth, int P_height, int P_width) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;
	int row_o = blockIdx.y * TILE_SIZE + ty;
	int col_o = blockIdx.x * TILE_SIZE + tx;
	int height_o = blockIdx.z * TILE_SIZE + tz;

	int row_i = row_o - ((k_size - 1) / 2);
	int col_i = col_o - ((k_size - 1) / 2);
	int height_i = height_o - ((k_size - 1) / 2);

	float output = 0.0f;
	extern __shared__ float Ns[];

	if((row_i >= 0) && (row_i < N_height) && (col_i >= 0) && (col_i < N_width) && (height_i >= 0) && (height_i < N_depth))
		Ns[tz * (TILE_SIZE + (k_size - 1)) * (TILE_SIZE + (k_size - 1)) + ty * (TILE_SIZE + (k_size - 1)) + tx] = N[height_i*N_width*N_height + row_i*N_width + col_i];
	else
		Ns[tz * (TILE_SIZE + (k_size - 1)) * (TILE_SIZE + (k_size - 1)) + ty * (TILE_SIZE + (k_size - 1)) + tx] = 0.0f;

	__syncthreads();

	if(ty < TILE_SIZE && tx < TILE_SIZE && tz < TILE_SIZE){
		for(int i = 0; i < k_size; i++)
			for(int j = 0; j < k_size; j++)
				for(int k = 0; k < k_size; k++)
					output += Mc[i*k_size*k_size + j*k_size + k] * Ns[(i+tz)*(TILE_SIZE + (k_size - 1)) * (TILE_SIZE + (k_size - 1)) + (j+ty) * (TILE_SIZE + (k_size - 1)) + (k+tx)];
		if (row_o < P_height && col_o < P_width && height_o < P_depth)
			P[height_o * P_height * P_width +  row_o * P_width + col_o] = output;
	}
	
}

int main(int argc, char **argv)
{
	FILE *input_file, *kernel_file, *output_file;
	float *N, *M, *P, *d_N, *d_P, *answer;
	int i_depth, i_height, i_width, k_size, o_depth, o_height, o_width;
	ssize_t N_size, P_size;
	bool equal;

	input_file = fopen(argv[1], "r");
	if(input_file == NULL){
		printf("input_file open failed\n");
		exit(0);
	}
	kernel_file = fopen(argv[2], "r");
	if(kernel_file == NULL){
		printf("kernel_file open failed\n");
		exit(0);
	}	
	output_file = fopen(argv[3], "r");
	if(output_file == NULL){
		printf("output_file open failed\n");
		exit(0);
	}

	fscanf(input_file, "%d %d %d", &i_depth, &i_height, &i_width);
	N_size = i_depth * i_height * i_width * sizeof(float);
	N = (float*)malloc(N_size);
	for (int i = 0; i < i_depth * i_height * i_width; i++)
		fscanf(input_file, "%f", &N[i]);
	
	fscanf(kernel_file, "%d", &k_size);
	M = (float*)malloc(k_size * k_size * k_size * sizeof(float));
	for (int i = 0; i < k_size * k_size * k_size; i++)
		fscanf(kernel_file, "%f", &M[i]);	
	cudaMemcpyToSymbol(Mc, M, sizeof(float) * k_size * k_size * k_size);

	fscanf(output_file, "%d %d %d", &o_depth, &o_height, &o_width);
	P_size = o_depth * o_height * o_width * sizeof(float);
	P = (float*)malloc(P_size);
	memset(P, 0, P_size);
	answer = (float*)malloc(P_size);
	for (int i = 0; i < o_depth * o_height * o_width; i++)
		fscanf(output_file, "%f", &answer[i]);
	cudaMalloc((void**)&d_N, N_size);
	cudaMalloc((void**)&d_P, P_size);

	cudaMemcpy(d_N, N, N_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_P, P, P_size, cudaMemcpyHostToDevice);

	dim3 dimGrid(ceil((float)o_width/TILE_SIZE), ceil((float)o_height/TILE_SIZE), ceil((float)o_depth/TILE_SIZE));
	int block_size = TILE_SIZE + (k_size - 1);
	dim3 dimBlock(block_size, block_size, block_size);

	cudaEvent_t start, end;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start);
	Convolution<<<dimGrid, dimBlock, sizeof(float) * block_size * block_size * block_size>>>(d_N, d_P, i_depth, i_height, i_width, k_size, o_depth, o_height, o_width);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&elapsedTime, start, end);
	cudaMemcpy(P, d_P, P_size, cudaMemcpyDeviceToHost);
	
	equal = true;
	for (int i = 0; i < o_depth * o_height * o_width && equal; i++) {
		if (abs(answer[i] - P[i]) >= 0.001f) {
			equal = false;
			printf("NOT EQUAL!\n");
		}
	}

	if (equal) {
		printf("Results are equal!\n");
	}
	else {
		printf("Results are NOT equal!\n");
	}
	printf("GPU execution Elapsed Time : %fms\n", elapsedTime);
	free(N);
	free(M);
	free(P);
	free(answer);
	cudaFree(d_N);
	cudaFree(Mc);
	cudaFree(d_P);
	fclose(input_file);
	fclose(kernel_file);
	fclose(output_file);
	
	return 0;
}
