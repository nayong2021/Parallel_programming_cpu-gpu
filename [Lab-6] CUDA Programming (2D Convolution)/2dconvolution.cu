#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define KERNEL_SIZE 5
#define TILE_SIZE 3
#define MASK_WIDTH KERNEL_SIZE
#define BLOCK_SIZE (TILE_SIZE + (KERNEL_SIZE - 1))

__constant__ float Mc[MASK_WIDTH][MASK_WIDTH];

void verification(const float *N, const float *M, const float *P, int Rows, int Columns, int P_Rows, int P_Columns) {
	int r, c, h, w;
	int row_i, col_i;
	bool equal;
	float* results;

	results = (float*)malloc(P_Rows * P_Columns * sizeof(float));
	memset(results, 0, P_Rows * P_Columns * sizeof(float));
 	
	for (r = 0; r < P_Rows; r++) {
		for (c = 0; c < P_Columns; c++) {
			for (h = 0; h < KERNEL_SIZE; h++) {
				for (w = 0; w < KERNEL_SIZE; w++) {
					row_i = r - ((KERNEL_SIZE - 1) / 2) + h;
					col_i = c - ((KERNEL_SIZE - 1) / 2) + w;
					if ((row_i >= 0) && (row_i < Rows) && (col_i >= 0) && (col_i < Columns)) {
						results[r*P_Columns + c] += (M[h*KERNEL_SIZE+w] * N[row_i*Columns + col_i]);
					}
				}
			}
		}
	}
	equal = true;
	for (int i = 0; i < P_Rows * P_Columns && equal; i++) {
		if (abs(results[i] - P[i]) >= 0.001f) {
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

	free(results);
	return;
}

__global__ void Convolution(float *N, float *P, int N_height, int N_width, int Rows, int Columns) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row_o = blockIdx.y * TILE_SIZE + ty;
	int col_o = blockIdx.x * TILE_SIZE + tx;

	int row_i = row_o - ((KERNEL_SIZE - 1) / 2);
	int col_i = col_o - ((KERNEL_SIZE - 1) / 2);

	float output = 0.0f;
	__shared__ float Ns[TILE_SIZE+KERNEL_SIZE-1][TILE_SIZE+KERNEL_SIZE-1];

	if((row_i >= 0) && (row_i < N_height) && (col_i >= 0) && (col_i < N_width))
		Ns[ty][tx] = N[row_i*N_width + col_i];
	else
		Ns[ty][tx] = 0.0f;
	if(ty < TILE_SIZE && tx < TILE_SIZE){
		for(int i = 0; i < KERNEL_SIZE; i++)
			for(int j = 0; j < KERNEL_SIZE; j++)
				output += Mc[i][j] * Ns[i+ty][j+tx];
		if (row_o < Rows && col_o < Columns)
			P[row_o * Columns + col_o] = output;
	}
	
}

int main()
{
	srand((unsigned int)time(NULL));
	float* M;
	M = (float*)malloc(sizeof(float)*MASK_WIDTH*MASK_WIDTH);
	for(int i = 0; i < MASK_WIDTH; i++)
		for(int j = 0; j < MASK_WIDTH; j++)
			M[i*MASK_WIDTH+j] = rand()%3;
	cudaMemcpyToSymbol(Mc, M, sizeof(float) * MASK_WIDTH * MASK_WIDTH);
	int Rows, Columns, P_Rows, P_Columns;

	printf("Input Rows, Columns : ");
	scanf("%d %d", &Rows, &Columns);
	P_Rows = Rows-KERNEL_SIZE+1;
	P_Columns = Columns-KERNEL_SIZE+1;
	size_t N_size, P_size;
	N_size = Rows*Columns*sizeof(float);
	P_size = P_Rows*P_Columns*sizeof(float);

	float *P, *N, *d_P, *d_N;
	N = (float*)malloc(N_size);
	P = (float*)malloc(P_size);
	memset(P, 0, P_size);
	for(int i = 0; i < Rows*Columns; i++)
		N[i] = (float)(rand()%100) / 10;
	cudaMalloc((void**)&d_P, P_size);
	cudaMalloc((void**)&d_N, N_size);

	cudaMemcpy(d_N, N, N_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_P, P, P_size, cudaMemcpyHostToDevice);

	dim3 dimGrid(ceil((float)P_Columns/TILE_SIZE), ceil((float)P_Rows/TILE_SIZE));
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	Convolution<<<dimGrid, dimBlock>>>(d_N, d_P, Rows, Columns, P_Rows, P_Columns);
	cudaMemcpy(P, d_P, P_size, cudaMemcpyDeviceToHost);
	verification(N, M, P, Rows, Columns, P_Rows, P_Columns);

	cudaFree(d_N);
	cudaFree(d_P);
	free(N);
	free(P);
	cudaFree(M);
	return 0;
}
