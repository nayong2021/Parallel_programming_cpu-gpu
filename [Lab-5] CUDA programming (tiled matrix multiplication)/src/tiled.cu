#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define TILE_WIDTH 32

void print_matrix(float* matrix, int x_width, int y_width, int width)
{
	for(int i = 0; i < x_width; i++)
	{
		for(int j = 0; j < y_width; j++)
			printf("%3.f", matrix[i*width+j]);
		printf("\n");
	}
}

int init_matrix(float** h_M, float** h_N, int M_xlen, int M_ylen, int N_xlen, int N_ylen)
{
	int P_xlen, P_ylen, width;
	if (M_xlen > N_xlen)
		P_xlen = M_xlen;
	else
		P_xlen = N_xlen;

	if (M_ylen > N_ylen)
		P_ylen = M_ylen;
	else
		P_ylen = N_ylen;
	
	if (P_xlen > P_ylen)
		width = P_xlen;
	else
		width = P_ylen;

	width = (width / TILE_WIDTH + (width % TILE_WIDTH ? 1 : 0)) * TILE_WIDTH;

	size_t matrix_size = width*width*sizeof(float);

	*h_M = (float*) malloc(matrix_size);
	*h_N = (float*) malloc(matrix_size);

	memset(*h_M, 0, matrix_size);
	memset(*h_N, 0, matrix_size);

	for(int i = 0; i < M_xlen; i++)
		for(int j = 0; j < M_ylen; j++)
			(*h_M)[i*width + j] = (rand() % 10 + 1);
	for(int i = 0; i < N_xlen; i++)
		for(int j = 0; j < N_ylen; j++)
			(*h_N)[i*width + j] = (rand() % 10 + 1);
	return width;
}

void serial_matrix_mul(float *h_M, float *h_N, float *h_P, int M_xlen, int N_ylen, int width)
{
	float sum;
	for (int i = 0; i < M_xlen; i++)
	{
		for (int j = 0; j < N_ylen; j++)
		{
			sum = 0;
			for (int k = 0; k < width; k++)
				sum += h_M[i*width + k] * h_N[k*width + j];
			h_P[i * width + j] = sum;
		}
	}
}

__global__ void MatrixMulKernel (float *d_M, float *d_N, float *d_P, int width)
{
	__shared__ float d_M_tile[TILE_WIDTH][TILE_WIDTH];
	__shared__ float d_N_tile[TILE_WIDTH][TILE_WIDTH];

	int Row = blockIdx.y*blockDim.y+threadIdx.y;
	int Col = blockIdx.x*blockDim.x+threadIdx.x;
		
	if((Row < width) && (Col < width))
	{
		float Pvalue = 0;
		
		for(int tileNUM = 0; tileNUM < gridDim.x; tileNUM++){
			int i = tileNUM * TILE_WIDTH + threadIdx.y;
			int j = tileNUM * TILE_WIDTH + threadIdx.x;

			d_M_tile[threadIdx.y][threadIdx.x] = d_M[Row * width + j];
			d_N_tile[threadIdx.y][threadIdx.x] = d_N[i * width + Col];

			__syncthreads();
			for(int k = 0; k < TILE_WIDTH; k++)
				Pvalue += d_M_tile[threadIdx.y][k] * d_N_tile[k][threadIdx.x];
			__syncthreads();
		}
		d_P[Row*width+Col] = Pvalue;
	}
}

int main()
{
	srand((unsigned int)time(NULL));
	int M_xlen, M_ylen, N_xlen, N_ylen, width;
	float *h_M, *h_N, *d_M, *d_N, *d_P, *P_result, time_ms;
	size_t matrix_size;

	printf("Input M_xlen M_ylen N_xlen N_ylen\n");
	scanf("%d %d %d %d", &M_xlen, &M_ylen, &N_xlen, &N_ylen);
	
	width = init_matrix(&h_M, &h_N, M_xlen, M_ylen, N_xlen, N_ylen);
	matrix_size = width*width*sizeof(float);

	P_result = (float*)malloc(matrix_size);
	
	cudaMalloc(&d_M, matrix_size);
	cudaMalloc(&d_N, matrix_size);
	cudaMalloc(&d_P, matrix_size);

	cudaMemcpy(d_M, h_M, matrix_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_N, h_N, matrix_size, cudaMemcpyHostToDevice);

	dim3 dimGrid(ceil(width/TILE_WIDTH), ceil(width/TILE_WIDTH), 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start, 0);

	MatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, width);
	
	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time_ms, start, end);

	printf("tiled mul : %fms\n", time_ms);

	cudaMemcpy(P_result, d_P, matrix_size, cudaMemcpyDeviceToHost);

	free(h_M);
	free(h_N);
	free(P_result);
	cudaFree(d_M);
	cudaFree(d_N);
	cudaFree(d_P);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	return 0;
}
