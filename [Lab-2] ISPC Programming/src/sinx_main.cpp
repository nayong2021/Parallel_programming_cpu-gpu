#include "sinx_ispc.h"
#include <stdio.h>
#include <x86intrin.h>
#include <inttypes.h>

void sinx(int N, int terms, float x[], float y[])
{
	for (int i=0; i<N; i++)
	{
		float value = x[i];
		float numer = x[i] * x[i] * x[i];
		int denom = 6; //3!
		int sign = -1;

		for (int j=1; j<=terms; j++)
		{
			value += sign * numer / denom;
			numer *= x[i] * x[i];
			denom *= (2*j+2) * (2*j+3);
			sign *= -1;
		}

		y[i] = value;
	}
}

int main(int argc, void** argv) {
	int N = 1024;
	int terms = 5;
	float* x = new float[N];
	float * result = new float[N];
	unsigned long int start, end;
	start = __rdtsc();
	sinx(N, terms, x, result);
	end = __rdtsc();
	printf("Elapsed time with SISD : %"PRIu64"\n", end - start);
	start = __rdtsc();
	ispc::sinx(N, terms, x, result);
	end = __rdtsc();
	printf("Elapsed time with SIMD : %"PRIu64"\n", end - start);

	return 0;
}
