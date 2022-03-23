#include "min_max_ispc.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, void** argv) {
	srand((unsigned int)time(NULL));
	int N = 1024;
	int* x = new int[N];
	int* min_x = new int;
	int* max_x = new int;
	*min_x = 2000;
	*max_x = 0;
	for(int i = 0; i < N; i++)
		x[i] = rand()%2000;
	ispc::min_ispc(N, x, min_x);
	ispc::max_ispc(N, x, max_x);
	printf("min = %d, max = %d\n", *min_x, *max_x);

	return 0;
}
