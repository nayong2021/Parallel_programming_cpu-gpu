export void min_ispc(
	uniform int N,
	uniform int x[],
	uniform int min_x[])
{
	// assume N % programCount = 0
	for (uniform int i=0; i<N; i+=programCount)
	{
		int idx = i + programIndex;
		uniform int min_temp = reduce_min(x[idx]);
		*min_x = *min_x < min_temp ? *min_x : min_temp;
		
	}
}
export void max_ispc(
	uniform int N,
	uniform int x[],
	uniform int max_x[])
{
	// assume N % programCount = 0
	for (uniform int i=0; i<N; i+=programCount)
	{
		int idx = i + programIndex;
		uniform int max_temp = reduce_max(x[idx]);
		*max_x = *max_x > max_temp ? *max_x : max_temp;
	}
}
