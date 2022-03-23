export void sinx(
	uniform int N,
	uniform int terms,
	uniform float x[],
	uniform float y[])
{
	// assume N % programCount = 0
	for (uniform int i=0; i<N; i+=programCount)
	{
		int idx = i + programIndex;
		float value = x[idx];
		float numer = x[idx] * x[idx] * x[idx];
		uniform int denom = 6; //3!
		uniform int sign = -1;
		for (uniform int j=1; j<=terms; j++)
		{
			value += sign * numer / denom;
			numer *= x[idx] * x[idx];
			denom *= (2*j+2) * (2*j+3);
			sign *= -1;
		}
		y[idx] = value;
	} 
}
