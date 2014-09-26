#include "CPU_streamCompaction.h"
void exPrefixSum(float * input, int n, float * out)
{
	float curSum = 0.0f;
	for(int i = 0;i < n+1 ;i++)
	{
		out[i] = curSum;
		curSum += input[i];
	}

}