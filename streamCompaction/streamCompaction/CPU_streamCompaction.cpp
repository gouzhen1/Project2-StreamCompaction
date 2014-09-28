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

void CPUstreamCompaction(float * input, int n, float * out)
{
	float * boolInput = new float[n];
	for(int i=0;i<n;i++)
	{
		boolInput[i] = (input[i] == 0.0f) ? 0.0f : 1.0f;
	}

	float * scannedBool = new float[n+1];
	exPrefixSum(boolInput,n,scannedBool);

	for(int i=0;i<n;i++)
	{
		if(boolInput[i] > 0.0f) out[(int)scannedBool[i]] = input[i];
	}

	
}