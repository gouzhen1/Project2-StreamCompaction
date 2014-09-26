#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "CPU_streamCompaction.h"

using namespace std;

__global__ void dev_initialize_array(int n, float * tar, float val)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < n) tar[index] = val;
}
__global__ void NaiveExclusivePrefixSum(int D, float * input,float * output, float * buffer, int n)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	//copy input to buffer
	buffer[index] = input[index];
	__syncthreads();
	if(index < n && index >= D)
	{
		input[index] = buffer[index - D ] + buffer[index ];
	}

	__syncthreads();
	output[index] = (index>0) ? input[index-1]:0.0f;
}

__global__ void SingleBlockExclusivePrefixSum(int D, float * input,float * output, int n)
{
	extern __shared__ float buffer[];	
	int index = threadIdx.x;
	//copy input to buffer
	if(index < n) buffer[index] = input[index];
	__syncthreads();

	if(index < n && index >= D)
	{
		input[index] = buffer[index - D ] + buffer[index];
	}

	__syncthreads();
	output[index] = (index>0) ? input[index-1]:0.0f;
}



int main(int argc, char** argv)
{

	//init
	float * in, *res, *dev_in, * dev_res, * dev_buffer;
	int n = 5;
	in = (float*)malloc(n * sizeof(float));
	res = (float*)malloc((1+n) * sizeof(float));
	cudaMalloc((void**) & dev_in, n * sizeof(float));
	cudaMalloc((void**) & dev_res, (n+1) * sizeof(float));
	cudaMalloc((void**) & dev_buffer, (n+1) * sizeof(float));

	//load data
	for(int i=0;i<n;i++)
	{
		in[i] = (float) i;
	}
	cudaMemcpy(dev_in,in,n * sizeof(float),cudaMemcpyHostToDevice);

	//CPU exprefixsum
	//exPrefixSum(in,n,res);
	
	#if(0)//naive GPU ex prefix sum//////////////////////////////////////////////////////////////////
	int gridSize = 1;
	int blockSize = n+1;
	dim3 gridDim(gridSize);
	dim3 blockDim(blockSize);
	for(int i=1;i<=log(n+1)+1;i++)
	{
		int D = pow(2,i-1);
		NaiveExclusivePrefixSum<<<gridDim,blockDim>>>(D,dev_in,dev_res, dev_buffer,n);
	}
	#endif////////////////////////////////////////////////////////////////////////////////////////////////

	#if(1)//single block with shared memory////////////////////////////////////////////////////////
	for(int i=1;i<=log(n+1)+1;i++)
	{
		int D = pow(2,i-1);
		SingleBlockExclusivePrefixSum<<<1,n+1,n*sizeof(float)>>>(D,dev_in,dev_res,n);
	}
	#endif////////////////////////////////////////////////////////////////////////////////////////////

	cudaMemcpy(res, dev_res, (n+1)*sizeof(float),cudaMemcpyDeviceToHost);

	//print result
	cout<<"input: ";
	for(int i=0;i<n;i++)
	{
		cout<<in[i]<<" ";
	}
	cout<<endl;

	cout<<"ex prefix sum: ";
	for(int i=0;i<n+1;i++)
	{
		cout<<res[i]<<" ";
	}

	cin.get();
    return 0;
}
