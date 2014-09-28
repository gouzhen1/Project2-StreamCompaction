#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include "CPU_streamCompaction.h"

using namespace std;
#define BLOCKDIM 128;

float Log2(float n)
{
	return log(n)/log(2);
}

__global__ void dev_initialize_array(int n, float * tar, float val)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < n) tar[index] = val;
}

__global__ void NaiveInclusivePrefixSum(int D, float * input, float * buffer, int n)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	//copy input to buffer
	buffer[index] = input[index];
	__syncthreads();
	if(index < n && index >= D)
	{
		input[index] = buffer[index - D ] + buffer[index ];
	}
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

__global__ void AddAuxToBlockedPrefixSum(float * input, float * aux, float * res, int n)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < n && blockIdx.x > 0) input[index] += aux[blockIdx.x - 1];
	if(index < n+1)	res[index] = (index <1)? 0.0f : input[index - 1];

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

__global__ void MultiBlockInclusivePrefixSum(int D, float * input, float * aux, int n)
{
	extern __shared__ float buffer[];	
	int localIndex = threadIdx.x;
	int globalIndex = blockDim.x * blockIdx.x + localIndex;
	//copy input to buffer
	if(localIndex < n) buffer[localIndex] = input[globalIndex];
	__syncthreads();

	if(localIndex < n && localIndex >= D)
	{
		input[globalIndex] = buffer[localIndex - D ] + buffer[localIndex];
	}

	__syncthreads();
	//if( globalIndex < n) output[globalIndex] = input[globalIndex];
	if(localIndex ==  blockDim.x - 1) aux[blockIdx.x] = (globalIndex>n)? input[n-1]: input[globalIndex];
}

void NaiveGPUexclusiveScan(float * input, float * output, int n)
{
	int blockSize = BLOCKDIM;
	int gridSize = ceil((float)n/(float)blockSize); 
	float * dev_buffer;
	cudaMalloc((void**) & dev_buffer, (n + 1) * sizeof(float));

	dim3 gridDim(gridSize);
	dim3 blockDim(blockSize);
	for(int i=1;i<=Log2(n) + 1;i++)
	{
		int D = pow(2,i-1);
		NaiveExclusivePrefixSum<<<gridDim,blockDim>>>(D,input,output, dev_buffer,n);
	}

}

void GPUexclusiveScan(float * input, float * output, int n)
{
	int blockDim = BLOCKDIM;
	int gridDim = ceil((float)n/(float)blockDim);
	float *dev_input, * dev_aux, * dev_buffer;
	cudaMalloc((void**) & dev_input, (n) * sizeof(float));
	cudaMalloc((void**) & dev_aux, (gridDim) * sizeof(float));
	cudaMalloc((void**) & dev_buffer, (n + 1) * sizeof(float));

	cudaMemcpy(dev_input, input, n*sizeof(float),cudaMemcpyDeviceToDevice);

	int D(0);
	for(int i=1;i< Log2(n) + 1;i++)
	{
		D = pow(2,i-1);
		MultiBlockInclusivePrefixSum<<<gridDim,blockDim,n*sizeof(float)>>>(D,dev_input,dev_aux,n);
	}
	/*
	//print aux
	cudaMemcpy(res, dev_aux, (gridDim)*sizeof(float),cudaMemcpyDeviceToHost);
	cout<<"aux: ";
	for(int i=0;i<gridDim;i++)
	{
		cout<<res[i]<<" ";
	}
	cout<<endl;*/
	//scan aux
	for(int i=1;i< Log2(gridDim) + 1;i++)
	{
		D = pow(2,i-1);
		NaiveInclusivePrefixSum<<<ceil((float)gridDim/(float)blockDim),blockDim,gridDim*sizeof(float)>>>(D,dev_aux,dev_buffer,gridDim);
	}
	//print scanned aux
	/*
	cudaMemcpy(res, dev_aux, (gridDim)*sizeof(float),cudaMemcpyDeviceToHost);
	cout<<"scanned aux: ";
	for(int i=0;i<gridDim;i++)
	{
		cout<<res[i]<<" ";
	}
	cout<<endl;*/

	//add aux to dev_in
	AddAuxToBlockedPrefixSum<<<gridDim+1,blockDim>>>(dev_input,dev_aux,output,n+1);
}

__global__ void generateBoolArray(float * input, float * out, int n)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index <n) out[index] = (input[index] == 0.0f )? 0.0f: 1.0f;
}

__global__ void generateCompactArray(float * input,float * boolArray, float * scannedBool, float * output, int n)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < n)
	{
		if(boolArray[index] > 0.0f) output[(int)scannedBool[index]] = input[index];
	}
}
void GPUstreamCompaction(float * input, float * output, int n)
{
	int blockDim = BLOCKDIM;
	int gridDim = ceil((float)n/(float)blockDim);

	float * boolArray, * scannedBool;
	cudaMalloc((void**) & boolArray, n * sizeof(float));
	cudaMalloc((void**) & scannedBool, (n+1) * sizeof(float));

	generateBoolArray<<<gridDim,blockDim>>>(input,boolArray,n);
	GPUexclusiveScan(boolArray,scannedBool,n);
	generateCompactArray<<<gridDim,blockDim>>>(input,boolArray, scannedBool, output, n);

}

int main(int argc, char** argv)
{
	//timers
	cudaEvent_t start, stop; 
	float time = 0.0f;
	//init
	float * in, *res, *dev_in, * dev_res;
	int n = 1000000;
	in = (float*)malloc(n * sizeof(float));
	res = (float*)malloc((1+n) * sizeof(float));
	cudaMalloc((void**) & dev_in, n * sizeof(float));
	cudaMalloc((void**) & dev_res, (n+1) * sizeof(float));
	//load data
	for(int i=0;i<n;i++)
	{
		//in[i] = (float) i;
		in[i] = (i%2 == 0) ? i : 0.0f;
	}
//	in[0] = 1.0f;in[1] = 3.0f;in[2] = 2.0f;in[3] = 1.0f;in[4] = 4.0f;in[5] = 2.5f;
	cudaMemcpy(dev_in,in,n * sizeof(float),cudaMemcpyHostToDevice);

	//print input
	cout<<"input: ";
	for(int i=0;i<n;i++)
	{
		//cout<<in[i]<<" ";
	}
	cout<<endl;

	//CPU exprefixsum////////////////////////////////////////////////////////////////////////////////
	
	CPUstreamCompaction(in,n,res);

	cout<<"CPU stream compact runtime: "<<time<<" ms"<<endl;
	cout<<"CPU compact stream result: ";
	for(int i=0;i<n+1;i++)
	{
		//cout<<res[i]<<" ";
	}
	cout<<endl;
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	#if(1)//naive GPU ex prefix sum//////////////////////////////////////////////////////////////////
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );
	NaiveGPUexclusiveScan(dev_in,dev_res,n);
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	cout<<"Naive GPU scan runtime: "<<time<<" ms"<<endl;
	#endif////////////////////////////////////////////////////////////////////////////////////////////////

	#if(1)//naive GPU ex prefix sum//////////////////////////////////////////////////////////////////
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );
	GPUexclusiveScan(dev_in,dev_res,n);
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	cout<<"GPU scan runtime: "<<time<<" ms"<<endl;
	#endif////////////////////////////////////////////////////////////////////////////////////////////////

	#if(0)//single block with shared memory ex prefix sum////////////////////////////////////////////////////////
	for(int i=1;i< Log2(n) + 1;i++)
	{
		int D = pow(2,i-1);
		SingleBlockExclusivePrefixSum<<<1,n+1,n*sizeof(float)>>>(D,dev_in,dev_res,n);
	}
	#endif////////////////////////////////////////////////////////////////////////////////////////////


	#if(0)//////////////////////////////////////////////////////////////////////////////////////////////////////
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );

	GPUstreamCompaction(dev_in,dev_res,n);

	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );


	cout<<"GPU stream compact runtime: "<<time<<" ms"<<endl;
	#endif//////////////////////////////////////////////////////////////////////////////////////////////////////

	cudaMemcpy(res, dev_res, (n)*sizeof(float),cudaMemcpyDeviceToHost);
	cout<<"GPU stream compact result: ";
	for(int i=0;i<n+1;i++)
	{
		//cout<<res[i]<<" ";
	}

	cin.get();
    return 0;
}
