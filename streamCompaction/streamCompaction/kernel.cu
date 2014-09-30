#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/count.h>

#include <iostream>
#include <cmath>
#include <ctime>

#include "CPU_streamCompaction.h"

using namespace std;
#define BLOCKDIM 256;

float Log2(float n)
{
	return log(n)/log(2);
}

__global__ void dev_initialize_array(int n, float * tar, float val)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < n) tar[index] = val;
}

//helper for shared mem ex scan
__global__ void AddAuxToBlockedPrefixSum(float * input, float * aux, int n)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < n && blockIdx.x > 0) input[index] += aux[blockIdx.x - 1];
	//if(index < n+1)	res[index] = (index <1)? 0.0f : input[index - 1];

}

//single block shared memory ex scan
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

//helper for arbitrary array length shared mem scan
__global__ void MultiBlockInclusivePrefixSum(float * input, float * aux, int n)
{
	extern __shared__ float buffer[];	
	int localIndex = threadIdx.x;
	int globalIndex = blockDim.x * blockIdx.x + localIndex;
	//copy input to buffer
	for(int D = 1;D<n; D*=2)
	{
		if(localIndex < blockDim.x) buffer[localIndex] = input[globalIndex];
		__syncthreads();

		if(localIndex < blockDim.x && localIndex >= D)
		{
			input[globalIndex] = buffer[localIndex - D ] + buffer[localIndex];
		}

		__syncthreads();

		if(localIndex ==  blockDim.x - 1)
		{
			aux[blockIdx.x] = (globalIndex>n)? input[n-1]: input[globalIndex];
		}

	}

}

__global__ void NaiveExclusivePrefixSum(int D, float * input,float * output, float * buffer, int n)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	//copy input to buffer
	if(index < n) buffer[index] = input[index];
	if(index < n && index >= D)
	{
		input[index] = buffer[index - D ] + buffer[index ];
	}

	__syncthreads();
	if(index < n+1) 
	{
		output[index] = (index>0) ? input[index-1]:0.0f;
	}
	__syncthreads();


}

__global__ void NaiveCopyInputToBuffer(float * input, float * buffer, int n)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	//copy input to buffer
	if(index < n) buffer[index] = input[index];
}

__global__ void NaiveOneIteration(int D, float * input, float * buffer, int n)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(index < n && index >= D)
	{
		input[index] = buffer[index - D ] + buffer[index ];
	}
}

__global__ void NaiveWriteToOutputExclusive(float * input,float * output, int n)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < n+1) 
	{
		output[index] = (index>0) ? input[index-1]:0.0f;
	}
}

__global__ void NaiveWriteToOutputInclusive(float * input,float * output, int n)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < n) 
	{
		output[index] = input[index];
	}
}

//will change input
void NaiveGPUexclusiveScan(float * input,float * output, int n)
{
	int blockSize = BLOCKDIM;
	int gridSize = ceil((float)n/(float)blockSize); 
	float * dev_buffer;
	cudaMalloc((void**) & dev_buffer, (n) * sizeof(float));

	dim3 gridDim(gridSize);
	dim3 blockDim(blockSize);

	for(int D=1;D< n; D*= 2)
	{
		//NaiveExclusivePrefixSum<<<gridDim,blockDim>>>(D,input,output, dev_buffer,n);
		NaiveCopyInputToBuffer<<<gridDim,blockDim>>>(input, dev_buffer,n);
		NaiveOneIteration<<<gridDim,blockDim>>>(D,input, dev_buffer,n);
	}
	NaiveWriteToOutputExclusive<<<gridDim,blockDim>>>(input,output,n);

	cudaFree(dev_buffer);
	#if(0)
	float * show;
	show = (float*)malloc((n+1) * sizeof(float));
	cudaMemcpy(show, output, (n+1)*sizeof(float),cudaMemcpyDeviceToHost);
	cout<<"Naive Exclusive Scan result: ";
	for(int i=0;i<n;i++)
	{
		if(i==n-1) cout<<show[i]<<" ";
	}
	cout<<endl;
	free(show);
	#endif
}

//in-place
void NaiveGPUinclusiveScan(float * input,int n)
{
	int blockSize = BLOCKDIM;
	int gridSize = ceil((float)n/(float)blockSize); 
	float * dev_buffer;
	cudaMalloc((void**) & dev_buffer, (n) * sizeof(float));

	dim3 gridDim(gridSize);
	dim3 blockDim(blockSize);

	for(int D=1;D< n; D*= 2)
	{
		//NaiveExclusivePrefixSum<<<gridDim,blockDim>>>(D,input,output, dev_buffer,n);
		NaiveCopyInputToBuffer<<<gridDim,blockDim>>>(input, dev_buffer,n);
		NaiveOneIteration<<<gridDim,blockDim>>>(D,input, dev_buffer,n);
	}
	//NaiveWriteToOutputInclusive<<<gridDim,blockDim>>>(input,output,n);

	cudaFree(dev_buffer);
	#if(0)
	float * show;
	show = (float*)malloc((n) * sizeof(float));
	cudaMemcpy(show, input, (n)*sizeof(float),cudaMemcpyDeviceToHost);
	cout<<"Naive Inclusive Scan result: ";
	for(int i=0;i<n;i++)
	{
		if(i==n-1) cout<<show[i]<<" ";
	}
	cout<<endl;
	free(show);
	#endif

}

//will change the input
void GPUexclusiveScan(float * input, float * output, int n)
{
	int blockDim = BLOCKDIM;
	int gridDim = ceil((float)n/(float)blockDim);

	//allocate memory
	float * dev_aux, * dev_buffer;
	cudaMalloc((void**) & dev_aux, (gridDim) * sizeof(float));
	cudaMalloc((void**) & dev_buffer, (n + 1) * sizeof(float));
	/*
	for(int D=1;D< n; D*=2)
	{
		MultiBlockInclusivePrefixSum<<<gridDim,blockDim,blockDim*sizeof(float)>>>(D,input,dev_aux,n);
	}*/
	MultiBlockInclusivePrefixSum<<<gridDim,blockDim,blockDim*sizeof(float)>>>(input,dev_aux,n);
	#if(0)
	float * show;
	show = (float*)malloc((n) * sizeof(float));
	cudaMemcpy(show, input, (n)*sizeof(float),cudaMemcpyDeviceToHost);
	cout<<"GPU multiblock result: ";
	for(int i=0;i<n;i++)
	{
		 cout<<show[i]<<" ";
	}
	cout<<endl;
	free(show);
	#endif

	#if(0)
	float * show;
	show = (float*)malloc((gridDim) * sizeof(float));
	cudaMemcpy(show, dev_aux, (gridDim)*sizeof(float),cudaMemcpyDeviceToHost);
	cout<<"GPU aux result: ";
	for(int i=0;i<gridDim;i++)
	{
		 cout<<show[i]<<" ";
	}
	cout<<endl;
	free(show);
	#endif

	/*
	int specialGridDim = ceil((float)gridDim/(float)blockDim);
	for(int i=1;i< Log2(gridDim) + 1;i++)
	{
		D = pow(2,i-1);
		NaiveGPUinclusiveScan<<<specialGridDim,blockDim,gridDim*sizeof(float)>>>(D,dev_aux,dev_buffer,gridDim);
	}*/
	NaiveGPUinclusiveScan(dev_aux,gridDim);
	#if(0)
	float * show;
	show = (float*)malloc((gridDim) * sizeof(float));
	cudaMemcpy(show, dev_aux, (gridDim)*sizeof(float),cudaMemcpyDeviceToHost);
	cout<<"GPU scanned aux result: ";
	for(int i=0;i<gridDim;i++)
	{
		 cout<<show[i]<<" ";
	}
	cout<<endl;
	free(show);
	#endif

	//add aux to dev_in
	AddAuxToBlockedPrefixSum<<<gridDim+1,blockDim>>>(input,dev_aux,n);
	NaiveWriteToOutputExclusive<<<gridDim+1,blockDim>>>(input,output,n);

	#if(0)
	float * show;
	show = (float*)malloc((n+1) * sizeof(float));
	cudaMemcpy(show, output, (n+1)*sizeof(float),cudaMemcpyDeviceToHost);
	cout<<"GPU shared arbitrary ex scan result check (last value): ";
	cout<<show[n];
	for(int i=0;i<n;i++)
	{
		// cout<<show[i]<<" ";
	}
	cout<<endl;
	free(show);
	#endif

	//free memory
	cudaFree(dev_aux);
	cudaFree(dev_buffer);
}

__global__ void generateBoolArray(float * input, float * out, int n)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index <n) out[index] = (input[index] == 0.0f )? 0.0f: 1.0f;
}

__global__ void generateCompactArray(float * input, float * scannedBool, float * output, int n)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < n)
	{
		if(input[index] != 0.0f) output[(int)scannedBool[index]] = input[index];
	}
}
void GPUstreamCompaction(float * input, float * output, int n)
{


	int blockDim = BLOCKDIM;
	int gridDim = ceil((float)n/(float)blockDim);

	//allocate memory
	float * boolArray, * scannedBool;
	cudaMalloc((void**) & boolArray, n * sizeof(float));
	cudaMalloc((void**) & scannedBool, (n+1) * sizeof(float));

	cudaEvent_t start, stop,start2,stop2; 
	float time1, time2;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );

	generateBoolArray<<<gridDim,blockDim>>>(input,boolArray,n);
	GPUexclusiveScan(boolArray,scannedBool,n);

	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time1, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	//cout<<"GPU scan bool array: "<<time1<<" ms"<<endl;

	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
	cudaEventRecord( start2, 0 );
	generateCompactArray<<<gridDim,blockDim>>>(input, scannedBool, output, n);

	cudaEventRecord( stop2, 0 );
	cudaEventSynchronize( stop2 );
	cudaEventElapsedTime( &time2, start2, stop2 );
	cudaEventDestroy( start2 );
	cudaEventDestroy( stop2 );
	//cout<<"GPU generate compact array: "<<time2<<" ms"<<endl;
	cout<<"GPU stream compaction: "<<(time1 + time2)<<" ms"<<endl;

	//free memory
	cudaFree(boolArray);
	cudaFree(scannedBool);

}

void ThrustStreamCompaction(float * input, float * output, int n)
{


	int blockDim = BLOCKDIM;
	int gridDim = ceil((float)n/(float)blockDim);

	//allocate memory
	float * boolArray, * scannedBool;
	cudaMalloc((void**) & boolArray, n * sizeof(float));
	cudaMalloc((void**) & scannedBool, (n+1) * sizeof(float));

	cudaEvent_t start, stop,start2,stop2; 
	float time1, time2;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );

	generateBoolArray<<<gridDim,blockDim>>>(input,boolArray,n);
	thrust::device_ptr<float> thrustInput(boolArray);
	thrust::device_ptr<float> thrustRes(scannedBool);
	thrust::exclusive_scan(thrustInput, thrustInput + n+1, thrustRes);
	scannedBool = thrust::raw_pointer_cast(thrustRes);

	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time1, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	//cout<<"Thrust scan bool array: "<<time1<<" ms"<<endl;
	

	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
	cudaEventRecord( start2, 0 );
	generateCompactArray<<<gridDim,blockDim>>>(input, scannedBool, output, n);

	cudaEventRecord( stop2, 0 );
	cudaEventSynchronize( stop2 );
	cudaEventElapsedTime( &time2, start2, stop2 );
	cudaEventDestroy( start2 );
	cudaEventDestroy( stop2 );
	//cout<<"Thrust generate compact array: "<<time2<<" ms"<<endl;
	cout<<"Thrust GPU stream compaction: "<<(time1 + time2)<<" ms"<<endl;
	//free memory
	cudaFree(boolArray);
	cudaFree(scannedBool);

}
struct NoneZero
{
	__host__ __device__ bool operator()(const float num)
	{
		return num!=0.0f;
	}
};

float * ThrustStreamCompaction(float * input, int N, int & resN)
{
	resN = thrust::count_if(input, input + N, NoneZero());
	float * res = new float[resN];
	thrust::copy_if(input, input + N, res, NoneZero());
	return res;
}

int main(int argc, char** argv)
{
	//init
	float * in, *res, *dev_in, * dev_res;
	/*cout<<" enter array size N:"<<endl;
	int n(0);
	cin>>n;
	cin.ignore();*/
	int n = 10000000;

	in = (float*)malloc(n * sizeof(float));
	res = (float*)malloc((1+n) * sizeof(float));
	cudaMalloc((void**) & dev_in, n * sizeof(float));
	cudaMalloc((void**) & dev_res, (n+1) * sizeof(float));

	//load testing data 0 1 0 3 0 5 0 7.....................
	for(int i=0;i<n;i++)
	{
		in[i] = (i%2 != 0) ? float(i): 0.0f;
	}



	//print input
	cout<<"input: ";
	for(int i=0;i<n;i++)
	{
		//cout<<in[i]<<" ";
	}
	cout<<endl;


	//CPU scan////////////////////////////////////////////////////////////////////////////////
	clock_t startTime = clock();
	exPrefixSum(in,n,res);
	clock_t endTime = clock();
	double timeInMilli =( (double)endTime - (double)startTime)/((double) CLOCKS_PER_SEC)*1000.0000000f;
	cout<<"CPU scan runtime: "<<timeInMilli<<" ms"<<endl;
	///////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	//GPU scan/////////////////////////////////////////////////////////////////////
	cudaMemcpy(dev_in,in,n * sizeof(float),cudaMemcpyHostToDevice);
	cudaEvent_t start, stop,start2,stop2; 
	float time1, time2;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );
	NaiveGPUexclusiveScan(dev_in,dev_res,n);
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time1, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	cout<<"Naive GPU scan: "<<time1<<" ms"<<endl;

	cudaMemcpy(dev_in,in,n * sizeof(float),cudaMemcpyHostToDevice);
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
	cudaEventRecord( start2, 0 );
	GPUexclusiveScan(dev_in,dev_res,n);
	cudaEventRecord( stop2, 0 );
	cudaEventSynchronize( stop2 );
	cudaEventElapsedTime( &time2, start2, stop2 );
	cudaEventDestroy( start2 );
	cudaEventDestroy( stop2 );
	cout<<"GPU scan: "<<time2<<" ms"<<endl;
	///////////////////////////////////////////////////////////////////////////////////////////

	#if(0)//single block with shared memory ex prefix sum////////////////////////////////////////////////////////
	for(int i=1;i< Log2(n) + 1;i++)
	{
		int D = pow(2,i-1);
		SingleBlockExclusivePrefixSum<<<1,n+1,n*sizeof(float)>>>(D,dev_in,dev_res,n);
	}
	#endif////////////////////////////////////////////////////////////////////////////////////////////

	//CPU stream compact////////////////////////////////////////////////////////////////////////////////////////
	startTime = clock();
	CPUstreamCompaction(in,n,res);
	endTime = clock();
	timeInMilli =( (double)endTime - (double)startTime)/((double) CLOCKS_PER_SEC)*1000.0000000f;
	cout<<"CPU stream compact runtime: "<<timeInMilli<<" ms"<<endl;
	///////////////////////////////////////////////////////////////////////////////////////////////////////


	//MY GPU Stream compaction
	#if(1)///////////////////////////////////////////////////////////////////////////////////////////////////////
	cudaMemcpy(dev_in,in,n * sizeof(float),cudaMemcpyHostToDevice);
	GPUstreamCompaction(dev_in,dev_res,n);
	//ThrustStreamCompaction(dev_in,dev_res,n);
	
	cudaMemcpy(res, dev_res, (n+1)*sizeof(float),cudaMemcpyDeviceToHost);
	//cout<<"GPU stream compact result: ";
	//cout<<res[n/2 - 2]<<endl;

	#endif///////////////////////////////////////////////////////////////////////////////////////////////////////

	//GPU stream compaction using thrust
	#if(1)///////////////////////////////////////////////////////////////////////////////////////////////////////
	cudaMemcpy(dev_in,in,n * sizeof(float),cudaMemcpyHostToDevice);
	ThrustStreamCompaction(dev_in,dev_res,n);
	//ThrustStreamCompaction(dev_in,dev_res,n);
	
	cudaMemcpy(res, dev_res, (n+1)*sizeof(float),cudaMemcpyDeviceToHost);
	//cout<<"Thrust GPU stream compact result: ";
	//cout<<res[n/2 - 2]<<endl;

	#endif///////////////////////////////////////////////////////////////////////////////////////////////////////
	
	//Thrust stream compaction
	#if(0)///////////////////////////////////////////////////////////////////////////////////////////////////////
	cudaEvent_t start3, stop3;
	float time3;
	cudaEventCreate(&start3);
	cudaEventCreate(&stop3);
	cudaEventRecord( start3, 0 );

	int resLen(0);
	float * ThrustRes = ThrustStreamCompaction(in,n,resLen);

	
	cudaEventRecord( stop3, 0 );
	cudaEventSynchronize( stop3 );
	cudaEventElapsedTime( &time3, start3, stop3 );
	cudaEventDestroy( start3 );
	cudaEventDestroy( stop3 );
	cout<<"Thrust stream compaction: "<<time3<<" ms"<<endl;

	cout<<"Thrust GPU stream compact result: ";
	cout<<ThrustRes[resLen - 1]<<endl;
	free(ThrustRes);
	#endif///////////////////////////////////////////////////////////////////////////////////////////////////////

	//free memories
	free(in);
	free(res);
	cudaFree(dev_in);
	cudaFree(dev_res);
	
	cin.get();
    return 0;
}
