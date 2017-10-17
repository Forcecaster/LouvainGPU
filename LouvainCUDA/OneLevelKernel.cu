/*
*  Copyright 2017 Richard Forster
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*/

#include <fstream>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>

#include "StatusCUDA.h"
#include "cached_allocator.h"
#include "Graph.h"
#include "partition.h"

__global__ void neighbourCommunitySetKernel(StatusCUDA status, int limit)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= limit)
		return;

	//if (status.neighbourTarget[idx] == -1)
	//{
	//	printf("Possibly?\n");
	//	status.neighbourCommunities[idx] = -1;
	//	//status.neighbourWeights[idx] = 0;
	//}
	//else
	{
		status.neighbourCommunities[idx] = status.nodesToCom[status.hash_idx[idx]];
	}

	if (status.neighbourSource[idx] == status.neighbourTarget[idx])
	{
		status.neighbourWeights[idx] = 0;
	}
}

__device__ void remove(int node, int com, float weight, StatusCUDA &status)
{
	float gdegree = 0;
	float loop = 0;

	gdegree = status.gdegrees[node];

	loop = status.loops[node];

	atomicSub(&status.degreeUpd[com], gdegree);
	atomicSub(&status.internalUpd[com], weight + loop);
}

__device__ void insert(int node, int com, float weight, StatusCUDA &status)
{
	long long gdegree = 0;
	float loop = 0;

	gdegree = status.gdegrees[node];
	
	loop = status.loops[node];

	status.nodesToComNext[node] = com;

	atomicAdd(&status.degreeUpd[com], gdegree);
	atomicAdd(&status.internalUpd[com], weight + loop);
}

__global__ void communityInitKernel(StatusCUDA status, int limit)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	if (i >= limit)
		return;

	status.comSize[i] = 1;
	status.comSizeUpd[i] = 0;
	status.degreeUpd[i] = 0;
	status.internalUpd[i] = 0;
}

__global__ void communitySetKernel(StatusCUDA status, int limit)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	if (i >= limit)
		return;

	status.comSize[i] += status.comSizeUpd[i];
	status.degrees[i] += status.degreeUpd[i];
	status.internals[i] += status.internalUpd[i];

	status.comSizeUpd[i] = 0;
	status.degreeUpd[i] = 0;
	status.internalUpd[i] = 0;
}

__global__ void mergeInitKernel(StatusCUDA status, int limit)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	if (i >= limit)
		return;

	// status.comSize[i] = 1;
	status.comSizeUpd[i] = 0;
	status.degreeUpd[i] = 0;
	status.internalUpd[i] = 0;
}

__global__ void mergeSetKernel(StatusCUDA status, int limit)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	if (i >= limit)
		return;

	status.comSize[i] += status.comSizeUpd[i];
	status.degrees[i] += status.degreeUpd[i];
	status.internals[i] += status.internalUpd[i];

	status.comSizeUpd[i] = 0;
	status.degreeUpd[i] = 0;
	status.internalUpd[i] = 0;
}

__global__ void mergeIsolatedNodeKernel(StatusCUDA status, int limit)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= limit || status.nrOfNeighbours[i] == 0)
		return;
		
	int neighCommunitiesSize = status.nrOfNeighbours[i];

	if (status.comSize[status.nodesToCom[status.hash_idx[i]]] != 1)
	{
		return;
	}

	int comNode = status.nodesToCom[status.hash_idx[i]];
	int bestCom = comNode;
	float bestInc = 0;
	float degcTotW = (status.gdegrees[status.hash_idx[i]]) / (status.totalWeight * (float)2.0);

	float weight = 0;

	if (neighCommunitiesSize == 1)
	{
		if ((status.neighbourCommunities_local + status.startOfNeighbours[i])[0] != comNode && comNode < (status.neighbourCommunities_local + status.startOfNeighbours[i])[0])
		{
			remove(status.hash_idx[i], comNode, weight, status);

			insert(status.hash_idx[i], (status.neighbourCommunities_local + status.startOfNeighbours[i])[0], weight, status);

			atomicSub(&status.comSizeUpd[comNode], 1);
			atomicAdd(&status.comSizeUpd[(status.neighbourCommunities_local + status.startOfNeighbours[i])[0]], 1);
		}
	}
}

//__global__ void SetNewCommChild(StatusCUDA status, int nrOfNeighbours, int startOfNeighbours, int *neighbourCommunities_local, float *neighbourWeights_local, float *communityIncrease)
//{
//	__shared__ float sh_neighbourWeights_local[32];
//	__shared__ float sh_communityIncrease[32];
//	__shared__ int sh_neighbourCommunities_local[32];
//
//	int i = blockIdx.x *blockDim.x + threadIdx.x;
//
//	if (i >= nrOfNeighbours)
//		return;
//
//	sh_neighbourWeights_local[threadIdx.x] = neighbourWeights_local[i];
//	sh_communityIncrease[threadIdx.x] = communityIncrease[i];
//	sh_neighbourCommunities_local[threadIdx.x] = neighbourCommunities_local[i];
//
//	__syncthreads();
//
//	//for (int j = 0; j < status.nrOfNeighbours[i]; ++j)
//	{
//		if (neighbourWeights_local[i] == 0)
//			return;
//
//		int comNode = status.nodesToCom[status.hash_idx[i]];
//
//		float weight = 0;
//
//		int bestCom = comNode;
//		float bestInc = 0;
//
//		float incr = communityIncrease[i];
//
//		if ((incr > bestInc) || (incr == bestInc && incr != 0 && neighbourCommunities_local[i] < bestCom))
//		{
//			bestInc = incr;
//			bestCom = neighbourCommunities_local[i];
//			weight = neighbourWeights_local[i];
//		}
//	}
//}

//__global__ void SetNewCommunity(StatusCUDA status, int limit)
//{
//	int j = blockIdx.x*blockDim.x + threadIdx.x;
//
//	int i = blockIdx.x;
//
//	if (threadIdx.x >= 1 || status.nrOfNeighbours[i] == 0)
//		return;
//
//	int comNode = status.nodesToCom[status.hash_idx[i]];
//
//	float weight = 0;
//
//	int bestCom = comNode;
//	float bestInc = 0;
//
//	for (int j = 0; j < status.nrOfNeighbours[i]; ++j)
//	{
//		if ((status.neighbourWeights_local + status.startOfNeighbours[i])[j] == 0)
//			continue;
//
//		float incr = (status.communityIncrease + status.startOfNeighbours[i])[j];
//
//		if ((incr > bestInc) || (incr == bestInc && incr != 0 && (status.neighbourCommunities_local + status.startOfNeighbours[i])[j] < bestCom))
//		{
//			bestInc = incr;
//			bestCom = (status.neighbourCommunities_local + status.startOfNeighbours[i])[j];
//			weight = (status.neighbourWeights_local + status.startOfNeighbours[i])[j];
//		}
//	}
//
//	(status.neighbourWeights_local + status.startOfNeighbours[i])[0] = weight;
//	(status.neighbourCommunities_local + status.startOfNeighbours[i])[0] = bestCom;
//}

__global__ void SetNewCommunityWarpShuffle(StatusCUDA status, int limit)
{
	__shared__ int sh_bestCom;
	__shared__ float sh_bestInc;
	__shared__ float sh_weight;

	int j = blockIdx.x*blockDim.x + threadIdx.x;

	int i = blockIdx.x;

	if (/*j >= limit ||*/ status.nrOfNeighbours[i] == 0 || threadIdx.x >= status.nrOfNeighbours[i])
		return;

	if (threadIdx.x == 0)
	{
		//printf("%u\n", status.hash_idx[i]);

		int comNode = status.nodesToCom[status.hash_idx[i]];

		sh_weight = 0;

		sh_bestCom = comNode;
		sh_bestInc = 0;
	}

	int offset = 0;
	int remain = status.nrOfNeighbours[i];
	int size = remain;
	float my_weight = 0;
	float my_increase = 0;
	int my_community = 0;

	//if (i == 9)
	//{
		//if (threadIdx.x == 0)
	//	{
			//printf("size %i\n", size);

	//		printf("neighbour communities: \n");

	//		for (int k = 0; k < remain; ++k)
	//			printf("%i %f %f\n",(status.neighbourCommunities_local + status.startOfNeighbours[i])[k], (status.neighbourWeights_local + status.startOfNeighbours[i])[k], (status.communityIncrease + status.startOfNeighbours[i])[k]);
	//	}
	//}

	do
	{
		if (offset + threadIdx.x < size)
		{
			my_weight = (status.neighbourWeights_local + status.startOfNeighbours[i])[offset + threadIdx.x];
			
			my_community = (status.neighbourCommunities_local + status.startOfNeighbours[i])[offset + threadIdx.x];

			//my_increase = (status.communityIncrease + status.startOfNeighbours[i])[offset + threadIdx.x];

			if (my_weight != 0)
			{
				float degree = 0;

				float degcTotW = (status.gdegrees[(status.neighbourSource_local + status.startOfNeighbours[i])[offset + threadIdx.x]]) / (status.totalWeight * (float)2.0);

				degree = status.degrees[my_community];

				float weight = my_weight;

				float incr = weight - degree * degcTotW;

				my_increase = incr;
			}

			bool changed = false;

			for (int idx = 16; idx >= 1; idx /= 2)
			{
				//value += __shfl_xor(value, i, 32);
				float shifted_weight = __shfl_down(my_weight, idx);
				float shifted_increase = __shfl_down(my_increase, idx);
				int shifted_community = __shfl_down(my_community, idx);

				if (shifted_weight == 0)
					continue;

				if ((shifted_increase > my_increase) || (shifted_increase == my_increase && my_increase != 0 && shifted_community < my_community))
				{
					changed = true;

					my_increase = shifted_increase;
					my_community = shifted_community;
					my_weight = shifted_weight;
				}
			}
		}

		__syncthreads();

		if (threadIdx.x == 0)
		{
		//	if (i == 9)
		//	{
		//		printf("%f %i\n", my_weight, my_community);
		//	}

			//if (my_increase > sh_bestInc)
			//if (changed)
			{
				//if (i == 1)
				//{
				//	printf("change %f %f\n", my_increase, sh_bestInc);
				//}

				if ((my_increase > sh_bestInc) || (my_increase == sh_bestInc && sh_bestInc != 0 && my_community < sh_bestCom))
				{
					sh_bestCom = my_community;
					sh_bestInc = my_increase;
					sh_weight = my_weight;
				}
			}
		}

		__syncthreads();

		offset += 32;
		remain -= 32;

		//if (i == 1 && threadIdx.x == 0)
		//{
		//	printf("%i\n", remain);
		//}

	} while (remain > 0);

	__syncthreads();

	if (threadIdx.x == 0)
	{
		(status.neighbourWeights_local + status.startOfNeighbours[i])[0] = sh_weight;
		(status.neighbourCommunities_local + status.startOfNeighbours[i])[0] = sh_bestCom;
	}
}

__global__ void oneLevelKernel(StatusCUDA status, int limit)
{
	__shared__ float sh_totalWeight;

	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= limit)
		return;

	if (status.nrOfNeighbours[i] == 0)
		return;

	int hash_idx = status.hash_idx[i];

	int comNode = status.nodesToCom[hash_idx];

	float weight = (status.neighbourWeights_local + status.startOfNeighbours[i])[0];
	int bestCom = (status.neighbourCommunities_local + status.startOfNeighbours[i])[0];

	if (status.comSize[comNode] == 1 && status.comSize[bestCom] == 1 && bestCom > comNode)
	{
		bestCom = comNode;
	}

	if (bestCom != comNode)
	{
		remove(status.hash_idx[i], comNode, weight, status);
		insert(status.hash_idx[i], bestCom, weight, status);

		atomicSub(&status.comSizeUpd[comNode], 1);
		atomicAdd(&status.comSizeUpd[bestCom], 1);
	}
	else
		status.nodesToComNext[hash_idx] = status.nodesToCom[hash_idx];
}

#define __PASS_MAX -1
#define __MIN 0.0000001

__global__ void modularity(StatusCUDA status, int limit)
{
	__shared__ float links;

	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (threadIdx.x == 0)
		links = status.totalWeight;

	__syncthreads();

	if (i >= limit)
		return;

	float result = 0.0;

	float inDegree = status.internals[i];

	float degree = status.degrees[i];

	if (links > 0)
		result += inDegree / links - pow((degree / ((float)2.0 * links)), 2);

	status.result[i] = result;
}

struct binaryEq
{
	__device__ bool operator()
		(const thrust::tuple<int, int>& lhs, const thrust::tuple<int, int>& rhs)
	{
		return (thrust::get<0>(lhs) == thrust::get<0>(rhs)) && (thrust::get<1>(lhs) == thrust::get<1>(rhs));
	}
};

struct CommunityCompare
{
	__device__ bool operator()
		(const thrust::tuple<int, int, float>& lhs, const thrust::tuple<int, int, float>& rhs)
	{
		if (thrust::get<0>(lhs) == thrust::get<0>(rhs))
			if (thrust::get<2>(lhs) > thrust::get<2>(rhs))
				return true;
			else if (thrust::get<2>(lhs) == thrust::get<2>(rhs))
				if (thrust::get<1>(lhs) < thrust::get<1>(rhs))
					return true;
		return false;
	}
};

struct CounterOp
{
	__device__ bool operator()
		(const int& lhs, const int& rhs)
	{
		if (rhs == -1)
			return 0;
		return 1;
	}
};

//__global__ void ComputeNeighbourCommunityIncrement(StatusCUDA status, int limit)
//{
//	int i = blockIdx.x*blockDim.x + threadIdx.x;
//
//	if (i >= limit)
//	{
//		return;
//	}
//
//	if (status.neighbourWeights_local[i] == 0)
//	{
//		status.communityIncrease[i] = 0;
//		return;
//	}
//
//	float degree = 0;
//
//	float degcTotW = (status.gdegrees[status.hash_idx[i]]) / (status.totalWeight * (float)2.0);
//
//	degree = status.degrees[status.neighbourCommunities_local[i]];
//
//	float weight = status.neighbourWeights_local[i];
//
//	float incr = weight - degree * degcTotW;
//
//	status.communityIncrease[i] = incr;
//}

extern cached_allocator alloc;

float oneLevelCUDA(StatusCUDA &status, int nodeCount, int neighboursCount, int neighbourComputeCount, float &lower)
{
	int threads = 512;
	int blocks = nodeCount / threads;

	if (nodeCount % threads != 0)
		++blocks;

	int pass = 0;

	float curMod = -1;//modularity(status);
	float newMod = -1;

	clock_t w1, w2;

	w1 = clock();

	communityInitKernel << <blocks, threads >> >(status, nodeCount);
	cudaDeviceSynchronize();

	w2 = clock();

	//if (run == 0)
		//std::cout << "community init time: " << (double)(w2 - w1) / CLOCKS_PER_SEC << std::endl;

	//std::cout << "community init " << cudaGetErrorName(cudaGetLastError()) << std::endl;

	thrust::device_ptr<float> d_result(status.result);

	w1 = clock();

	static float *neighbourWeights_store = NULL;

	if (neighbourWeights_store == NULL)
		neighbourWeights_store = new float[neighbourComputeCount];

	cudaMemcpy(neighbourWeights_store, status.neighbourWeights, sizeof(float) * neighbourComputeCount, cudaMemcpyDeviceToHost);

	w2 = clock();

	//std::cout << "original weight copy time: " << (double)(w2 - w1) / CLOCKS_PER_SEC << std::endl;

	clock_t t1 = clock();

	static int onerun = 1;

	int go = 1;

	//cudppHashRetrieve(status.hash_table_handle, status.node,
	//	status.hash_idx, status.keys);

	//cudaThreadSynchronize();

	while (pass != __PASS_MAX)
	{
		++pass;

		w1 = clock();

		cudppHashRetrieve(status.hash_table_handle, status.neighbourTarget,
			status.hash_idx, neighbourComputeCount);

		cudaThreadSynchronize();

		w2 = clock();

		//std::cout << "hash retrieve: " << (double)(w2 - w1) / CLOCKS_PER_SEC << std::endl;

		w1 = clock();

		clock_t cuda1 = clock();

		int threadsNeighbour = 512;
		int blocksNeighbour = neighbourComputeCount / threadsNeighbour;

		if (neighbourComputeCount % threadsNeighbour != 0)
			++blocksNeighbour;

		neighbourCommunitySetKernel << <blocksNeighbour, threadsNeighbour>> > (status, neighbourComputeCount);
		cudaDeviceSynchronize();

		w2 = clock();

		//std::cout << "community set " << cudaGetErrorName(cudaGetLastError()) << std::endl;
		//if (run == 0)
			//std::cout << "optimized community set in while time: " << (double)(w2 - w1) / CLOCKS_PER_SEC << std::endl;

		clock_t thrust1 = clock();
		
		thrust::device_ptr<int> key1(status.neighbourSource);
		thrust::device_ptr<int> key2(status.neighbourCommunities);

		auto ziped_key = thrust::make_zip_iterator(thrust::make_tuple(key1, key2));

		clock_t thrust2 = clock();

		//if (run == 0)
			//std::cout << "device_ptr time: " << (double)(thrust2 - thrust1) / CLOCKS_PER_SEC << std::endl;

		thrust1 = clock();

		thrust::sort_by_key(thrust::cuda::par(alloc), ziped_key, ziped_key + neighboursCount, status.neighbourWeights);

		cudaDeviceSynchronize();

		thrust2 = clock();

		//if (run == 0)
			//std::cout << "sort time: " << (double)(thrust2 - thrust1) / CLOCKS_PER_SEC << std::endl;

		thrust1 = clock();

		thrust::device_ptr<int> key3(status.neighbourSource_local);
		thrust::device_ptr<int> key4(status.neighbourCommunities_local);
		auto ziped_key_reduced = thrust::make_zip_iterator(thrust::make_tuple(key3, key4));

		thrust2 = clock();

		//if (run == 0)
			//std::cout << "make zip time: " << (double)(thrust2 - thrust1) / CLOCKS_PER_SEC << std::endl;

		cudaMemcpy(status.neighbourCounts, status.neighbourTarget, sizeof(int) * neighbourComputeCount, cudaMemcpyDeviceToDevice);

		thrust1 = clock();

		thrust::pair<thrust::zip_iterator<thrust::tuple<thrust::device_ptr<int>, thrust::device_ptr<int>>>, thrust::zip_iterator<thrust::tuple<float*, int*>>> new_end;
		new_end = thrust::reduce_by_key(thrust::cuda::par(alloc), ziped_key, ziped_key + neighbourComputeCount, status.neighbourWeights, ziped_key_reduced, status.neighbourWeights_local, binaryEq());

		thrust2 = clock();

		cudaMemcpy(status.hash_idx_source, status.neighbourCounts, sizeof(int) * neighbourComputeCount, cudaMemcpyDeviceToDevice);

		cudaMemset(status.neighbourCounts, 0, sizeof(int) * neighbourComputeCount);

		//if (run == 0)
			//std::cout << "reduced_by_key time: " << (double)(thrust2 - thrust1) / CLOCKS_PER_SEC << std::endl;

		thrust1 = clock();

		//cudppHashRetrieve(status.hash_table_handle, status.neighbourSource_local,
		//	status.hash_idx, neighbourComputeCount);

		//cudaThreadSynchronize();

		thrust2 = clock();

		//std::cout << "incerement retrieve time: " << (double)(thrust2-thrust1) / CLOCKS_PER_SEC << std::endl;

		thrust1 = clock();

		//blocksNeighbour = (new_end.first.get_iterator_tuple().get_tail().get_head() - key4) / threadsNeighbour;

		//if ((new_end.first.get_iterator_tuple().get_tail().get_head() - key4) % threadsNeighbour != 0)
		//	++blocksNeighbour;

		//std::cout << "increment length: " << (new_end.first.get_iterator_tuple().get_tail().get_head() - key4) << std::endl;

		//ComputeNeighbourCommunityIncrement << <blocksNeighbour, threadsNeighbour >> > (status, (new_end.first.get_iterator_tuple().get_tail().get_head() - key4));
		//cudaDeviceSynchronize();

		thrust2 = clock();

		//std::cout << "increment " << cudaGetErrorName(cudaGetLastError()) << std::endl;

		//std::cout << "time for increment: " << (double)(thrust2 - thrust1) / CLOCKS_PER_SEC << std::endl;

		thrust1 = clock();

		thrust::transform(thrust::cuda::par(alloc), status.neighbourCounts, status.neighbourCounts + (new_end.first.get_iterator_tuple().get_tail().get_head() - key4), status.neighbourCommunities_local, status.neighbourCounts, CounterOp());

		thrust2 = clock();

		//std::cout << "transform " << cudaGetErrorName(cudaGetLastError()) << std::endl;

		//if (run == 0)
			//std::cout << "transform time: " << (double)(thrust2 - thrust1) / CLOCKS_PER_SEC << std::endl;

		//Reducing to get the number of neighbours for each unique source

		thrust1 = clock();

		cudaMemset(status.nrOfNeighbours, 0, sizeof(int) * status.keys);

		thrust2 = clock();

		//std::cout << "memset " << cudaGetErrorName(cudaGetLastError()) << std::endl;

		//if (run == 0)
			//std::cout << "temp_count memset time: " << (double)(thrust2 - thrust1) / CLOCKS_PER_SEC << std::endl;

		//std::cout << "temp cudamemset " << cudaGetErrorName(cudaGetLastError()) << std::endl;

		thrust1 = clock();

		thrust::pair<int*, int*> new_end3 = thrust::reduce_by_key(thrust::cuda::par(alloc), status.neighbourSource_local, status.neighbourSource_local + (new_end.first.get_iterator_tuple().get_tail().get_head() - key4), status.neighbourCounts, status.temp_source, status.nrOfNeighbours);

		thrust2 = clock();

		//std::cout << "reduce_by_key " << cudaGetErrorName(cudaGetLastError()) << std::endl;

		//if (run == 0)
			//std::cout << "reduce_by_key time: " << (double)(thrust2 - thrust1) / CLOCKS_PER_SEC << std::endl;

		//std::cout << "reduce_by_key " << cudaGetErrorName(cudaGetLastError()) << std::endl;

		thrust1 = clock();
		
		//cudaMemset(status.neighbourCounts + (new_end3.first - status.temp_source), 0, sizeof(int) * (status.keys - (new_end3.first - status.temp_source)));

		//std::cout << "Cudamemset+cpy " << cudaGetErrorName(cudaGetLastError()) << std::endl;
		
		//cudaMemcpy(status.nrOfNeighbours, status.temp_count, sizeof(int) * status.keys, cudaMemcpyDeviceToDevice);

		//std::cout << "Cudamemset+cpy " << cudaGetErrorName(cudaGetLastError()) << std::endl;

		thrust2 = clock();

		//std::cout << "memory " << cudaGetErrorName(cudaGetLastError()) << std::endl;

		//std::cout << "memcpy " << cudaGetErrorName(cudaGetLastError()) << std::endl;

		//if (run == 0)
			//std::cout << "memset+cpy time: " << (double)(thrust2 - thrust1) / CLOCKS_PER_SEC << std::endl;

		//std::cout << "Cudamemset+cpy " << cudaGetErrorName(cudaGetLastError()) << std::endl;

		thrust1 = clock();

		thrust::exclusive_scan(thrust::cuda::par(alloc), status.nrOfNeighbours, status.nrOfNeighbours + status.keys, status.startOfNeighbours, 0);

		thrust2 = clock();

		//std::cout << "exclusive " << cudaGetErrorName(cudaGetLastError()) << std::endl;

		//if (run == 0)
			//std::cout << "exclusive_scan time: " << (double)(thrust2 - thrust1) / CLOCKS_PER_SEC << std::endl;

		//std::cout << "exclusive_scan " << cudaGetErrorName(cudaGetLastError()) << std::endl;

		//cudaDeviceSynchronize();

		int threads = 32;
		int blocks = nodeCount;// / threads;

		//int *tComm, *h_tC;
		//float *tWeight, *h_tW;

		//h_tC = new int[blocks];
		//h_tW = new float[blocks];

		//cudaMalloc(&tComm, sizeof(int) * blocks);
		//cudaMalloc(&tWeight, sizeof(float) * blocks);

		//if (nodeCount % threads != 0)
		//	++blocks;

	
		cudppHashRetrieve(status.hash_table_handle, status.neighbourSource_local,
			status.hash_idx, neighbourComputeCount);

		cudaDeviceSynchronize();

		cudaMemcpy(status.neighbourSource_local, status.hash_idx, sizeof(int) * neighbourComputeCount, cudaMemcpyDeviceToDevice);

		w1 = clock();

		cudppHashRetrieve(status.hash_table_handle, status.temp_source,
			status.hash_idx, new_end3.first - status.temp_source);

		cudaThreadSynchronize();

		w2 = clock();

		//std::cout << "hash retrieve " << cudaGetErrorName(cudaGetLastError()) << std::endl;

		//std::cout << "hash retrieve: " << (double)(w2 - w1) / CLOCKS_PER_SEC << std::endl;

		static double time1 = 0;

		w1 = clock();

		SetNewCommunityWarpShuffle << <blocks, threads >> > (status, status.keys/*, tComm, tWeight*/);
		cudaDeviceSynchronize();

		w2 = clock();

		time1 += (double)(w2 - w1) / CLOCKS_PER_SEC;

		//cudaMemcpy(h_tC, tComm, sizeof(int) * blocks, cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_tW, tWeight, sizeof(float) * blocks, cudaMemcpyDeviceToHost);

		//if (go == 0)
		//{
	/*		std::ofstream out("setcommwarp-" + std::to_string(onerun) + "-" + std::to_string(go) + ".txt");

			for (int i = 0; i < blocks; ++i)
				out << i << " " << h_tC[i] << " " << h_tW[i] << std::endl;

			out.close();*/
		//}

		//std::cout << "setting new community in while with warp shuffle operations time: " << (double)(w2 - w1) / CLOCKS_PER_SEC << std::endl;
		//std::cout << time1 << std::endl;
		//std::cout << "setting new community in while with warp shuffle operations status: " << cudaGetErrorName(cudaGetLastError()) << std::endl;

		//static double time2 = 0;

		w1 = clock();

		//SetNewCommunity << <blocks, threads >> > (status, status.keys);
		cudaDeviceSynchronize();

		w2 = clock();

		//time2 += (double)(w2 - w1) / CLOCKS_PER_SEC;

		//cudaMemcpy(h_tC, tComm, sizeof(int) * blocks, cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_tW, tWeight, sizeof(float) * blocks, cudaMemcpyDeviceToHost);

		//if (go == 0)
		//{
			//std::ofstream out("setcomm-"+std::to_string(onerun)+"-"+std::to_string(go)+".txt");

			//for (int i = 0; i < blocks; ++i)
			//	out << i << " " << h_tC[i] << " " << h_tW[i] << std::endl;

			//out.close();

		//}

		++go;

		//std::cout << "new community " << cudaGetErrorName(cudaGetLastError()) << std::endl;

		//std::cout << "setting new community in while time: " << (double)(w2 - w1) / CLOCKS_PER_SEC << std::endl;
		//std::cout << time2 << std::endl;
		//std::cout << "new community set " << cudaGetErrorName(cudaGetLastError()) << std::endl;
			
		blocks = nodeCount / threads;

		if (nodeCount % threads != 0)
			++blocks;

		w1 = clock();

		oneLevelKernel << <blocks, threads >> >(status, nodeCount);
		cudaDeviceSynchronize();

		w2 = clock();

		//std::cout << "onelevelkernel " << cudaGetErrorName(cudaGetLastError()) << std::endl;

		//if (run == 0)
			//std::cout << "onelevelkernel in while time: " << (double)(w2 - w1) / CLOCKS_PER_SEC << std::endl;

		w1 = clock();

		modularity <<<blocks, threads>>>(status, nodeCount);
		cudaDeviceSynchronize();

		//std::cout << "modularity " << cudaGetErrorName(cudaGetLastError()) << std::endl;

		newMod = thrust::reduce(thrust::cuda::par(alloc), d_result, d_result + nodeCount, (float)0, thrust::plus<float>());

		//std::cout << "thrust reduce " << cudaGetErrorName(cudaGetLastError()) << std::endl;

		w2 = clock();

		//if (run == 0)
			//std::cout << "modularity in while time: " << (double)(w2 - w1) / CLOCKS_PER_SEC << std::endl;

		if ((newMod - curMod) < __MIN)
		{
			break;
		}

		curMod = newMod;
		if (curMod < lower)
			curMod = lower;

		communitySetKernel << <blocks, threads >> >(status, nodeCount);
		cudaDeviceSynchronize();

		cudaMemcpy(status.neighbourTarget, status.hash_idx_source, sizeof(int) * neighbourComputeCount, cudaMemcpyDeviceToDevice);
		cudaMemcpy(status.neighbourWeights, neighbourWeights_store, sizeof(float) * neighbourComputeCount, cudaMemcpyHostToDevice);

		//std::cout << "communitysetkernel " << cudaGetErrorName(cudaGetLastError()) << std::endl;

		int *tmp = status.nodesToComPrev;
		status.nodesToComPrev = status.nodesToCom; //Previous holds the current
		status.nodesToCom = status.nodesToComNext; //Current holds the chosen assignment
		status.nodesToComNext = tmp;
	}
	
	clock_t t2 = clock();

	++onerun;

	//if (run == 0)
		//std::cout << "while time: " << (double)(t2 - t1) / CLOCKS_PER_SEC << std::endl;

	w1 = clock();

	mergeInitKernel << <blocks, threads >> >(status, nodeCount);
	cudaDeviceSynchronize();

	mergeIsolatedNodeKernel << <blocks, threads >> >(status, nodeCount);
	cudaDeviceSynchronize();

	mergeSetKernel << <blocks, threads >> >(status, nodeCount);
	cudaDeviceSynchronize();

	w2 = clock();

	//std::cout << "merge time: " << (double)(w2 - w1) / CLOCKS_PER_SEC << std::endl;

	int *tmp = status.nodesToComPrev;
	status.nodesToComPrev = status.nodesToCom; //Previous holds the current
	status.nodesToCom = status.nodesToComNext; //Current holds the chosen assignment
	status.nodesToComNext = tmp;

	clock_t thrust1 = clock();

	modularity << <blocks, threads >> >(status, nodeCount);
	cudaDeviceSynchronize();

	newMod = thrust::reduce(thrust::cuda::par(alloc), d_result, d_result + nodeCount, (float)0, thrust::plus<float>());

	clock_t thrust2 = clock();

	//if (run == 0)
		//std::cout << "modularity time: " << (double)(thrust2 - thrust1) / CLOCKS_PER_SEC << std::endl;

	//std::cout << "modularity of onelevel: " << newMod << std::endl;

	return newMod;
}

struct ReducePlus
{
	__device__ bool operator()
		(const thrust::tuple<int, int, int>& lhs, const thrust::tuple<int, int, int>& rhs)
	{
		return (thrust::get<0>(lhs) == thrust::get<0>(rhs)) && (thrust::get<1>(lhs) == thrust::get<1>(rhs));
	}
};

float modularityCUDA(StatusCUDA &status, int nodeCount)
{
	int threads = 512;
	int blocks = nodeCount / threads;

	if (nodeCount % threads != 0)
		++blocks;

	thrust::device_ptr<float> d_result(status.result);

	modularity << <blocks, threads >> >(status, nodeCount);
	cudaDeviceSynchronize();

	float newMod = thrust::reduce(thrust::cuda::par(alloc), d_result, d_result + nodeCount, (float)0, thrust::plus<float>());

	return newMod;
}