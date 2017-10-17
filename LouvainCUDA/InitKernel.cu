/*
*  Copyright 2017 Richard Forster
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*/

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "StatusCUDA.h"
#include "CUDAGraph.h"
#include "cached_allocator.h"

#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include <iostream>
#include <fstream>

extern cached_allocator alloc;

__global__ void setHashKey(StatusCUDA status, int limit)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx >= limit)
		return;

	//Reuse hash_idx so we don't have to allocate more memory for the initializion
	status.hash_idx[idx] = status.neighbourSource[status.startOfNeighbours[idx]];
}

__device__ float getDegreeForNodeAt(StatusCUDA &status, int i, int &degIdx)
{
	//***New degree computation***

	float weight = 0;

	for (int j = 0; j < status.nrOfNeighbours[i]; ++j)
	{
		int *temp = status.startOfNeighbours + i;
		int temp2 = j;

		float value = status.neighbourWeights[status.startOfNeighbours[i] + j];
		if (value == 0)
			value = 1;

		if (status.neighbourSource[status.startOfNeighbours[i] + j] == status.neighbourTarget[status.startOfNeighbours[i] + j])
			value *= 2;

		weight += value;
	}

	//Will be in a hash table, the idx will come from there
	degIdx = status.comSize[i];

	return weight;
}

__device__ float getEdgeWeight(StatusCUDA &status, int node1, int node2, int &loopIdx)
{
	//***Proposed new getEdgeWeight currently with erronous indexing***

	float weight = 0;

	for (int j = 0; j < status.nrOfNeighbours[node1]; ++j)
	{
		if (status.neighbourSource[status.startOfNeighbours[node1] + j] == status.neighbourTarget[status.startOfNeighbours[node2] + j])
			weight = status.neighbourWeights[status.startOfNeighbours[node1] + j];
	}

	return weight;
}

__global__ void initCommsKernel(StatusCUDA status,/* CUDAGraph graph,*/ int limit)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= limit)
		return;

	status.degrees[i] = 0;
	status.gdegrees[i] = 0;
	status.loops[i] = 0;
	status.internals[i] = 0;

	__syncthreads();

	//status.node[i] = graph.node[i];
	status.hash_idx[i] = i;

	status.nodesToCom[i] = i;
	status.nodesToComPrev[i] = i;
	status.nodesToComNext[i] = i;
}

__global__ void initValuesKernel(StatusCUDA status,/* CUDAGraph graph,*/ int limit)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= limit)
		return;

	int degIdx = i;
	float deg = 0;

	if (status.nrOfNeighbours[i] == 0)
		return;

	deg = getDegreeForNodeAt(status, i, degIdx);

	//if (deg < 0)
	//	throw("A node has a negative degree. Use positive weights.");

	status.degrees[degIdx] = deg;
	status.gdegrees[degIdx] = deg;

	int loopIdx = i;

	status.loops[loopIdx] = getEdgeWeight(status, i, i, loopIdx);
	status.internals[i] = status.loops[i];
}

void initCUDA(StatusCUDA &status, CUDAGraph &graph)
{
	int threads = 512;
	int blocks = graph.nodesSize / threads;

	if (graph.nodesSize % threads != 0)
		++blocks;

	status.keys = graph.nodesSize;

	clock_t t1 = clock();

	initCommsKernel << <blocks, threads >> > (status, graph.nodesSize);
	cudaDeviceSynchronize();

	clock_t t2 = clock();

	//std::cout << "initCommsKernel " << (double)(t2 - t1) / CLOCKS_PER_SEC << std::endl;

	t1 = clock();

	//cudaMemcpy(status.node, graph.node, sizeof(int) * graph.nodesSize, cudaMemcpyDeviceToDevice);

	t2 = clock();

	//std::cout << "memcpy " << (double)(t2 - t1) / CLOCKS_PER_SEC << std::endl;

	//int *temp = new int[status.keys];

	t1 = clock();

	cudppHashInsert(status.hash_table_handle, status.node,
		status.hash_idx, status.keys);

	cudaDeviceSynchronize();

	//std::cout << cudaGetErrorName(cudaGetLastError()) << std::endl;

	t2 = clock();

	cudppHashRetrieve(status.hash_table_handle, status.node,
		status.hash_idx, status.keys);

	cudaThreadSynchronize();

	//std::cout << "hashinsert " << (double)(t2 - t1) / CLOCKS_PER_SEC << std::endl;

	t1 = clock();

	setHashKey << <blocks, threads >> > (status, graph.nodesSize);
	cudaDeviceSynchronize();

	t2 = clock();

	//cudaMemcpy(temp, status.hash_idx, sizeof(int) * graph.nodesSize, cudaMemcpyDeviceToHost);

	//std::cout << "sethashkey " << (double)(t2 - t1) / CLOCKS_PER_SEC << std::endl;

	t1 = clock();

	cudppHashRetrieve(status.hash_table_handle, status.hash_idx,
		status.comSize, status.keys);

	cudaThreadSynchronize();

	t2 = clock();

	//std::cout << "hash retrieve " << (double)(t2 - t1) / CLOCKS_PER_SEC << std::endl;

	t1 = clock();

	initValuesKernel << <blocks, threads >> > (status, graph.nodesSize);
	cudaDeviceSynchronize();

	t2 = clock();

	static int go = 0;

	//if (go == 0)
	//{
	//	float *nr = new float[graph.nodesSize];

	//	cudaMemcpy(nr, status.degrees, sizeof(float) * graph.nodesSize, cudaMemcpyDeviceToHost);

	//	std::ofstream out("init_degrees.txt");

	//	for (int i = 0; i < graph.nodesSize; ++i)
	//		out << nr[i] << std::endl;

	//	out.close();
	//	++go;
	//}

	//std::cout << "initValuesKernel " << (double)(t2 - t1) / CLOCKS_PER_SEC << std::endl;
}