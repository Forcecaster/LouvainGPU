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

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>

#include "cached_allocator.h"

#include "StatusCUDA.h"
#include "CUDAGraph.h"

#include <iostream>
#include <fstream>
#include <time.h>

extern cached_allocator alloc;

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

void CalculateNeighbour(StatusCUDA &status, CUDAGraph &graph)
{
	clock_t t1 = clock();

	status.totalWeight = thrust::reduce(thrust::cuda::par(alloc), status.neighbourWeights, status.neighbourWeights + graph.edgeSize, (float)0, thrust::plus<float>());

	clock_t t2 = clock();

	//std::cout << "totalweight time " << (double)(t2 - t1) / CLOCKS_PER_SEC << std::endl;
	
	//std::cout << graph.nodesSize << " " << graph.edgeSize << std::endl;

	t1 = clock();

	//cudaMemcpy(status.neighbourSource, graph.edgeSource, sizeof(int) * graph.edgeSize, cudaMemcpyHostToDevice);
	//cudaMemcpy(status.neighbourTarget, graph.edgeTarget, sizeof(int) * graph.edgeSize, cudaMemcpyHostToDevice);
	//cudaMemcpy(status.neighbourWeights, graph.edgeWeight, sizeof(float) * graph.edgeSize, cudaMemcpyHostToDevice);

	t2 = clock();

	//std::cout << (double)(t2 - t1) / CLOCKS_PER_SEC << std::endl;

	t1 = clock();

	cudaMemcpy(status.neighbourSource + graph.edgeSize, status.neighbourTarget, sizeof(int) * graph.edgeSize, cudaMemcpyDeviceToDevice);
	cudaMemcpy(status.neighbourTarget + graph.edgeSize, status.neighbourSource, sizeof(int) * graph.edgeSize, cudaMemcpyDeviceToDevice);
	cudaMemcpy(status.neighbourWeights + graph.edgeSize, status.neighbourWeights, sizeof(float) * graph.edgeSize, cudaMemcpyDeviceToDevice);

	t2 = clock();
	//std::cout << (double)(t2 - t1) / CLOCKS_PER_SEC << std::endl;

	graph.nrOfAllNeighbours = graph.edgeSize * 2;
	
	thrust::device_ptr<int> key1(status.neighbourSource);
	thrust::device_ptr<int> key2(status.neighbourTarget);

	auto ziped_key = thrust::make_zip_iterator(thrust::make_tuple(key1, key2));

	t1 = clock();

	thrust::sort_by_key(thrust::cuda::par(alloc), ziped_key, ziped_key + graph.edgeSize * 2, status.neighbourWeights);
	
	t2 = clock();
	//std::cout << "sort " << (double)(t2 - t1) / CLOCKS_PER_SEC << std::endl;

	thrust::pair<int*, int*> new_end;

	cudaMemset(status.neighbourCounts, 0, sizeof(int) * graph.nrOfAllNeighbours);

	t1 = clock();

	thrust::transform(thrust::cuda::par(alloc), status.neighbourCounts, status.neighbourCounts + graph.nrOfAllNeighbours, status.neighbourTarget, status.neighbourCounts, CounterOp());
	
	t2 = clock();
	//std::cout << (double)(t2 - t1) / CLOCKS_PER_SEC << std::endl;

	t1 = clock();

	new_end = thrust::reduce_by_key(thrust::cuda::par(alloc), status.neighbourSource, status.neighbourSource + graph.nrOfAllNeighbours, status.neighbourCounts, status.neighbourSource_local, status.neighbourCommunities);

	t2 = clock();
	//std::cout << "reduce " << (double)(t2 - t1) / CLOCKS_PER_SEC << std::endl;

	//std::cout << new_end.first - status.neighbourSource_local << std::endl;

	cudaMemset(status.neighbourCommunities + (new_end.first - status.neighbourSource_local), 0, sizeof(int) * (graph.nodesSize - (new_end.first - status.neighbourSource_local)));

	cudaMemcpy(status.nrOfNeighbours, status.neighbourCommunities, sizeof(int) * graph.nodesSize, cudaMemcpyDeviceToDevice);

	t1 = clock();

	thrust::exclusive_scan(thrust::cuda::par(alloc), status.nrOfNeighbours, status.nrOfNeighbours + graph.nodesSize, status.startOfNeighbours, 0);

	t2 = clock();

	//std::cout << (double)(t2 - t1) / CLOCKS_PER_SEC << std::endl;
}