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
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>

#include "cached_allocator.h"

#include "StatusCUDA.h"
#include "CUDAGraph.h"
#include "Partition.h"

extern cached_allocator alloc;

__global__ void inducedGraphEdgeGenerationKernel(StatusCUDA status, CUDAGraph graph, Partition d_partition, int limit)
{
	int edgeIdx = blockIdx.x*blockDim.x + threadIdx.x;

	if (edgeIdx >= limit)
		return;

	float w_prec = 0;
	float weight = 0;

	if (status.neighbourWeights[edgeIdx] == 0)
		++status.neighbourWeights[edgeIdx];

	int com1 = status.hash_idx_source[edgeIdx];
	int com2 = status.hash_idx_target[edgeIdx];

	status.neighbourSource[edgeIdx] = com1;
	status.neighbourTarget[edgeIdx] = com2;

	//printf("Comms in edgegeneration: %i %i\n", com1, com2);
}

void inducedGraphCUDA(CUDPPHandle &theCudpp, StatusCUDA &status, CUDAGraph &cuGraph, Partition &d_partition)
{
	thrust::device_ptr<unsigned int> key(status.hash_idx);
	//thrust::device_ptr<unsigned int> value(status.node);

	//std::cout << d_partition.count << std::endl;

	thrust::sort/*_by_key*/(thrust::cuda::par(alloc), key, key + d_partition.count/*, value*/);

	unsigned int* result = thrust::unique_copy(thrust::cuda::par(alloc), key, key + d_partition.count, cuGraph.node);
	cuGraph.nodesSize = result - cuGraph.node;

	//thrust::sort_by_key(thrust::cuda::par(alloc), value, value + d_partition.count, key);

	int edgeThreads = 32;
	int blocks = cuGraph.edgeSize / edgeThreads;

	if (cuGraph.edgeSize % edgeThreads != 0)
		++blocks;

	cudppHashRetrieve(status.nodeCommMap_hash_table_handle, status.neighbourSource,
		status.hash_idx_source, cuGraph.edgeSize);

	cudaThreadSynchronize();

	cudppHashRetrieve(status.nodeCommMap_hash_table_handle, status.neighbourTarget,
		status.hash_idx_target, cuGraph.edgeSize);

	cudaThreadSynchronize();

	inducedGraphEdgeGenerationKernel << <blocks, edgeThreads >> > (status, cuGraph, d_partition, cuGraph.edgeSize);

	cudaDeviceSynchronize();

	thrust::device_ptr<int> key1(status.neighbourSource);
	thrust::device_ptr<int> key2(status.neighbourTarget);

	thrust::device_ptr<float> value2(status.neighbourWeights);

	auto ziped_key = thrust::make_zip_iterator(thrust::make_tuple(key1, key2));

	thrust::sort_by_key(thrust::cuda::par(alloc), ziped_key, ziped_key + cuGraph.edgeSize, value2);

	thrust::device_ptr<int> key_r1(cuGraph.edgeSource_temp);
	thrust::device_ptr<int> key_r2(cuGraph.edgeTarget_temp);

	auto ziped_key_reduced = thrust::make_zip_iterator(thrust::make_tuple(key_r1, key_r2));

	thrust::pair<thrust::zip_iterator<thrust::tuple<thrust::device_ptr<int>, thrust::device_ptr<int>>>, float*> new_end;
	new_end = thrust::reduce_by_key(thrust::cuda::par(alloc), ziped_key, ziped_key + cuGraph.edgeSize, status.neighbourWeights, ziped_key_reduced, cuGraph.edgeWeight_temp);

	int count = new_end.first.get_iterator_tuple().get_head() - key_r1;

	clock_t t1 = clock();

	cudaMemcpy(status.neighbourSource, cuGraph.edgeSource_temp, sizeof(int) * count, cudaMemcpyDeviceToDevice);
	cudaMemcpy(status.neighbourTarget, cuGraph.edgeTarget_temp, sizeof(int) * count, cudaMemcpyDeviceToDevice);
	cudaMemcpy(status.neighbourWeights, cuGraph.edgeWeight_temp, sizeof(float) * count, cudaMemcpyDeviceToDevice);

	clock_t t2 = clock();

	//std::cout << "induced memcpy time: " << (double)(t2 - t1) / CLOCKS_PER_SEC << std::endl;

	cuGraph.edgeSize = count;
}