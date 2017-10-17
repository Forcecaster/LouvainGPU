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
#include "Partition.h"

extern cached_allocator alloc;

__global__ void renumberKernel(StatusCUDA status, Partition p, int limit)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= limit)
		return;

	p.partition.node[i] = status.node[i];

	//These will be in hash table
//	p.partition.community[i] = p.commMap[status.nodesToCom[i]];

	//p.nodeCommMap[p.partition.node[i]] = p.partition.community[i];
}

__global__ void setHashIdx(StatusCUDA status, int limit)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx >= limit)
		return;

	//Reuse hash_idx so we don't have to allocate more memory for the initializion
	status.hash_idx[idx] = idx;
}

struct equal_to_unique
{
	__device__ bool operator ()
		(const thrust::tuple<int, int>& lhs, const thrust::tuple<int, int>& rhs)
	{
		return (thrust::get<0>(lhs) == thrust::get<0>(rhs));
	}
};

extern int first;

Partition renumberCUDA(CUDPPHandle &theCudpp, StatusCUDA &status)
{
	clock_t t1 = clock();

	//std::cout << status.keys << std::endl;

	Community ret;
	ret.community = new int[status.keys];
	ret.node = new int[status.keys];
	std::map<int, int> new_values;

	Partition p;
	

	p.count = status.keys;
	p.partition = ret;

	clock_t t2 = clock();

	//std::cout << "community creation time: " << (double)(t2 - t1) / CLOCKS_PER_SEC << std::endl;

	int threads = 512;
	int blocks = status.keys / threads;

	if (status.keys % threads != 0)
		++blocks;

	//cout << "sethashidx" << std::endl;

	t1 = clock();

	setHashIdx << <blocks, threads >> > (status, status.keys);
	cudaDeviceSynchronize();

	t2 = clock();

	//std::cout << "sethashidx time: " << (double)(t2 - t1) / CLOCKS_PER_SEC << std::endl;

	//cout << cudaGetErrorName(cudaGetLastError()) << std::endl;

	//cout << "memcpy" << std::endl;

	//cudaMemcpy(status.nodesToComNext, status.nodesToCom, sizeof(int) * status.keys, cudaMemcpyDeviceToDevice);
	//t1 = clock();

	//thrust::sort(thrust::device, status.nodesToComNext, status.nodesToComNext + status.keys);
	//thrust::unique(thrust::device, status.nodesToComNext, status.nodesToComNext + status.keys);

	//cudaDeviceSynchronize();

	//t2 = clock();

	//std::cout << "renumber sort on only nodesToCom time: " << (double)(t2 - t1) / CLOCKS_PER_SEC << std::endl;

	cudaMemcpy(status.nodesToComPrev, status.nodesToCom, sizeof(int) * status.keys, cudaMemcpyDeviceToDevice);

	t1 = clock();

	auto ziped_value = thrust::make_zip_iterator(thrust::make_tuple(status.node, status.hash_idx));

	thrust::sort_by_key(thrust::cuda::par(alloc), status.nodesToComPrev, status.nodesToComPrev + status.keys, ziped_value);

	cudaMemcpy(status.nodesToCom, status.nodesToComPrev, sizeof(int) * status.keys, cudaMemcpyDeviceToDevice);

	auto ziped_key = thrust::make_zip_iterator(thrust::make_tuple(status.nodesToComPrev, status.hash_idx));

	thrust::zip_iterator<thrust::tuple<int*, unsigned int*>> new_end = thrust::unique(thrust::device, ziped_key, ziped_key + status.keys, equal_to_unique());

	int size = new_end.get_iterator_tuple().get_tail().get_head() - status.hash_idx;

	thrust::sort_by_key(thrust::cuda::par(alloc), status.hash_idx, status.hash_idx + size, status.nodesToComPrev);

	cudaDeviceSynchronize();

	t2 = clock();

	//std::cout << "renumber sort+unique time: " << (double)(t2 - t1) / CLOCKS_PER_SEC << std::endl;

	////cout << "unique elements in renumber: " << status.keys << " " << size << " " << new_values.size() << std::endl;

	if (first == 1)
	{
		status.commMap_hash_table_config.type = CUDPP_BASIC_HASH_TABLE;
		status.commMap_hash_table_config.kInputSize = size;
		status.commMap_hash_table_config.space_usage = 1.05;

		CUDPPResult result = cudppHashTable(theCudpp, &status.commMap_hash_table_handle, &status.commMap_hash_table_config);
		if (result != CUDPP_SUCCESS)
		{
			fprintf(stderr, "Error in cudppHashTable call in"
				"testHashTable (make sure your device is at"
				"least compute version 2.0\n");
		}
	}

	t1 = clock();

	setHashIdx << <blocks, threads >> > (status, size);
	cudaDeviceSynchronize();

	cudppHashInsert(status.commMap_hash_table_handle, status.nodesToComPrev,
		status.hash_idx, size);

	cudaThreadSynchronize();

	t2 = clock();

	//std::cout << "renumber commMap time: " << (double)(t2 - t1) / CLOCKS_PER_SEC << std::endl;

	if (first == 1)
	{
		status.nodeCommMap_hash_table_config.type = CUDPP_BASIC_HASH_TABLE;
		status.nodeCommMap_hash_table_config.kInputSize = status.keys;
		status.nodeCommMap_hash_table_config.space_usage = 1.05;

		CUDPPResult result = cudppHashTable(theCudpp, &status.nodeCommMap_hash_table_handle, &status.nodeCommMap_hash_table_config);
		if (result != CUDPP_SUCCESS)
		{
			fprintf(stderr, "Error in cudppHashTable call in"
				"testHashTable (make sure your device is at"
				"least compute version 2.0\n");
		}

		first = 0;
	}

	t1 = clock();

	cudppHashRetrieve(status.commMap_hash_table_handle, status.nodesToCom,
		status.hash_idx, status.keys);

	cudaThreadSynchronize();

	t2 = clock();

	//std::cout << "renumber retrieve time: " << (double)(t2-t1) / CLOCKS_PER_SEC << std::endl;

	t1 = clock();

	cudaMemcpy(p.partition.community, status.hash_idx, sizeof(int) * status.keys, cudaMemcpyDeviceToHost);
	cudaMemcpy(p.partition.node, status.node, sizeof(int) * status.keys, cudaMemcpyDeviceToHost);

	t2 = clock();

	//std::cout << "partition back copy time: " << (double)(t2 - t1) / CLOCKS_PER_SEC << std::endl;

	t1 = clock();

	cudppHashInsert(status.nodeCommMap_hash_table_handle, status.node,
		status.hash_idx, status.keys);

	cudaThreadSynchronize();

	t2 = clock();

	//std::cout << "renumber insert time: " << (double)(t2 - t1) / CLOCKS_PER_SEC << std::endl;

	return p;
}