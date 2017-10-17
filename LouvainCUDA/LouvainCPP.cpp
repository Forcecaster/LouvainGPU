/*
*  Copyright 2017 Richard Forster
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*/

#include "community.h"
#include "Graph.h"
#include "Status.h"
#include "StatusCUDA.h"
#include "CUDAGraph.h"
#include "partition.h"
#include "cached_allocator.h"
//#include "IntroSort.h"
//#include "vectorSummation.h"
#include "utils.h"
#include <set>
#include <map>
#include <utility>
#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>

#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <list>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <thrust/reduce.h>
#include <thrust/unique.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

//#include <cudpp_hash.h>

#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/class.hpp>

using namespace std;

#define __PASS_MAX -1
#define __MIN 0.0000001

cached_allocator alloc;

float modularity(Status status, long localWeight)
{
	float links = status.totalWeight;
        
	float result = 0.0;

	for (int i = 0; i < status.keys; ++i)
	{
        if (status.node[i] == -1)
            continue;
        
		float inDegree = status.internals[i];
        
		float degree = status.degrees[i];

		if (links > 0)
			// result += inDegree / (float)localWeight - pow(degree, 2) / (float)localWeight / (float)localWeight;
			result += inDegree / links - pow((degree / ((float)2.0 * links)), 2);
	}
  
	return result;
}

Partition partitionAtLevel(std::vector<Partition> &dendogram, int level)
{
	Partition partition = dendogram[0];

	for (int i = 1; i < level + 1; ++i)
	{
		for (int j = 0; j < partition.count; ++j)
		{
			int node = partition.partition.node[j];
			int com = partition.partition.community[j];

			int newCom = -1;

			for (int k = 0; k < dendogram[i].count; ++k)
			{
				if (dendogram[i].partition.node[k] == com)
					newCom = dendogram[i].partition.community[k];
			}

			partition.partition.community[j] = newCom;
		}
	}

	return partition;
}

Partition getPartitions(std::vector<Partition> &dendogram, int level = 0)
{
	if ((level < 0) || (level > dendogram.size() - 1))
		throw "Invalid argument: level is not between 1 and len(dendogram) - 1 included.";

	return partitionAtLevel(dendogram, level || dendogram.size() - 1);
}

int countPartitions(Partition partitions)
{
	int max = 0;

	for (int i = 0; i < partitions.count; ++i)
	{
		if (partitions.partition.community[i] > max)
			max = partitions.partition.community[i];
	}

	return 1 + max;
}

extern float modularityCUDA(StatusCUDA &status, int nodeCount);
extern float oneLevelCUDA(StatusCUDA &status, int nodeCount, int neighboursCount, int neighbourComputeCount, float &lower);
extern void CalculateNeighbour(StatusCUDA &status, CUDAGraph &graph);
extern void initCUDA(StatusCUDA &status, CUDAGraph &graph);
extern Partition renumberCUDA(CUDPPHandle &theCudpp, StatusCUDA &status);
extern void inducedGraphCUDA(CUDPPHandle &theCudpp, StatusCUDA &status, CUDAGraph &cuGraph, Partition &d_partition);

void AllocateMemory(StatusCUDA &d_status, Graph &graph, CUDAGraph &cuGraph)
{
	cudaMalloc(&d_status.uint_memory, sizeof(unsigned int) * (graph.getNodeSize() + (graph.getEdgeSize() * 4)));

	cout << cudaGetErrorName(cudaGetLastError()) << endl;
	cout << sizeof(unsigned int) * (graph.getNodeSize() + (graph.getEdgeSize() * 4)) / 1024 / 1024 << endl;

	cudaMalloc(&d_status.int_memory, sizeof(int) * (graph.getNodeSize() * 12 + (graph.getEdgeSize() * 2) * 5));

	cout << cudaGetErrorName(cudaGetLastError()) << endl;
	cout << sizeof(int) * (graph.getNodeSize() * 12 + (graph.getEdgeSize() * 2) * 5) / 1024 / 1024 << endl;

	cudaMalloc(&d_status.float_memory, sizeof(float) * (graph.getNodeSize() * 3 + (graph.getEdgeSize() * 2) * 2));

	cout << cudaGetErrorName(cudaGetLastError()) << endl;
	cout << sizeof(float) * (graph.getNodeSize() * 3 + (graph.getEdgeSize() * 2) * 2) / 1024 / 1024 << endl;

	d_status.node = d_status.uint_memory;
	d_status.hash_idx = d_status.uint_memory + graph.getNodeSize();
	d_status.hash_idx_source = d_status.uint_memory + graph.getNodeSize() + (graph.getEdgeSize() * 2);
	d_status.hash_idx_target = d_status.hash_idx;// d_status.uint_memory + graph.getNodeSize() + (graph.getEdgeSize() * 3);

	d_status.nodesToCom = d_status.int_memory;
	d_status.nodesToComPrev = d_status.int_memory + graph.getNodeSize();
	d_status.nodesToComNext = d_status.int_memory + graph.getNodeSize() * 2;
	d_status.internals = d_status.int_memory + graph.getNodeSize() * 3;
	d_status.degrees = d_status.int_memory + graph.getNodeSize() * 4;

	d_status.bestCom = d_status.int_memory + graph.getNodeSize() * 5;

	d_status.comSize = d_status.int_memory + graph.getNodeSize() * 6;
	d_status.comSizeUpd = d_status.int_memory + graph.getNodeSize() * 7;
	d_status.degreeUpd = d_status.int_memory + graph.getNodeSize() * 8;
	d_status.internalUpd = d_status.int_memory + graph.getNodeSize() * 9;

	d_status.nrOfNeighbours = d_status.int_memory + graph.getNodeSize() * 10;
	d_status.startOfNeighbours = d_status.int_memory + graph.getNodeSize() * 11;

	cuGraph.node = d_status.node;// d_status.int_memory + graph.getNodeSize() * 12;

	d_status.neighbourSource_local = d_status.int_memory + graph.getNodeSize() * 12;
	d_status.neighbourCounts = d_status.int_memory + graph.getNodeSize() * 12 + (graph.getEdgeSize() * 2);
	d_status.neighbourCommunities_local = d_status.int_memory + graph.getNodeSize() * 12 + ((graph.getEdgeSize() * 2) * 2);
	d_status.neighbourCommunities = d_status.int_memory + graph.getNodeSize() * 12 + ((graph.getEdgeSize() * 2) * 3);
	d_status.neighbourSource = d_status.int_memory + graph.getNodeSize() * 12 + ((graph.getEdgeSize() * 2) * 4);
	d_status.neighbourTarget = d_status.neighbourCommunities_local;// d_status.int_memory + graph.getNodeSize() * 12 + ((graph.getEdgeSize() * 2) * 5);
	//d_status.temp_count = d_status.int_memory + graph.getNodeSize() * 12 + ((graph.getEdgeSize() * 2) * 6);
	d_status.temp_source = d_status.neighbourCommunities;// d_status.int_memory + graph.getNodeSize() * 13 + ((graph.getEdgeSize() * 2) * 7);
	//cuGraph.edgeSource = d_status.int_memory + graph.getNodeSize() * 12 + ((graph.getEdgeSize() * 2) * 5);
	//cuGraph.edgeTarget = d_status.int_memory + graph.getNodeSize() * 12 + ((graph.getEdgeSize() * 2) * 5 + graph.getEdgeSize());
	cuGraph.edgeSource_temp = d_status.neighbourSource_local;// d_status.int_memory + graph.getNodeSize() * 13 + ((graph.getEdgeSize() * 2) * 7 + graph.getEdgeSize() * 2);
	cuGraph.edgeTarget_temp = d_status.neighbourSource_local + graph.getEdgeSize();// d_status.int_memory + graph.getNodeSize() * 13 + ((graph.getEdgeSize() * 2) * 7 + graph.getEdgeSize() * 3);

	d_status.gdegrees = d_status.float_memory;
	d_status.loops = d_status.float_memory + graph.getNodeSize();
	d_status.result = d_status.float_memory + graph.getNodeSize() * 2;
	d_status.neighbourWeights = d_status.float_memory + graph.getNodeSize() * 3;
	d_status.neighbourWeights_local = d_status.float_memory + graph.getNodeSize() * 3 + (graph.getEdgeSize() * 2);
	//d_status.neighbourWeights_store = d_status.float_memory + graph.getNodeSize() * 3 + (graph.getEdgeSize() * 2) * 2;
	//d_status.communityIncrease = d_status.float_memory + graph.getNodeSize() * 3 + (graph.getEdgeSize() * 2) * 3;
	//cuGraph.edgeWeight = d_status.float_memory + graph.getNodeSize() * 3 + (graph.getEdgeSize() * 2) * 3;
	cuGraph.edgeWeight_temp = d_status.neighbourWeights_local;// d_status.float_memory + graph.getNodeSize() * 3 + (graph.getEdgeSize() * 2) * 4 + graph.getEdgeSize();
}

void DeAllocateMemory(StatusCUDA &d_status)
{
	cudaFree(d_status.uint_memory);
	cudaFree(d_status.int_memory);
	cudaFree(d_status.float_memory);
}

int first = 0;

std::vector<Partition> generate_dendogram(Graph &original_graph, map<int,pair<int,float>> &part_init)
{
	if (original_graph.getEdgeSize() == 0)
	{
		Community part;
		part.community = new int[original_graph.getNodeSize()];
		part.node = new int[original_graph.getNodeSize()];

		for (int i = 0; i < original_graph.getEdgeSize(); ++i)
		{
			part.community[i] = part.node[i] = original_graph.getNodeAt(i);
		}

		Partition p;
		p.partition = part;
		p.count = original_graph.getNodeSize();

		std::vector<Partition> ret;
		ret.push_back(p);

		return ret;
	}

	std::vector<Partition> status_list;
	StatusCUDA d_status;
	CUDAGraph cuGraph;

	//CUDPPHandle theCudpp;

	//CUDPPResult result = cudppCreate(&theCudpp);

	//d_status.hash_table_config.type = CUDPP_BASIC_HASH_TABLE;
	//d_status.hash_table_config.kInputSize = 10;// original_graph.getNodeSize();
	//d_status.hash_table_config.space_usage = 1.05;

	//clock_t begin = clock();

	//result = cudppHashTable(theCudpp, &d_status.hash_table_handle, &d_status.hash_table_config);
	//if (result != CUDPP_SUCCESS)
	//{
	//	fprintf(stderr, "Error in cudppHashTable call in"
	//		"testHashTable (make sure your device is at"
	//		"least compute version 2.0\n");
	//}

	AllocateMemory(d_status, original_graph, cuGraph);

	cuGraph.nodesSize = original_graph.getNodeSize();
	cuGraph.edgeSize = original_graph.getEdgeSize();

	cout << "nodes: " << cuGraph.nodesSize << endl;
	cout << "edges: " << cuGraph.edgeSize << endl;

	double time = 0;

	//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	int *neighbourSource = new int[original_graph.getEdgeSize()];
	int *neighbourTarget = new int[original_graph.getEdgeSize()];
	float *neighbourWeights = new float[original_graph.getEdgeSize()];

	double memcpy = 0;

	for (int round = 0; round < 1; ++round)
	{
		first = 1;

		clock_t test1 = clock();

		clock_t t1 = clock();

		cuGraph.nodesSize = original_graph.getNodeSize();
		cuGraph.edgeSize = original_graph.getEdgeSize();

		cudaMemcpy(cuGraph.node, original_graph.getNodes(), sizeof(unsigned int) * cuGraph.nodesSize, cudaMemcpyHostToDevice);
		cudaMemcpy(d_status.neighbourSource, original_graph.getEdgeContainer().source, sizeof(int) * cuGraph.edgeSize, cudaMemcpyHostToDevice);
		cudaMemcpy(d_status.neighbourTarget, original_graph.getEdgeContainer().target, sizeof(int) * cuGraph.edgeSize, cudaMemcpyHostToDevice);
		
		cudaMemcpy(d_status.neighbourWeights, original_graph.getEdgeContainer().weight, sizeof(float) * cuGraph.edgeSize, cudaMemcpyHostToDevice);

		//cudaMemcpy(cuGraph.edgeSource, original_graph.getEdgeContainer().source, sizeof(int) * cuGraph.edgeSize, cudaMemcpyHostToDevice);
		//cudaMemcpy(cuGraph.edgeTarget, original_graph.getEdgeContainer().target, sizeof(int) * cuGraph.edgeSize, cudaMemcpyHostToDevice);

		//cudaMemcpy(cuGraph.edgeWeight, original_graph.getEdgeContainer().weight, sizeof(float) * cuGraph.edgeSize, cudaMemcpyHostToDevice);

		clock_t t2 = clock();

		memcpy += (double)(t2 - t1) / CLOCKS_PER_SEC;

		//cout << "cuGraph init memcpy time: " << (double)(t2 - t1) / CLOCKS_PER_SEC << endl;

		CUDPPHandle theCudpp;

		t1 = clock();

		CUDPPResult result = cudppCreate(&theCudpp);

		if (result != CUDPP_SUCCESS)
		{
			fprintf(stderr, "Error initializing CUDPP Library.\n");
			unsigned int retval = 1;
			//return retval;
		}

		d_status.hash_table_config.type = CUDPP_BASIC_HASH_TABLE;
		d_status.hash_table_config.kInputSize = original_graph.getNodeSize();
		d_status.hash_table_config.space_usage = 1.05;

		clock_t begin = clock();

		result = cudppHashTable(theCudpp, &d_status.hash_table_handle, &d_status.hash_table_config);
		if (result != CUDPP_SUCCESS)
		{
			fprintf(stderr, "Error in cudppHashTable call in"
				"testHashTable (make sure your device is at"
				"least compute version 2.0\n");
		}
		t2 = clock();

		//cout << "cudpp creation: " << (double)(t2 - t1) / CLOCKS_PER_SEC << endl;

		double neighbour = 0, init = 0, onelevel = 0, renumber = 0, induce = 0;

		t1 = clock();

		CalculateNeighbour(d_status, cuGraph);

		t2 = clock();

		neighbour += (double)(t2 - t1) / CLOCKS_PER_SEC;

		//cout << "cuGraph CalculateNeighbour time: " << (double)(t2 - t1) / CLOCKS_PER_SEC << endl;

		clock_t start = clock();

		initCUDA(d_status, cuGraph);

		clock_t end = clock();

		init += (double)(end - start) / CLOCKS_PER_SEC;

		//cout << "init time cuda: " << (double)(end - start) / CLOCKS_PER_SEC << endl;

		map<int, pair<int, float>> empty_map;

		start = clock();

		float mod = modularityCUDA(d_status, original_graph.getNodeSize());// modularity(status, localWeight);

		end = clock();

		//cout << "modularity time: " << (double)(end - start) / CLOCKS_PER_SEC << endl;

		//cout << "modularity: " << mod << endl;

		start = clock();

		float new_mod =  oneLevelCUDA(d_status, cuGraph.nodesSize, cuGraph.nrOfAllNeighbours, cuGraph.edgeSize * 2, mod);

		end = clock();

		onelevel += (double)(end - start) / CLOCKS_PER_SEC;

		//cout << "modularity with onelevel time: " << (double)(end - start) / CLOCKS_PER_SEC << endl;

		begin = clock();

		Partition p2 = renumberCUDA(theCudpp, d_status);

		end = clock();

		renumber += (double)(end - start) / CLOCKS_PER_SEC;

		//cout << "renumberCUDA time: " << (double)(end - begin) / CLOCKS_PER_SEC << endl;

		status_list.push_back(p2);

		mod = new_mod;

		start = clock();

		cudaMemcpy(d_status.neighbourSource, original_graph.getEdgeContainer().source, sizeof(int) * cuGraph.edgeSize, cudaMemcpyHostToDevice);
		cudaMemcpy(d_status.neighbourTarget, original_graph.getEdgeContainer().target, sizeof(int) * cuGraph.edgeSize, cudaMemcpyHostToDevice);

		cudaMemcpy(d_status.neighbourWeights, original_graph.getEdgeContainer().weight, sizeof(float) * cuGraph.edgeSize, cudaMemcpyHostToDevice);

		end = clock();

		memcpy += (double)(end - start) / CLOCKS_PER_SEC;

		start = clock();

		inducedGraphCUDA(theCudpp, d_status, cuGraph, p2);

		end = clock();

		induce += (double)(end - start) / CLOCKS_PER_SEC;

		start = clock();

		cudaMemcpy(neighbourSource, d_status.neighbourSource, sizeof(int) * cuGraph.edgeSize, cudaMemcpyDeviceToHost);
		cudaMemcpy(neighbourTarget, d_status.neighbourTarget, sizeof(int) * cuGraph.edgeSize, cudaMemcpyDeviceToHost);

		cudaMemcpy(neighbourWeights, d_status.neighbourWeights, sizeof(float) * cuGraph.edgeSize, cudaMemcpyDeviceToHost);

		end = clock();

		memcpy += (double)(end - start) / CLOCKS_PER_SEC;

		//cout << "inducedCUDA time: " << (double)(end - start) / CLOCKS_PER_SEC << endl;

		while (true)
		{
			//break;

			t1 = clock();

			CalculateNeighbour(d_status, cuGraph);

			t2 = clock();

			neighbour += (double)(t1 - t1) / CLOCKS_PER_SEC;

			//cout << "cuGraph CalculateNeighbour time: " << (double)(t2 - t1) / CLOCKS_PER_SEC << endl;

			start = clock();

			initCUDA(d_status, cuGraph);

			end = clock();

			init += (double)(end - start) / CLOCKS_PER_SEC;

			//cout << "init time cuda: " << (double)(end - start) / CLOCKS_PER_SEC << endl;

			start = clock();

			new_mod = oneLevelCUDA(d_status, cuGraph.nodesSize, cuGraph.nrOfAllNeighbours, cuGraph.edgeSize * 2, mod);

			end = clock();

			//cout << "onelevel time: " << (double)(end - start) / CLOCKS_PER_SEC << endl;

			////cout << "after onelevel" << endl;

			begin = clock();

			Partition p2 = renumberCUDA(theCudpp, d_status);

			end = clock();

			renumber += (double)(end - start) / CLOCKS_PER_SEC;

			//cout << "renumberCUDA time: " << (double)(end - begin) / CLOCKS_PER_SEC << endl;

			status_list.push_back(p2);

			if ((new_mod - mod) < __MIN)
			{
				cout << "modularity: " << mod << endl;

				break;
			}

			mod = new_mod;

			start = clock();

			cudaMemcpy(d_status.neighbourSource, neighbourSource, sizeof(int) * cuGraph.edgeSize, cudaMemcpyHostToDevice);
			cudaMemcpy(d_status.neighbourTarget, neighbourTarget, sizeof(int) * cuGraph.edgeSize, cudaMemcpyHostToDevice);

			cudaMemcpy(d_status.neighbourWeights, neighbourWeights, sizeof(float) * cuGraph.edgeSize, cudaMemcpyHostToDevice);

			end = clock();

			memcpy += (double)(end - start) / CLOCKS_PER_SEC;

			start = clock();

			inducedGraphCUDA(theCudpp, d_status, cuGraph, p2);

			end = clock();

			induce += (double)(end - start) / CLOCKS_PER_SEC;

			start = clock();

			cudaMemcpy(neighbourSource, d_status.neighbourSource, sizeof(int) * cuGraph.edgeSize, cudaMemcpyDeviceToHost);
			cudaMemcpy(neighbourTarget, d_status.neighbourTarget, sizeof(int) * cuGraph.edgeSize, cudaMemcpyDeviceToHost);

			cudaMemcpy(neighbourWeights, d_status.neighbourWeights, sizeof(float) * cuGraph.edgeSize, cudaMemcpyDeviceToHost);

			end = clock();

			memcpy += (double)(end - start) / CLOCKS_PER_SEC;

			//cout << "inducedCUDA time: " << (double)(end - start) / CLOCKS_PER_SEC << endl;

		}

		cout << "Memcpy: " << memcpy << endl;
		cout << "neighbour: " << neighbour << endl;
		cout << "init: " << init << endl;
		cout << "onelevel: " << onelevel << endl;
		cout << "renumber: " << renumber << endl;
		cout << "induce: " << induce << endl;

		t1 = clock();

		cudppDestroyHashTable(theCudpp, d_status.hash_table_handle);
		//cudppDestroyHashTable(theCudpp, d_status.commMap_hash_table_handle);
		//cudppDestroyHashTable(theCudpp, d_status.nodeCommMap_hash_table_handle);

		result = cudppDestroy(theCudpp);
		if (result != CUDPP_SUCCESS)
		{
			fprintf(stderr, "Error shutting down CUDPP Library.\n");
		}
		//cout << "############################################################" << endl;

		t2 = clock();

		//cout << "cudpp destroy" << (double)(t2 - t1) / CLOCKS_PER_SEC << endl;

		clock_t test2 = clock();

		//for (int i = 0; i < status_list.size(); ++i)
		//	status_list[i].Deallocate();

		//status_list.clear();

		if (round > 0)
			time += (double)(test2 - test1) / CLOCKS_PER_SEC;

		cout << (double)(test2 - test1) / CLOCKS_PER_SEC << endl;
	}

	DeAllocateMemory(d_status);

	cout << "time in avg 100 runs: " << time / double(100) << endl;

	cudaProfilerStop();

	return status_list;
}

int* calculateClusterSize(Graph &graph, Partition partitions, int numberOfClusters)
{
    int *clusterSize = new int[numberOfClusters];
    
    for (int i = 0; i < numberOfClusters; ++i)
        clusterSize[i] = 0;
    
    for (int i = 0; i < graph.getNodeSize(); ++i)
    {
        int cluster;
        for (int j = 0; j < partitions.count; ++j)
        {
            if (partitions.partition.node[j] == graph.getNodeAt(i))
            {
                cluster = partitions.partition.community[j];
                break;
            }
        }
        
        graph.setCluster(i, cluster);
        graph.setIndexInCluster(i,clusterSize[cluster]);
        clusterSize[cluster] += 1;
    }
    
    for (int i = 1; i < numberOfClusters; ++i)
        clusterSize[i] += clusterSize[i-1];
    
    return clusterSize;
}

void circleLayout(float *x, float *y, int i, int N)
{
    *x = (50 * cos(2 * M_PI * i / N));
    *y = (50 + 50 * sin(2 * M_PI * i / N));
}

struct LayoutData
{
    int begin, end;
    int *clusterSize;
    int nrPartitions;
    int nodeCount;
    Graph *graph;
};

void layoutProcessThread(LayoutData data)
{
    int metr = 0;
    for (int j = 0; j < data.nrPartitions; ++j)
    {
        if (j > 0)
            metr = data.clusterSize[j-1];
        for (int i = data.begin; i < data.end; ++i)
        {   
            if (data.graph->getCluster(i) == j)
            {
                circleLayout(data.graph->getXPtr(i), data.graph->getYPtr(i), metr+data.graph->getIndexInCluster(i), data.nodeCount);
            }
        }
    }
}

#define numberOfProc 8

void layoutProcess(int *clusterSize, Graph &graph, int nrPartitions)
{    
    LayoutData data[numberOfProc];
    thread *threads[numberOfProc];
    
    int nodeCount = graph.getNodeSize();
    
    for (int i = 0; i < numberOfProc; ++i)
    {
        int parts = nodeCount / numberOfProc;
        if (nodeCount % numberOfProc != 0)
            ++parts;

        data[i].begin = i * parts;
        if ((data[i].begin + parts) > nodeCount)
            data[i].end = nodeCount;
        else
            data[i].end = data[i].begin + parts;
        
        data[i].clusterSize = clusterSize;
        data[i].nodeCount = nodeCount;
        data[i].graph = &graph;
        data[i].nrPartitions = nrPartitions;

        threads[i] = new thread(layoutProcessThread, data[i]);
    }
    
    for (int i = 0; i < numberOfProc; ++i)
    {
        threads[i]->join();
        delete threads[i];
    }
}
    
int nodesCount;
int edgesCount;

#include <set>

set<int> nodeSet;

int *loadNodes()
{
	ifstream nodes("nodes.txt");
	int *loadedNodes;

	////////cout << nodes.fail() << endl;
	if (!nodes.fail())
	{
		int count = 0;

		string input;
		nodes >> input;
		//////cout << input << endl;
		loadedNodes = new int[atoi(input.c_str())];

		nodes >> input;
		int i = 0;

		while (!nodes.eof())
		{
			////////cout << i << endl;
			loadedNodes[i] = atoi(input.c_str());

			pair<set<int>::iterator, bool> result = nodeSet.insert(loadedNodes[i]);

			if (result.second == true)
				++count;

			++i;

			nodes >> input;
			////////cout << "in" << endl;
		}

		nodesCount = i;
		//cout << "newly added nodes after node load: " << count << endl;
	}

	//	//////cout << "out" << endl;

	return loadedNodes;
}

Edge *loadEdges()
{
	ifstream edges("edges.txt");
	Edge *loadedEdges;
	if (!edges.fail())
	{
		string input;
		edges >> input;
		loadedEdges = new Edge[atoi(input.c_str())];

		edges >> input;
		int i = 0;

		while (!edges.eof())
		{
			loadedEdges[i].source = atoi(input.c_str());
			edges >> input;
			loadedEdges[i].target = atoi(input.c_str());
			edges >> input;
			loadedEdges[i].weight = atof(input.c_str());
			edges >> input;
			
			nodeSet.insert(loadedEdges[i].source);
			nodeSet.insert(loadedEdges[i].target);
			
			++i;
		}

		edgesCount = i;
	}

	//cout << "number of nodes after edge load: " << nodeSet.size() << endl;

	return loadedEdges;
}

int main(int argc, char* argv[])
{
	Graph graph;

	clock_t load1 = clock();

	Edge *edges = loadEdges();

	clock_t load2 = clock();

	cout << "loadedges time: " << (double)(load2 - load1) / CLOCKS_PER_SEC << endl;

	load1 = clock();

	int *nodes = loadNodes();

	load2 = clock();

	cout << "loadnodes time: " << (double)(load2 - load1) / CLOCKS_PER_SEC << endl;

	//cout << "after load" << endl;

	//cout << nodesCount << endl;
	//cout << edgesCount << endl;

	graph.setNodes(nodes, nodesCount);

	cout << "after node set" << endl;

	graph.setEdges(edges, edgesCount);

	cout << "after edge set" << endl;

	clock_t gpu1 = clock();

	cudaSetDevice(0);

	clock_t gpu2 = clock();

	//cout << "GPU set time: " << (double)(gpu2 - gpu1) / CLOCKS_PER_SEC << endl;

	gpu1 = clock();

	int *d_warmup;
	cudaMalloc(&d_warmup, sizeof(int)*1024*1024*200);
	cudaFree(d_warmup);

	gpu2 = clock();

	//cout << "GPU meminit time: " << (double)(gpu2 - gpu1) / CLOCKS_PER_SEC << endl;

	clock_t start = clock();

	map<int, pair<int, float>> init_state;

	std::vector<Partition> dendogram = generate_dendogram(graph, init_state);

	clock_t end = clock();

	cout << "generate time: " << (double)(end - start) / CLOCKS_PER_SEC << endl;

	cout << countPartitions(getPartitions(dendogram)) << endl;

	//for (int i = 0; i < dendogram.size(); ++i)
	//	dendogram[i].Deallocate();

	return 0;
}

boost::python::dict start(boost::python::list nodeList, boost::python::list edgeList, boost::python::list part_init)
{
	bool tryToRecoverFromException = false;
	boost::python::dict partitionDictionary;

	do
	{
		try
		{
			try
			{
				tryToRecoverFromException = false;

				Graph original_graph(nodeList, edgeList);
				//original_graph.makeAssocMat();

				map<int, pair<int, float>> init_state;
				map<int, int> reComm;
				int newComm = 0;

				for (int i = 0; i < boost::python::len(part_init); ++i)
				{
					boost::python::dict partDict = boost::python::extract<boost::python::dict>(part_init[i]);
					pair<int, float> partition;

					int comm = boost::python::extract<int>(partDict["community"]);

					if (reComm.find(comm) != reComm.end())
						comm = reComm[comm];
					else
					{
						reComm[comm] = newComm;
						comm = newComm;
						++newComm;
					}

					float weight = boost::python::extract<float>(partDict["weight"]);

					partition.first = comm;
					partition.second = weight;
					init_state[boost::python::extract<int>(partDict["node"])] = partition;
				}

				clock_t start = clock();

				std::vector<Partition> dendogram = generate_dendogram(original_graph, init_state);

				clock_t end = clock();

				//cout << "louvain in " << (double)(end-start) / CLOCKS_PER_SEC << endl;

				start = clock();

				Partition partitions = getPartitions(dendogram);

				end = clock();

				cout << "get partitions " << (double)(end-start) / CLOCKS_PER_SEC << endl;

				start = clock();

				int nrPartitions = countPartitions(partitions);

				end = clock();

				cout << "count partitions " << (double)(end-start) / CLOCKS_PER_SEC << " " << nrPartitions << endl;

				start = clock();

				int *clusterSize = calculateClusterSize(original_graph, partitions, nrPartitions);

				end = clock();

				cout << "cluster size time " << (double)(end-start) / CLOCKS_PER_SEC << endl;

				start = clock();

				layoutProcess(clusterSize, original_graph, nrPartitions);

				delete[] clusterSize;

				end = clock();

				cout << "layout process time " << (double)(end - start) / CLOCKS_PER_SEC << endl;

				start = clock();

				partitionDictionary["size"] = nrPartitions;
				for (int i = 0; i < partitions.count; ++i)
				{
					partitionDictionary[partitions.partition.node[i]] = partitions.partition.community[i];
				}

				end = clock();

				cout << "dictionary done" << endl;

				for (int i = 0; i < dendogram.size(); ++i)
					dendogram[i].Deallocate();

				cout << "deallocate done" << endl;

				for (int i = 0; i < original_graph.getNodeSize(); ++i)
				{
					boost::python::dict nodeDict = boost::python::extract<boost::python::dict>(nodeList[i]);
					nodeDict["cluster"] = original_graph.getCluster(i);
					nodeDict["my_community"] = original_graph.getCluster(i);
					nodeDict["x"] = original_graph.getX(i);
					nodeDict["y"] = original_graph.getY(i);
				}

				cout << "ready to return" << endl;
			}
			catch (/*bad_alloc exception*/ exception e)
			{
				//cout << "bad alloc?" << endl;
				cout << e.what() << endl;
				tryToRecoverFromException = false;
			}
			
			cout << "ready to return 2" << endl;
		}
		catch (...)
		{
			tryToRecoverFromException = true;
		}

		cout << "ready to return 3" << endl;
	} while (tryToRecoverFromException);

	cout << "before return" << endl;

	return partitionDictionary;
}

#if defined(_WIN32)
BOOST_PYTHON_MODULE(Louvain_ext_Windows)
#elif defined(__linux__)
BOOST_PYTHON_MODULE(Louvain_ext_CentOS)
#endif
{
	using namespace boost::python;
	def("start", start);
}
