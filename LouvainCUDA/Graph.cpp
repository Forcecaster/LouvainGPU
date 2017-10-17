/*
*  Copyright 2017 Richard Forster
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*/

#include <set>

#include "Graph.h"
#include "utils.h"
#include <iostream>

#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/class.hpp>
#include <boost/python/extract.hpp>

#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>

Graph::Graph()
{
	this->nodes = NULL;
	this->x = NULL;
	this->y = NULL;
	this->cluster = NULL;
	this->indexInCluster = NULL;

	this->neighbourSource = NULL;
	this->neighbourTarget = NULL;
	this->neighbourWeight = NULL;

	this->neighbours = NULL;
	this->nrOfNeighbours = NULL;
	this->startOfNeighbours = NULL;

	this->neighCommunities = NULL;

	neighbourSource_temp = NULL;
	neighbourTarget_temp = NULL;
	neighbourCounts = NULL;
	neighbourCounts_temp = NULL;

	this->edgeSize = 0;
	this->nodeSize = 0;

	lastNodeForIndex = -1;
}

Graph::Graph(boost::python::list nodeList, boost::python::list edgeList)
{
	this->nodeSize = boost::python::len(nodeList);
	this->edgeSize = boost::python::len(edgeList);

	this->nodes = new int[nodeSize];

	this->x = new float[nodeSize];
	this->y = new float[nodeSize];
	this->cluster = new int[nodeSize];
	this->indexInCluster = new int[nodeSize];

	for (int i = 0; i < nodeSize; ++i)
	{
		boost::python::dict nodeDict = boost::python::extract<boost::python::dict>(nodeList[i]);
		this->nodes[i] = boost::python::extract<int>(nodeDict["id"]);

		nodeMap[this->nodes[i]] = i;
	}

	this->edges.Populate(edgeList, edgeSize, nodeMap);

	this->neighbourSource = NULL;
	this->neighbourTarget = NULL;
	this->neighbourWeight = NULL;

	this->neighbours = NULL;
	this->nrOfNeighbours = NULL;
	this->startOfNeighbours = NULL;

	this->neighCommunities = NULL;

	neighbourSource_temp = NULL;
	neighbourTarget_temp = NULL;
	neighbourCounts = NULL;
	neighbourCounts_temp = NULL;

	lastNodeForIndex = -1;
}

int* Graph::getNrOfNeighbours()
{
	return nrOfNeighbours;
}

int* Graph::getStartOfNeighbours()
{
	return startOfNeighbours;
}

AssocPair* Graph::getNeighbours()
{
	return neighbours;
}

CommPair* Graph::getNeighCommunities()
{
	return neighCommunities;
}

int Graph::getNrOfAllNeighbours()
{
	return nrOfAllNeighbours;
}

int Graph::setNodes(int *nodes, int size)
{
	if (this->nodes != NULL)
		delete[] this->nodes;

	this->nodes = new int[size];
	this->nodeSize = size;

	for (int i = 0; i < size; ++i)
	{
		this->nodes[i] = nodes[i];
		nodeMap[this->nodes[i]] = i;
	}

	return 0;
}

int Graph::setEdges(Edge *edges, int size)
{
	this->edges.source = (int*)aligned_memory_alloc(size * sizeof(int), 16);
	this->edges.target = (int*)aligned_memory_alloc(size * sizeof(int), 16);
	this->edges.weight = (float*)aligned_memory_alloc(size * sizeof(float), 16);
	this->edges.count = size;

	this->edgeSize = size;

	for (int i = 0; i < size; ++i)
	{
		if (edges[i].source == edges[i].target)
			std::cout << "loop" << std::endl;

		this->edges.source[i] = edges[i].source;
		this->edges.target[i] = edges[i].target;

		this->edges.weight[i] = edges[i].weight;
	}

	return 0;
}

Graph::~Graph()
{
	if (this->nodes != NULL)
		delete[] this->nodes;

	if (this->x != NULL)
		delete[] this->x;

	if (this->y != NULL)
		delete[] this->y;

	if (this->cluster != NULL)
		delete[] this->cluster;

	if (this->indexInCluster != NULL)
		delete[] this->indexInCluster;

	if (this->neighbourSource == NULL)
		delete[] this->neighbourSource;
	if (this->neighbourTarget == NULL)
		delete[] this->neighbourTarget;
	if (this->neighbourWeight == NULL)
		delete[] this->neighbourWeight;

	if (this->nrOfNeighbours != NULL)
	{
		delete[] this->nrOfNeighbours;
	}

	if (this->startOfNeighbours != NULL)
	{
		delete[] this->startOfNeighbours;
	}

	if (this->neighbours != NULL)
	{
		delete[] this->neighbours;
	}

	if (this->neighCommunities != NULL)
	{
		delete[] this->neighCommunities;
	}

	if (neighbourSource_temp != NULL)
		delete[] neighbourSource_temp;
	if (neighbourTarget_temp != NULL)
		delete[] neighbourTarget_temp;
	if (neighbourCounts != NULL)
		delete[] neighbourCounts;
	if (neighbourCounts_temp != NULL)
		delete[] neighbourCounts_temp;

	//std::cout << "destructor" << std::endl;
}

void Graph::allocateNodes(int size)
{
	if (this->nodes == NULL)
	{
		this->nodes = new int[size];
	}
}
void Graph::allocateEdges(int size)
{
	if (edges.count == 0)
	{
		edges.count = size;

		edges.source = (int*)aligned_memory_alloc(size * sizeof(int), 16);
		edges.target = (int*)aligned_memory_alloc(size * sizeof(int), 16);
		edges.weight = (float*)aligned_memory_alloc(size * sizeof(float), 16);
	}
}

int* Graph::getNodes()
{
	return nodes;
}

EdgeContainer& Graph::getEdgeContainer()
{
	return edges;
}

void Graph::makeNodeMap()
{
	for (int i = 0; i < nodeSize; ++i)
	{
		nodeMap[nodes[i]] = i;
	}
}

void Graph::addNode(int node)
{
	this->nodes[this->nodeSize] = node;

	nodeMap[node] = this->nodeSize;

	++this->nodeSize;
}

void Graph::addEdgeFromVector(std::vector<Edge> edge)
{
	for (int i = 0; i < edge.size(); ++i)
	{
		edges.source[edgeSize] = edge[i].source;
		edges.target[edgeSize] = edge[i].target;
		edges.weight[edgeSize] = edge[i].weight;
		++edgeSize;
	}
}

Edge Graph::getEdgeAt(int idx)
{
	Edge edge;
	edge.source = edges.source[idx];
	edge.target = edges.target[idx];
	edge.weight = edges.weight[idx];

	return edge;
}

void Graph::addEdge(Edge edge)
{
	edges.source[edgeSize] = edge.source;
	edges.target[edgeSize] = edge.target;
	edges.weight[edgeSize] = edge.weight;

	++edgeSize;
}

void Graph::setCluster(int i, int cluster)
{
	this->cluster[i] = cluster;
}

void Graph::setIndexInCluster(int i, int idx)
{
	indexInCluster[i] = idx;
}

int Graph::getCluster(int i)
{
	return this->cluster[i];
}

int Graph::getIndexInCluster(int i)
{
	return indexInCluster[i];
}

float *Graph::getXPtr(int i)
{
	return x + i;
}

float *Graph::getYPtr(int i)
{
	return y + i;
}

float Graph::getX(int i)
{
	return x[i];
}

float Graph::getY(int i)
{
	return y[i];
}

int *Graph::getNeighbourTarget()
{
	return neighbourTarget;
}

int *Graph::getNeighbourSource()
{
	return neighbourSource;
}

float *Graph::getNeighbourWeight()
{
	return neighbourWeight;
}

struct CounterOp
{
	bool operator()
		(const int& lhs, const int& rhs)
	{
		if (rhs == -1)
			return 0;
		return 1;
	}
};

void Graph::CalculateNeighbour()
{
	if (neighbourSource == NULL)
	{
		neighbourSource = new int[edges.count * 2];
		neighbourTarget = new int[edges.count * 2];
		neighbourWeight = new float[edges.count * 2];
	}

	memcpy(neighbourSource, edges.source, edges.count * sizeof(int));
	memcpy(neighbourTarget, edges.target, edges.count * sizeof(int));
	memcpy(neighbourWeight, edges.weight, edges.count * sizeof(float));

	memcpy(neighbourSource + edges.count, edges.target, edges.count * sizeof(int));
	memcpy(neighbourTarget + edges.count, edges.source, edges.count * sizeof(int));
	memcpy(neighbourWeight + edges.count, edges.weight, edges.count * sizeof(float));

	int idx = 0;

	nrOfAllNeighbours = edges.count * 2;

	auto ziped_key = thrust::make_zip_iterator(thrust::make_tuple(neighbourSource, neighbourTarget));

	thrust::sort_by_key(thrust::host, ziped_key, ziped_key + edges.count * 2, neighbourWeight);

	if (nrOfNeighbours == NULL)
	{
		nrOfNeighbours = new int[nodeSize]();
		startOfNeighbours = new int[nodeSize]();
	}

	thrust::pair<int*, int*> new_end;

	if (neighbourSource_temp == NULL)
	{
		neighbourSource_temp = new int[nrOfAllNeighbours];
		neighbourTarget_temp = new int[nrOfAllNeighbours];

		neighbourCounts = new int[nrOfAllNeighbours];
		neighbourCounts_temp = new int[nrOfAllNeighbours];
	}

	memset(neighbourCounts, 0, sizeof(int) * nrOfAllNeighbours);

	thrust::transform(thrust::host, neighbourCounts, neighbourCounts + nrOfAllNeighbours, neighbourTarget_temp, neighbourCounts, CounterOp());

	new_end = thrust::reduce_by_key(thrust::host, neighbourSource, neighbourSource + nrOfAllNeighbours, neighbourCounts, neighbourSource_temp, neighbourCounts_temp);

	memset(neighbourCounts_temp + (new_end.first - neighbourSource_temp), 0, sizeof(int) * (nodeSize - (new_end.first - neighbourSource_temp)));

	memcpy(getNrOfNeighbours(), neighbourCounts_temp, sizeof(int) * nodeSize);

	thrust::exclusive_scan(thrust::host, neighbourCounts_temp, neighbourCounts_temp + nodeSize, getStartOfNeighbours(), 0);
}

AssocPair* Graph::getNeighboursForNode(int node)
{
	return neighbours + startOfNeighbours[node];
}

int Graph::getNrOfNeighboursForNode(int node)
{
	return nrOfNeighbours[node];
}

CommPair* Graph::getNeighbourCommunities(int node)
{
	return neighCommunities + startOfNeighbours[node];
}

int Graph::getNodeSize()
{
	return this->nodeSize;
}

void Graph::setNodeCount(int n)
{
	nodeSize = n;
}

int Graph::getEdgeSize()
{
	return this->edgeSize;
}

int Graph::getNodeAt(int i)
{
	return this->nodes[i];
}

float Graph::getGraphSize()
{
	float size = 0;
	for (int i = 0; i < this->edgeSize; ++i)
		size += edges.weight[i];

	return size;
}

float Graph::getDegreeForNodeAt(int i, int &degIdx)
{
	//***New degree computation***

	float weight = 0;

	for (int j = 0; j < nrOfNeighbours[i]; ++j)
	{
		int *temp = startOfNeighbours + i;
		int temp2 = j;

		if (startOfNeighbours[i] + j >= edges.count * 2)
			std::cout << i << " " << startOfNeighbours[i] << " " << j << std::endl;

		float value = neighbourWeight[startOfNeighbours[i] + j];
		if (value == 0)
			value = 1;

		if (neighbourSource[startOfNeighbours[i] + j] == neighbourTarget[startOfNeighbours[i] + j])
			value *= 2;

		weight += value;
	}

	degIdx = nodeMap[neighbourSource[startOfNeighbours[i]]];

	return weight;
}

float Graph::getEdgeWeight(int node1, int node2, int &loopIdx)
{
	//***Proposed new getEdgeWeight currently with erronous indexing***

	float weight = 0;

	for (int j = 0; j < nrOfNeighbours[node1]; ++j)
	{
		if (neighbourSource[startOfNeighbours[node1] + j] == neighbourTarget[startOfNeighbours[node2] + j])
			weight = neighbourWeight[startOfNeighbours[node1] + j];
	}

	return weight;

	//int idx = -1;
	//for (int i = 0; i < this->mat.size(); ++i)
	//{
	//	if (this->mat[i].first == node1)
	//	{
	//		idx = i;
	//		for (int j = 0; j < this->mat[i].second.size(); ++j)
	//			if (this->mat[i].second[j].neighbour == node2)
	//				return this->mat[i].second[j].weight;

	//		return 0;
	//	}
	//}

	//if (idx == -1)
	//	return 0;

	//return -1;
}

int Graph::getIdxOfNode(int node)
{
	if (nodeMap.find(node) != nodeMap.end())
		return nodeMap[node];

	return -1;
}

void Graph::addEdgeFromVectorizedResult(int *source, int *target, float *sum, int count)
{
	memcpy(edges.source, source, sizeof(int) * count);
	memcpy(edges.target, target, sizeof(int) * count);
	memcpy(edges.weight, sum, sizeof(float) * count);

	edgeSize = count;
}