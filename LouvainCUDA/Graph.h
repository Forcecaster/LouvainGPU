/*
*  Copyright 2017 Richard Forster
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*/

#ifndef _GRAPH_H_
#define _GRAPH_H_

#include "edge.h"
#include "Status.h"
#include <vector>
#include <map>
#include <set>
#include <concurrent_unordered_map.h>

#include <cstddef>

#include <boost/python/list.hpp>

struct AssocPair
{
	int neighbour;
	int neighbourIdx;
	float weight;
};

struct CommPair
{
	int node;
	int community;
	float weight;
};

class Graph
{
public:
	Graph();
	Graph(boost::python::list, boost::python::list);
	~Graph();

	Edge getEdgeAt(int);
	void addEdge(Edge);

	void allocateNodes(int);
	void allocateEdges(int);
	void setCluster(int, int);
	void setIndexInCluster(int, int);
	void addNode(int);
	void setNodeCount(int);
	// void addEdge(Edge);
	void addEdgeFromVector(std::vector<Edge>);

	void addEdgeFromVectorizedResult(int*, int*, float*, int);

	std::vector<AssocPair> getNeighboursOfNode(int);
	AssocPair* getNeighboursForNode(int);
	CommPair* getNeighbourCommunities(int);
	int getNrOfNeighboursForNode(int);

	void CalculateNeighbour();
	
	int getNodeAt(int);
	int* getNodes();
	EdgeContainer& getEdgeContainer();
	int getNodeSize();
	int getEdgeSize();
	float getGraphSize();
	float getDegreeForNodeAt(int, int&);
	float getEdgeWeight(int, int, int&);
	int getCluster(int);
	int getIndexInCluster(int);
	float getX(int);
	float getY(int);
	float *getXPtr(int);
	float *getYPtr(int);

	int getIdxOfNode(int);

	int* getNrOfNeighbours();
	int* getStartOfNeighbours();
	AssocPair* getNeighbours();
	CommPair* getNeighCommunities();
	int getNrOfAllNeighbours();

	int setNodes(int*, int);
	int setEdges(Edge*, int);

	int *getNeighbourSource();
	int *getNeighbourTarget();
	float *getNeighbourWeight();

	void makeNodeMap();

private:
	int *nodes;
	EdgeContainer edges;
	float *x, *y;
	int *cluster;
	int *indexInCluster;

	int *neighbourSource, *neighbourTarget;
	float *neighbourWeight;

	int nodeSize;
	int edgeSize;

	std::map<int, int> nodeMap;

	int lastNodeForIndex;
	int lastNodeIdx;

	int *nrOfNeighbours;
	int *startOfNeighbours;
	AssocPair *neighbours;
	CommPair *neighCommunities;

	int nrOfAllNeighbours;

	int *neighbourSource_temp;
	int *neighbourTarget_temp;
	int *neighbourCounts;
	int *neighbourCounts_temp;

	friend class EdgeContainer;
};

#endif
