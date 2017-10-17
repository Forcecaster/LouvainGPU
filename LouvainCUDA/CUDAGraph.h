/*
*  Copyright 2017 Richard Forster
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*/

#ifndef __CUDA__GRAPH__
#define __CUDA__GRAPH__

struct CUDAGraph
{
	unsigned int *node;
	
	int *edgeSource;
	int *edgeTarget;
	float *edgeWeight;

	int *edgeSource_temp = NULL;
	int *edgeTarget_temp = NULL;
	float *edgeWeight_temp = NULL;

	int nodesSize, edgeSize;
	int nrOfAllNeighbours;
};

#endif // !__CUDA__GRAPH__

