/*
*  Copyright 2017 Richard Forster
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*/

#ifndef _STATUSCUDA_H_
#define _STATUSCUDA_H_

#include <cstddef>

#include "Graph.h"
#include <cudpp_hash.h>

struct StatusCUDA
{
	unsigned int *uint_memory;
	int *int_memory;
	float *float_memory;

	unsigned int *node;

	int *nodesToCom;
	int *nodesToComPrev;
	int *nodesToComNext;
	float totalWeight = 0;
	int *internals;
	int *degrees;
	float *gdegrees;
	float *loops;
    
    int *bestCom;
    
	int *comSize = NULL;
	int *comSizeUpd = NULL;
	int *degreeUpd = NULL;
	int *internalUpd = NULL;

	int keys;

	unsigned int *hash_idx;
	unsigned int *hash_idx_source;
	unsigned int *hash_idx_target;

	int* nrOfNeighbours;
	int* startOfNeighbours;	
	AssocPair* neighbours;
	CommPair* neighCommunities;

	int *neighbourCommunities;
	int *neighbourCommunities_local;
	float *neighbourWeights;
	float *neighbourWeights_local;
	float *neighbourWeights_store;

	float *result;

	int *temp_source;
	//int *temp_count;

	//float* communityIncrease = NULL;

	int *neighbourSource;
	int *neighbourSourceIdx;

	int *neighbourSource_local = NULL;
	int *neighbourCounts = NULL;

	int *neighbourTarget;

	//int *reorderedNodes;

	bool init = true;
	
	CUDPPHashTableConfig hash_table_config;
	CUDPPHandle hash_table_handle;

	CUDPPHashTableConfig commMap_hash_table_config;
	CUDPPHandle commMap_hash_table_handle;

	CUDPPHashTableConfig nodeCommMap_hash_table_config;
	CUDPPHandle nodeCommMap_hash_table_handle;
};

#endif
