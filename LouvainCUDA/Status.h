/*
*  Copyright 2017 Richard Forster
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*/

#ifndef _STATUS_H_
#define _STATUS_H_

#include "community.h"
#include <atomic>

#include <cstddef>

struct Status
{
	unsigned int *node;

	int *nodesToCom;
	int *nodesToComPrev;
	int *nodesToComNext;
	float totalWeight = 0;
	std::atomic<int> *internals;
	std::atomic<int> *degrees;
	float *gdegrees;
	float *loops;
    
    int *bestCom;

	unsigned int *hash_idx;
    
	std::atomic<int> *comSize = NULL;
	std::atomic<int> *comSizeUpd = NULL;
	std::atomic<int> *degreeUpd = NULL;
	std::atomic<int> *internalUpd = NULL;

	int *neighbourCommunities;
	float *neighbourWeights;

	int keys;
};

#endif
