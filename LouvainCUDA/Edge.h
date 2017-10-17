/*
*  Copyright 2017 Richard Forster
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*/

#ifndef _EDGE_H_
#define _EDGE_H_

#include <map>
#include <set>

#include "utils.h"

#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/class.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/list.hpp>

class EdgeContainer
{
public:
	EdgeContainer() {}
	~EdgeContainer()
	{
		if (count != 0)
		{
			aligned_memory_free(target);
			aligned_memory_free(source);
			aligned_memory_free(weight);
		}
	}

	void Populate(boost::python::list edgeList, int count, std::map<int,int> &nodeMap)
	{
		this->count = count;

		target = (int*)aligned_memory_alloc(count * sizeof(int), 32);
		source = (int*)aligned_memory_alloc(count * sizeof(int), 32);
		weight = (float*)aligned_memory_alloc(count * sizeof(float), 32);

		for (int i = 0; i < count; ++i)
		{
			boost::python::dict edgeDict = boost::python::extract<boost::python::dict>(edgeList[i]);
			this->target[i] = boost::python::extract<int>(edgeDict["target"]);
			this->source[i] = boost::python::extract<int>(edgeDict["source"]);

			this->weight[i] = boost::python::extract<float>(edgeDict["weight"]);
		}
	}

	int* target;
	int* source;
	float* weight;

	int count = 0;
};

struct Edge
{
	int target, source;
	float weight;
};

#endif