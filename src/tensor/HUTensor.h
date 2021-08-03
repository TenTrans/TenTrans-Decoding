#pragma once
#include "HUMemory.h"
#include "HUShape.h"
#include "HUGlobal.h"
#include "Logging.h"
#include "HUDevice.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

namespace TenTrans{

namespace gpu{

template <typename T>
void copy(HUPtr<HUDevice> device, const T* begin, const T* end, T* dest);

template <typename T>
void fill(HUPtr<HUDevice> device, T* begin, T* end, T value);

void fill(HUPtr<HUDevice> device, float* begin, float* end, float value);

void alloc(float* data, size_t bufSize);
void free(float* data);
}

class HUTensor{
private:
	// memory space used in this tensor
	HUPtr<HUMemoryPiece> memory_;
	// shape of this this tensor 
	HUShape shape_;
	HUPtr<HUDevice> device_;
	HUPtr<HUMemPool> mem_ = NULL;

public:
	//constructor
	HUTensor(HUPtr<HUMemoryPiece> memory, HUShape shape, HUPtr<HUDevice> device);
	HUTensor(const int myOrder, const int * myDimSize, HUPtr<HUMemoryPiece> memory, HUPtr<HUDevice> device);
	HUTensor(const int myOrder, const int * myDimSize, HUPtr<HUMemoryPiece> memory, HUPtr<HUDevice> device, HUPtr<HUMemPool> mem);
	
	//deconstructor
	~HUTensor() {
/*
#ifdef CUDA_FOUND
		gpu::free(this->data()); 
#endif
*/
	};

	// reset memory space used in this tensor 
	virtual void reset(HUPtr<HUMemoryPiece> memory);

	HUPtr<HUMemPool> GetMemPool(){return this->mem_;}

	// get memory piece 
	virtual HUPtr<HUMemoryPiece> memory();

	// get shape of this tensor 
	virtual HUShape& shape();

	virtual int order() { return this->shape_.size(); }
	int& dim(int i){ return this->shape_.dim(i); }

	// get data of this tensor
	virtual float* data();

	// get element number of this tensor
	virtual size_t size();

	// return scalar value if this tensor is scalar
	virtual float scalar();

	// return backend of this tensor
	HUPtr<HUDevice> getDevice();

	// return device information used in  this tensor 
	DeviceId getDeviceId();

	// return subtensor of this tensor. Regardless of the shape of this tensor, the return subtensor is {1, size}
	HUTensor subtensor(int offset, int size);

	// get specified value indexed by i
	float get(size_t i);

	void toCuda();

	void set(size_t i, float value);

	// get data of this tensor and assign to v
	void get(std::vector<float>& v);

	// set data of this tensor using params begin and end
	void set(const float* begin, const float* end);

	// set data of this tensor using vector 
	void set(const std::vector<float>& v);
    // void set(const std::vector<int>& v);

	// assign to all data of this tensor with value
	void set(float value);

	// copy from other tensor
	void copyFrom(HUTensor in);

	std::string debug();
	
};
}
