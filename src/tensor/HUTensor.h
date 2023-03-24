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

#include <cuda_fp16.h>

namespace TenTrans {

namespace gpu {

template <typename T>
void copy(HUPtr<HUDevice> device, const T* begin, const T* end, T* dest);

template <typename T>
void fill(HUPtr<HUDevice> device, T* begin, T* end, T value);

}

// enum TENSOR_DATA_TYPE {TT_INT32, TT_INT8, TT_FLOAT32, TT_FLOAT16, TT_DOUBLE};

class HUTensor {
private:
	// memory space used in this tensor
	HUPtr<HUMemoryPiece> memory_;
	// shape of this this tensor 
	HUShape shape_;
	HUPtr<HUDevice> device_;
	HUPtr<HUMemPool> mem_ = NULL;
    TENSOR_DATA_TYPE dataType_ = TENSOR_DATA_TYPE::TT_FLOAT32;
    int unitSize_ = sizeof(float);

public:
	// constructor
	HUTensor(HUPtr<HUMemoryPiece> memory, HUShape shape, HUPtr<HUDevice> device, \
            TENSOR_DATA_TYPE dataType=TENSOR_DATA_TYPE::TT_FLOAT32);
	HUTensor(const int myOrder, const int * myDimSize, HUPtr<HUMemoryPiece> memory, HUPtr<HUDevice> device, \
            TENSOR_DATA_TYPE dataType=TENSOR_DATA_TYPE::TT_FLOAT32);
	HUTensor(const int myOrder, const int * myDimSize, HUPtr<HUMemoryPiece> memory, HUPtr<HUDevice> device, HUPtr<HUMemPool> mem, \
            TENSOR_DATA_TYPE dataType=TENSOR_DATA_TYPE::TT_FLOAT32);
	
	// deconstructor
	~HUTensor() { }

	// reset memory space used in this tensor 
	virtual void reset(HUPtr<HUMemoryPiece> memory);

	HUPtr<HUMemPool> GetMemPool(){ return this->mem_; }

	// get memory piece 
	virtual HUPtr<HUMemoryPiece> memory();

	// get shape of this tensor 
	virtual HUShape& shape();

	virtual int order() { return this->shape_.size(); }
	int& dim(int i){ return this->shape_.dim(i); }

    TENSOR_DATA_TYPE getDataType() { return this->dataType_; }

    void setUnitSize();

    int getUnitSize() { return this->unitSize_; }

	// get data of this tensor
    /*
    template <typename T>
    virtual T* data();
    */
    virtual TT_DATA_TYPE* data();

	// get element number of this tensor
	virtual size_t size();

	// return scalar value if this tensor is scalar
    /*
    template <typename T>
    virtual T scalar();
    */
    virtual TT_DATA_TYPE scalar();

	// return backend of this tensor
	HUPtr<HUDevice> getDevice();

	// return device information used in  this tensor 
	DeviceId getDeviceId();

	// return subtensor of this tensor. Regardless of the shape of this tensor, the return subtensor is {1, size}
	HUTensor subtensor(int offset, int size);

	// get specified value indexed by i
    /*
    template <typename T>
    T get(size_t i);
    */
    TT_DATA_TYPE get(size_t i);

	void toCuda();

    /*
    template <typename T>
    void set(size_t i, T value);
    */
    void set(size_t i, TT_DATA_TYPE value);

	// get data of this tensor and assign to v
    /*
    template <typename T>
    void get(std::vector<T>& v);
    */
    void get(std::vector<TT_DATA_TYPE>& v);

	// set data of this tensor using params begin and end
    /*
    template <typename T>
    void set(const T* begin, const T* end);
    */
    void set(const TT_DATA_TYPE* begin, const TT_DATA_TYPE* end);

    // void defaultSet(const void* data, size_t size);

	// set data of this tensor using vector 
    /*
    template <typename T>
    void set(const std::vector<T>& v);
    */
    void set(const std::vector<float>& v);

	// assign to all data of this tensor with value
    /*
    template <typename T>
    void set(T value);
    */
    void set(TT_DATA_TYPE value);

	// copy from other tensor
	void copyFrom(HUTensor in);

	std::string debug();
	
};
}
