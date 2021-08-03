
#pragma once
#include "HUGlobal.h"
#include "Logging.h"
#include "HUShape.h"
#include <iostream>

namespace TenTrans{

namespace Functional{

template <typename T, size_t N>
struct Array {
  typedef T value_type;

  T data_[N];

  __HDI__ const T* data() const { return data_; }

  __HDI__ T* data() { return data_; }

  __HDI__ constexpr static size_t size() { return N; }

  __HDI__ T& operator[](size_t i) { return data_[i]; }
  __HDI__ const T& operator[](size_t i) const { return data_[i]; }

  __HDI__ T* begin() { return data_; }
  __HDI__ const T* begin() const { return data_; }

  __HDI__ T* end() { return data_ + N; }
  __HDI__ const T* end() const { return data_ + N; }

  __HDI__ void fill(T val) {
    for(int i = 0; i < N; ++i)
      data_[i] = val;
  }
};

struct HUConstantShape{
	int	shape_[MAX_TENSOR_DIM_NUM];
	int stride_[MAX_TENSOR_DIM_NUM];
	int bstride_[MAX_TENSOR_DIM_NUM];
	size_t elements_ = {1};

	__HD__ HUConstantShape() {
		for(int i=0; i<MAX_TENSOR_DIM_NUM; i++)
			shape_[i] = 1;
		for(int i=0; i<MAX_TENSOR_DIM_NUM; i++)
			stride_[i] =1;
		for(int i=0; i<MAX_TENSOR_DIM_NUM; i++)
			bstride_[i] =0;
		this->elements_ = 1;
  	}

	__HD__ HUConstantShape(const HUConstantShape& shape){
		for(int i=0; i<MAX_TENSOR_DIM_NUM; i++)
			this->shape_[i] = shape.shape_[i];
		for(int i=0; i<MAX_TENSOR_DIM_NUM; i++)
            this->stride_[i] = shape.stride_[i];
		for(int i=0; i<MAX_TENSOR_DIM_NUM; i++)
            this->bstride_[i] = shape.bstride_[i];
		this->elements_ = shape.elements_;
	}

	HUConstantShape(const TenTrans::HUShape& shape) {
    	size_t filled = shape.size();

    	ABORT_IF(filled > MAX_TENSOR_DIM_NUM,
             	"[TenTrans] Recompile with MAX_TENSOR_DIM_NUM >= " + std::to_string(filled));
		
    	std::copy(shape.begin(), shape.end(), shape_ + MAX_TENSOR_DIM_NUM - filled);
		//std::cout << "test3" << std::endl;
		//for(int i=0; i<MAX_TENSOR_DIM_NUM; i++)
		//	std::cout << 
    	if(MAX_TENSOR_DIM_NUM - filled)
      		std::fill_n(shape_, MAX_TENSOR_DIM_NUM - filled, 1);
		//std::cout << "test4 "<< std::endl;
		//for(int i=0; i<MAX_TENSOR_DIM_NUM; i++)
		//	std::cout << "shape " << shape_[i] << std::endl;
    	updateStrides();
		//for(int i=0; i<MAX_TENSOR_DIM_NUM; i++)
		//	std::cout << "stride " << stride_[i] << std::endl;
		//for(int i=0; i<MAX_TENSOR_DIM_NUM; i++)
		//	std::cout << "bstride " << bstride_[i] << std::endl;
    	updateElements();
		//std::cout << elements_ << std::endl;
  	}

	__HDI__ void updateStrides() {
    	stride_[MAX_TENSOR_DIM_NUM - 1] = 1;
    	bstride_[MAX_TENSOR_DIM_NUM - 1] = shape_[MAX_TENSOR_DIM_NUM - 1] == 1 ? 0 : stride_[MAX_TENSOR_DIM_NUM - 1];

    	for(int i = MAX_TENSOR_DIM_NUM - 2; i >= 0; --i) {
      		stride_[i] = stride_[i + 1] * shape_[i + 1];
      		bstride_[i] = shape_[i] == 1 ? 0 : stride_[i];
    	}
		//std::cout << "test5" << std::endl;
  	}

  	__HDI__ void updateElements() {
    	elements_ = 1;
    	for(int i = 0; i < MAX_TENSOR_DIM_NUM; ++i)
      		elements_ *= shape_[i];
		//std::cout << "test6" << std::endl;
  	}

  	__HDI__ void set(int i, int dim) {
    	shape_[i] = dim;
    	updateStrides();
    	updateElements();
  	}

  	__HDI__ int dim(int i) { return shape_[i]; }

  	__HDI__ int dim(int i) const {
    	return const_cast<HUConstantShape&>(*this).dim(i);
  	}

  	__HDI__ int back() const { return dim(MAX_TENSOR_DIM_NUM - 1); }

	__HDI__ int operator[](int i) { return dim(i); }

  	__HDI__ int operator[](int i) const { return dim(i); }

  	__HDI__ int stride(int i) const { return stride_[i]; }

  	__HDI__ int bstride(int i) const { return bstride_[i]; }

  	__HDI__ static constexpr size_t size() { return MAX_TENSOR_DIM_NUM; }

  	__HDI__ int elements() const { return (int)elements_; }

  	__HDI__ int index(const int * da) const {
    	int i = 0;
    	for(int j = 0; j < MAX_TENSOR_DIM_NUM; ++j)
      		i += da[j] * stride_[j];
    	return i;
  	}

  	__HDI__ int bindex(const int* db) const {
    	int i = 0;
    	for(int j = 0; j < MAX_TENSOR_DIM_NUM; ++j)
      		i += db[j] * bstride_[j];
    	return i;
  	}

  	__HDI__ void dims(int i, int* dc) const {
    	for(int j = 0; j < MAX_TENSOR_DIM_NUM; ++j)
      		dc[j] = (i / stride_[j]) % shape_[j];
  	}

	__HDI__ int index(const Array<int, 4>& d) const {
    int i = 0;
    for(int j = 0; j < 4; ++j)
      i += d[j] * stride_[j];
    return i;
  }

  __HDI__ int bindex(const Array<int, 4>& d) const {
    int i = 0;
    for(int j = 0; j < 4; ++j)
      i += d[j] * bstride_[j];
    return i;
  }

  __HDI__ void dims(int i, Array<int, 4>& d) const {
    for(int j = 0; j < 4; ++j)
      d[j] = (i / stride_[j]) % shape_[j];
  }

  	__HDI__ bool operator==(const HUConstantShape& other) const {
    	for(int i = 0; i < MAX_TENSOR_DIM_NUM; ++i)
      		if(shape_[i] != other[i])
        		return false;
    	return true;
  	}

  	__HDI__ bool operator!=(const HUConstantShape& other) const {
    	return !(*this == other);
  	}
};

template <typename T>
struct HUTensor {
  T* data_;
  Functional::HUConstantShape shape_;

  __HD__ HUTensor() {}

  __HD__ HUTensor(T* ptr, const Functional::HUConstantShape& shape)
      : data_(ptr), shape_(shape) {}

  __H__ HUTensor(HUPtr<TenTrans::HUTensor> t) : data_(t->data()), shape_(t->shape()) { }

  __HDI__ float& operator[](size_t i) { return data_[i]; }
  __HDI__ const float& operator[](size_t i) const { return data_[i]; }

  __HDI__ float& GetItemByIndices(const int *indices) {
    return data_[shape_.index(indices)];
  }

  __HDI__ const float& GetItemByIndices(const int *indices) const {
    return data_[shape_.index(indices)];
  }

  __HDI__ float& operator[](
      const Functional::Array<int, 4>& indices) {
    return data_[shape_.index(indices)];
  }

  __HDI__ const float& operator[](
      const Functional::Array<int, 4>& indices) const {
    return data_[shape_.index(indices)];
  }

  __HDI__ T* data() { return data_; }
  __HDI__ const T* data() const { return data_; }

  __HDI__ HUConstantShape& shape() { return shape_; }
  __HDI__ const HUConstantShape& shape() const { return shape_; }
};



}

}
