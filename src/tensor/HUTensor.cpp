#include "HUTensor.h"

namespace TenTrans{

	HUTensor::HUTensor(HUPtr<HUMemoryPiece> memory, HUShape shape, HUPtr<HUDevice> device, TENSOR_DATA_TYPE dataType) {
		this->memory_ = memory;
		this->shape_ = shape;
		this->device_ = device;

        this->dataType_ = dataType;
        setUnitSize();
        /*
        if (dataType == TENSOR_DATA_TYPE::TT_INT32) {
            this->unitSize_ = sizeof(int);
        }
        if (dataType == TENSOR_DATA_TYPE::TT_FLOAT32) {
            this->unitSize_ = sizeof(float);
        }
        else if (dataType == TENSOR_DATA_TYPE::TT_DOUBLE) {
            this->unitSize_ = sizeof(double);
        }
        else if (dataType == TENSOR_DATA_TYPE::TT_FLOAT16) {
            this->unitSize_ = 2;
        }
        else if (dataType == TENSOR_DATA_TYPE::TT_INT8) {
            this->unitSize_ = 1;
        }
        else {
            this->unitSize_ = sizeof(float);
        } */
	}

	HUTensor::HUTensor(const int myOrder, const int* myDimSize, HUPtr<HUMemoryPiece> memory, HUPtr<HUDevice> device, \
            TENSOR_DATA_TYPE dataType) 
    {
		this->memory_ = memory;
		this->device_ = device;
		HUShape* newShape = new HUShape();
		newShape->resize(myOrder);
		for(int i=0; i< myOrder; i++)
			newShape->set(i, myDimSize[i]);
		this->shape_ = *newShape;

        this->dataType_ = dataType;
        setUnitSize();
	}

	HUTensor::HUTensor(const int myOrder, const int* myDimSize, HUPtr<HUMemoryPiece> memory, HUPtr<HUDevice> device, \
            HUPtr<HUMemPool> mem, TENSOR_DATA_TYPE dataType)
	{
		this->mem_ = mem;
		HUShape* newShape = new HUShape();
		newShape->resize(myOrder);
		for(int i=0; i< myOrder; i++)
			newShape->set(i, myDimSize[i]);
		auto memPiece = this->mem_->alloc<TT_DATA_TYPE>(newShape->elements());
		HUTensor(memPiece, *newShape, device);

        this->dataType_ = dataType;
        setUnitSize();
	}

    void HUTensor::setUnitSize()
    {
        if (this->dataType_ == TENSOR_DATA_TYPE::TT_INT32) {
            this->unitSize_ = sizeof(int);
        }
        else if (this->dataType_ == TENSOR_DATA_TYPE::TT_FLOAT32) {
            this->unitSize_ = sizeof(float);
        }
        else if (this->dataType_ == TENSOR_DATA_TYPE::TT_DOUBLE) {
            this->unitSize_ = sizeof(double);
        }
        else if (this->dataType_ == TENSOR_DATA_TYPE::TT_FLOAT16) {
            this->unitSize_ = 2;
        }
        else if (this->dataType_ == TENSOR_DATA_TYPE::TT_INT8) {
            this->unitSize_ = 1;
        }
        else {
            this->unitSize_ = sizeof(float);
        }
    }

	void HUTensor::reset(HUPtr<HUMemoryPiece> memory){
		this->memory_ = memory;
	}

	HUPtr<HUMemoryPiece> HUTensor::memory(){
		return this->memory_;
	}

	HUShape& HUTensor::shape(){
		return this->shape_;
	}

    // template <typename T>
	TT_DATA_TYPE* HUTensor::data(){
		return (TT_DATA_TYPE*)memory_->data();
	}

	size_t HUTensor::size(){
		return this->shape_.elements();
	}

    // template <typename T>
	TT_DATA_TYPE HUTensor::scalar(){
		ABORT_IF(size() != 1, "Tensor is not a scalar");
		return get(0);
	}

	HUPtr<HUDevice> HUTensor::getDevice(){
		return this->device_;
	}

	DeviceId HUTensor::getDeviceId(){
		return this->device_->getDeviceId();
	}

	HUTensor HUTensor::subtensor(int offset, int size){
		auto mem = HUNew<HUMemoryPiece>(this->memory_->data() + sizeof(float) * offset, sizeof(float) * size);
		return HUTensor(mem, HUShape{1, size}, device_);
	}

    // template <typename T>
	TT_DATA_TYPE HUTensor::get(size_t i){
        TT_DATA_TYPE temp;
    	if(device_->getDeviceId().type == DeviceType::cpu) {
      		std::copy(data() + i, data() + i + 1, &temp);
    	}
#ifdef CUDA_FOUND
		else {
			gpu::copy(device_, data() + i, data() + i + 1, &temp);
		}
#endif
		return temp;
	}

    // template <typename T>
	void HUTensor::set(size_t i, TT_DATA_TYPE value){
		if(device_->getDeviceId().type == DeviceType::cpu) {
      		std::copy(&value, &value + 1, data() + i);
    	}
#ifdef CUDA_FOUND
    	else {
      		gpu::copy(device_, &value, &value + 1, data() + i);
    	}
#endif
	}

    // template <typename T>
	void HUTensor::get(std::vector<TT_DATA_TYPE>& v){
		v.resize(size());
    	if(device_->getDeviceId().type == DeviceType::cpu) {
      		std::copy(data(), data() + size(), v.data());
    	}
#ifdef CUDA_FOUND
    	else {
      		gpu::copy(device_, data(), data() + size(), v.data());
    	}
#endif
	}

    // template <typename T>
	void HUTensor::set(const TT_DATA_TYPE* begin, const TT_DATA_TYPE* end){
		if(device_->getDeviceId().type == DeviceType::cpu) {
      		std::copy(begin, end, data());
    	}
#ifdef CUDA_FOUND
    	else {
            //// std::cout << "[begin, end] CUDA FOUND ..." << std::endl;
      		gpu::copy(device_, begin, end, data());
    	}
#endif
	}

    /*
    void HUTensor::defaultSet(const void* dataPtr, size_t size) {
        if(device_->getDeviceId().type == DeviceType::cpu) 
        {
            const float* begin = (float*)dataPtr;
            const float* end = begin + size;
            std::copy(begin, end, data());
        }
#ifdef CUDA_FOUND
        else 
        {
            //// std::cout << "[begin, end] CUDA FOUND ..." << std::endl;
            if (this->dataType_ == TENSOR_DATA_TYPE::TT_FLOAT16) // fp16
            {
                const half* begin = (half*)dataPtr;
                const half* end = begin + size;
                gpu::copy(device_, begin, end, (half*)data());
            }
            else                                          // fp32, int32 ... 
            {
                const float* begin = (float*)dataPtr;
                const float* end = begin + size;
                gpu::copy(device_, begin, end, data());
            }
            // gpu::copy(device_, begin, end, data());
        }
#endif
    }
    */

    // template <typename T>
	void HUTensor::set(const std::vector<float>& v) {
        TT_DATA_TYPE* p = new TT_DATA_TYPE[v.size()]; 
        for (int i = 0; i < v.size(); i++) {
            p[i] = (TT_DATA_TYPE)v[i];
        }
        set(p, p + v.size());
        delete [] p;

        // set(v.data(), v.data() + v.size());
	}

	void HUTensor::toCuda()
	{
#ifdef CUDA_FOUND
        //// std::cout << "[to cuda] CUDA FOUND ..." << std::endl;
		gpu::copy(device_, data(), data() + size(), data());
#endif
	}

    // template <typename T>
	void HUTensor::set(TT_DATA_TYPE value){
		if(device_->getDeviceId().type == DeviceType::cpu) {
      		std::fill(data(), data() + size(), value);
    	}
#ifdef CUDA_FOUND
   		else {
            //// std::cout << "[set value] CUDA FOUND ..." << std::endl;
      		gpu::fill(device_, data(), data() + size(), value);
    	}
#endif
	}

	void HUTensor::copyFrom(HUTensor in) {
	    if(in.getDevice()->getDeviceId().type == DeviceType::cpu && device_->getDeviceId().type == DeviceType::cpu) {
            std::copy(in.data(), in.data() + in.size(), data());
        }
#ifdef CUDA_FOUND
        else {
            gpu::copy(device_, in.data(), in.data() + in.size(), data());
        }
#endif
        this->dataType_ = in.getDataType();
        this->unitSize_ = in.getUnitSize();
    }

	std::string HUTensor::debug()
	{
		std::stringstream strm;
    	assert(shape_.size());
    	strm << shape_;
    	strm << " device=" << device_->getDeviceId();
    	strm << " ptr=" << (size_t)memory_->data();
    	strm << " bytes=" << memory_->size();
    	strm << std::endl;

    	// values
    	size_t totSize = shape_.elements();
    	std::vector<TT_DATA_TYPE> values(totSize);
    	get(values);

    	size_t dispCols = 5;
    	strm << std::fixed << std::setprecision(6) << std::setfill(' ');

    	for(int i = 0; i < values.size(); ++i) {
      		std::vector<int> dims;
      		shape().dims(i, dims);

      		bool disp = true;
      		for(int j = 0; j < dims.size(); ++j)
        		disp = disp && (dims[j] < dispCols || dims[j] >= shape()[j] - dispCols);

      		if(disp) {
        		if(dims.back() == 0) {
          			bool par = true;
          			std::vector<std::string> p;
          			for(int j = dims.size() - 1; j >= 0; --j) {
            			if(dims[j] != 0)
              				par = false;

            			p.push_back(par ? "[" : " ");
          			}
          			for(auto it = p.rbegin(); it != p.rend(); ++it)
            			strm << *it;
          			strm << " ";
        		}

        		strm << std::setw(12) << (float)values[i] << " ";

        		if(dims.back() + 1 == shape().back()) {
          			for(int j = dims.size() - 1; j >= 0; --j) {
            			if(dims[j] + 1 != shape()[j])
              				break;
            			strm << "]";
          			}
          			strm << std::endl;
        		}

        		bool prev = true;
        		for(int j = dims.size() - 1; j >= 0; --j) {
          			if(j < dims.size() - 1)
            			prev = prev && dims[j + 1] + 1 == shape()[j + 1];
          			if(prev && dims[j] + 1 == dispCols && shape()[j] > 2 * dispCols) {
            			if(j < dims.size() - 1)
              				for(int k = 0; k <= j; ++k)
                				strm << " ";
            			strm << "... ";
            			if(j < dims.size() - 1)
              				strm << std::endl;
            			break;
          			}
        		}
      		}
    	}
    	strm << std::endl;
    	return strm.str();
	}
}
