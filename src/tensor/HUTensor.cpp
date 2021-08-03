#include "HUTensor.h"

namespace TenTrans{

	HUTensor::HUTensor(HUPtr<HUMemoryPiece> memory, HUShape shape, HUPtr<HUDevice> device){
		this->memory_ = memory;
		this->shape_ = shape;
		this->device_ = device;
/*
#ifdef CUDA_FOUND
		gpu::alloc(this->data(), (size_t)this->size());
#endif
*/
	}

	HUTensor::HUTensor(const int myOrder, const int * myDimSize, HUPtr<HUMemoryPiece> memory, HUPtr<HUDevice> device){
		this->memory_ = memory;
		this->device_ = device;
		HUShape* newShape = new HUShape();
		newShape->resize(myOrder);
		for(int i=0; i< myOrder; i++)
			newShape->set(i, myDimSize[i]);
		this->shape_ = *newShape;
	}

	HUTensor::HUTensor(const int myOrder, const int * myDimSize, HUPtr<HUMemoryPiece> memory, HUPtr<HUDevice> device, HUPtr<HUMemPool> mem)
	{
		this->mem_ = mem;
		HUShape* newShape = new HUShape();
		newShape->resize(myOrder);
		for(int i=0; i< myOrder; i++)
			newShape->set(i, myDimSize[i]);
		auto memPiece = this->mem_->alloc<float>(newShape->elements());
		HUTensor(memPiece, *newShape, device);
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

	float* HUTensor::data(){
		return (float*)memory_->data();
	}

	size_t HUTensor::size(){
		return this->shape_.elements();
	}

	float HUTensor::scalar(){
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

	float HUTensor::get(size_t i){
		float temp;
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

	void HUTensor::set(size_t i, float value){
		if(device_->getDeviceId().type == DeviceType::cpu) {
      		std::copy(&value, &value + 1, data() + i);
    	}
#ifdef CUDA_FOUND
    	else {
      		gpu::copy(device_, &value, &value + 1, data() + i);
    	}
#endif
	}

	void HUTensor::get(std::vector<float>& v){
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

	void HUTensor::set(const float* begin, const float* end){
		if(device_->getDeviceId().type == DeviceType::cpu) {
      		std::copy(begin, end, data());
    	}
#ifdef CUDA_FOUND
    	else {
			//cout << "test " << endl;
      		gpu::copy(device_, begin, end, data());
    	}
#endif
	}

	void HUTensor::set(const std::vector<float>& v){
		set(v.data(), v.data() + v.size());
	}

	void HUTensor::toCuda()
	{
#ifdef CUDA_FOUND
		gpu::copy(device_, data(), data() + size(), data());
#endif
	}

	void HUTensor::set(float value){
		if(device_->getDeviceId().type == DeviceType::cpu) {
      		std::fill(data(), data() + size(), value);
    	}
#ifdef CUDA_FOUND
   		else {
      		gpu::fill(device_, data(), data() + size(), value);
    	}
#endif
	}

	void HUTensor::copyFrom(HUTensor in){
		if(in.getDevice()->getDeviceId().type == DeviceType::cpu && device_->getDeviceId().type == DeviceType::cpu) {
      std::copy(in.data(), in.data() + in.size(), data());
    }
#ifdef CUDA_FOUND
    else {
      gpu::copy(device_, in.data(), in.data() + in.size(), data());
    }
#endif
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
    	std::vector<float> values(totSize);
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

        		strm << std::setw(12) << values[i] << " ";

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
