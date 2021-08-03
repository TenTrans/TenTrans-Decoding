#include "HUDevice.h"
#include <vector>
#include "HUCudaHelper.h"

namespace TenTrans{

HUDevice::HUDevice(DeviceId deviceId, size_t alignment)
{
	this->deviceId_ = deviceId;
	setDevice();
	setHandles();

	this->data_ = 0;
	this->size_ = 0;
	this->alignment_ = alignment;
}

void HUDevice::setDevice()
{
	cudaSetDevice((int)deviceId_.no);
}

void HUDevice::synchronize()
{
	cudaStreamSynchronize(0);
}

cublasHandle_t HUDevice::getCublasHandle()
{
	return cublasHandle_;
}

void HUDevice::setHandles()
{
	cublasHandle_ = create_handle();
}

cublasHandle_t HUDevice::create_handle()
{
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);
	return cublasHandle;
}

HUDevice::~HUDevice(){
	CUDA_CHECK(cudaSetDevice(deviceId_.no));
	if(this->data_){
		CUDA_CHECK(cudaFree(data_));
	}
	CUDA_CHECK(cudaDeviceSynchronize());
}

void HUDevice::reserve(size_t size)
{
	size = align(size);
	CUDA_CHECK(cudaSetDevice(deviceId_.no));

	ABORT_IF(size < size_ || size == 0, "[TenTrans] New size must be larger than old size and larger than 0");

  if(data_) {
    // Allocate memory while temporarily parking original content in host memory
    std::vector<uint8_t> temp(size_);
	// Step 1. Copy data from the GPU to Host
    CUDA_CHECK(cudaMemcpy(temp.data(), data_, size_, cudaMemcpyDeviceToHost));
	// Step 2. Free memory on GPU
    CUDA_CHECK(cudaFree(data_));
    LOG(debug, "[TenTrans][memory] Re-allocating from {} to {} bytes on device {}", size_, size, deviceId_.no);
	// Step 3. Re-alloc new size
    CUDA_CHECK(cudaMalloc(&data_, size));
	// Step 4. Copy old data from the Host to GPU
    CUDA_CHECK(cudaMemcpy(data_, temp.data(), size_, cudaMemcpyHostToDevice));
  } else {
    // No data_ yet: Just alloc.
    LOG(debug, "[TenTrans][memory] Allocating {} bytes in device {}", size, deviceId_.no);
    CUDA_CHECK(cudaMalloc(&data_, size));
  }

  size_ = size;
}

uint8_t* HUDevice::data()
{
	return data_;
}

size_t HUDevice::size()
{
	return size_;
}	


}
