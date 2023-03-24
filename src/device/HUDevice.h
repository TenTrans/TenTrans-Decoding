#pragma once
#include "HUGlobal.h"
#include <cuda.h>
#include <cublas_v2.h>
#include <cmath>

#include <cuda_fp16.h>

namespace TenTrans{

class HUDevice{
private:
	DeviceId deviceId_;
	cublasHandle_t cublasHandle_;

	uint8_t* data_{0};
	size_t size_{0};
	size_t alignment_;

	size_t align(size_t size) {
		return ceil(size / (float)alignment_) * alignment_;
	}

public:
	HUDevice(DeviceId deviceId, size_t alignment = 256);
	~HUDevice();

	DeviceId getDeviceId() { return deviceId_; }

	void setDevice();
	void synchronize();
	void setHandles();
	cublasHandle_t getCublasHandle();
	cublasHandle_t create_handle();

	void reserve(size_t size);
	uint8_t* data();
	size_t size();

};

}
