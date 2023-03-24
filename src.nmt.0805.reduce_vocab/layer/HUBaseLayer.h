
#pragma once 
#include <iostream>
#include <string>
#include <vector>
#include "HUGlobal.h"
#include "HUConfig.h"
#include "HUMemory.h"
#include "HUShape.h"
#include "HUDevice.h"
#include "HUTensor.h"
#include "HUData.h"
#include "cnpy.h"

//using namespace std;

namespace TenTrans{
class HUBaseLayer{
public:
	HUBaseLayer(HUPtr<HUConfig> options, HUPtr<HUMemPool> memoryPool, HUPtr<HUDevice> device, cnpy::npz_t modelNpz, bool isEncoder)
			: options_(options),
			  memPool_(memoryPool), 
			  device_(device),
			  modelNpz_(modelNpz),
			  isEncoder_(isEncoder) 
    { 
        this->dataType_ = TENSOR_DATA_TYPE::TT_FLOAT32;
        if (options->get<bool>("use-fp16")) {
            this->dataType_ = TENSOR_DATA_TYPE::TT_FLOAT16;
        }
    }
	virtual ~HUBaseLayer(){}

	virtual void Init() = 0;
		
	//template <typename T>
	//virtual void NewBySuffix(T e){}

	//template <typename T>
	//virtual void InitBySuffix(T e){}

	HUPtr<HUShape> GetShapeByModel(std::string pname, cnpy::npz_t modelNpz);

public:
	HUPtr<HUConfig> options_;
	HUPtr<HUMemPool> memPool_;
	HUPtr<HUDevice> device_;
	cnpy::npz_t modelNpz_;
	bool isEncoder_;
    TENSOR_DATA_TYPE dataType_;
};

}
