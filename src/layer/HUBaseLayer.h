/*
 * Author: Danielkxwu
 * E-mial: danielkxwu@tencent.com
 * Created Date: 2021/4/2
 *
 */

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

namespace TenTrans
{

class HUBaseLayer
{

public:
	HUBaseLayer(HUPtr<HUConfig> options, HUPtr<HUMemPool> memoryPool, HUPtr<HUDevice> device, cnpy::npz_t modelNpz, bool isEncoder)
			: options_(options),
			  memPool_(memoryPool), 
			  device_(device),
			  modelNpz_(modelNpz),
			  isEncoder_(isEncoder)
    {

    }

	virtual ~HUBaseLayer()
    {

    }

	virtual void Init() = 0;
	HUPtr<HUShape> GetShapeByModel(std::string pname, cnpy::npz_t modelNpz);

public:
	HUPtr<HUConfig> options_;
	HUPtr<HUMemPool> memPool_;
	HUPtr<HUDevice> device_;
	cnpy::npz_t modelNpz_;
	bool isEncoder_;

};

}
