
#pragma once
#include<iostream>
#include "HUTensor.h"

namespace TenTrans{

/*
 * return Top-beamSize result 
 *
 * beamSizes: [dimBatch], TopKs
 * outCosts(squence score): [dimBatch* beamSize]
 * outKeys(word id):        [dimBatch* beamSize]
 *
 */
typedef std::function<void(const std::vector<size_t>& beamSizes,
                           HUPtr<HUTensor> logProbs,
                           std::vector<float>& outCosts,
                           std::vector<unsigned>& outKeys,
                           const bool isFirst)> GetNBestListFn;

GetNBestListFn createGetNBestListFn(size_t beamSize, size_t dimBatch, DeviceId deviceId);
}

