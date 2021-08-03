#pragma once
#include <iostream>
#include "HUEncoder.h"
#include "HUOutputLayer.h"

namespace TenTrans{

class Classifier : public HUBaseLayer {

public:
    Classifier(HUPtr<HUConfig> options, HUPtr<HUMemPool> memoryPool, HUPtr<HUDevice> device, cnpy::npz_t modelNpz, bool isEncoder=true);
    ~Classifier();

    void Init();
    // HUPtr<HUTensor> Forward(HUPtr<HUBatch> batch);
    // void Forward(HUPtr<HUBatch> batch, std::vector<size_t> &result);
    void Forward(HUPtr<HUBatch> batch);

public:
    HUPtr<HUEncoder> encoder_;
    HUPtr<HUOutputLayer> output_;
};

}
