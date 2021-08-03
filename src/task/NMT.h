#pragma once
#include <iostream>
#include "HUEncoder.h"
#include "HUDecoder.h"
#include "HUOutputLayer.h"

namespace TenTrans{

class NMT {
public:
    NMT(HUPtr<HUConfig> options, HUPtr<HUMemPool> memoryPool, HUPtr<HUDevice> device, cnpy::npz_t modelNpz);
    ~NMT();

    void Init();
    void Forward(HUPtr<HUBatch> batch);

public:
    HUPtr<HUEncoder>     encoder_;
    HUPtr<HUDecoder>     decoder_;
    HUPtr<HUOutputLayer> output_;
};

}
