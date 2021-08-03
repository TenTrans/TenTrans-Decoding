#pragma once

#include <iostream>
#include <fstream>
#include "HUConfig.h"
#include "HUConfigParser.h"
#include "HUVocab.h"
#include "HUDevice.h"
#include "HUGlobal.h"
#include "HUMemory.h"
#include "cnpy.h"

#include "HUEmbeddingLayer.h"
#include "HUMultiHeadAttention.h"
#include "HULayerNorm.h"
#include "HUFFNLayer.h"
#include "HUOutputLayer.h"
#include "HUUtil.h"
#include "HUData.h"
#include "HUTensorUtil.h"
#include "HUEncoderLayer.h"
#include "HUEncoder.h"

#include "HUDecoder.h"
#include "HUEncoderDecoder.h"
#include "HUBeamCell.h"
#include "HUHistory.h"
#include "HUNthElement.h"
#include "HUBeamSearch.h"
#include "HUResult.h"

#include <cuda_runtime.h>
using namespace TenTrans;

class OnlineDecoder
{
public:
    void Init(string configPath);
    void DoJob(std::vector<string> &inputs, std::vector<string> &outputs);

    OnlineDecoder() {}
    ~OnlineDecoder() {}

private:
    HUPtr<HUConfig> config_;
    HUPtr<HUMemPool> memPool_;
    HUPtr<HUDevice> device_;

    HUPtr<HUVocab> srcVocab_;
    HUPtr<HUVocab> tgtVocab_;
    cnpy::npz_t modelNpz_;

    HUPtr<HUEncoderDecoder> encDec_;
    HUPtr<HUBeamSearch> search_;
};
