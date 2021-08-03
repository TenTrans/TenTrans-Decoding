#include "NMT.h"

namespace TenTrans{

NMT::NMT(HUPtr<HUConfig> options, HUPtr<HUMemPool> memoryPool, HUPtr<HUDevice> device, cnpy::npz_t modelNpz)
{
    this->encoder_ = HUNew<HUEncoder>(options, memoryPool, device, modelNpz);
    this->decoder_ = HUNew<HUDecoder>(options, memoryPool, device, modelNpz);
    this->output_ = HUNew<HUOutputLayer>(options, memoryPool, device, modelNpz);   
}

NMT::~NMT()
{

}

void NMT::Init()
{
    this->encoder_->Init();
    this->decoder_->Init();
    this->output_->Init();
}

void NMT::Forward(HUPtr<HUBatch> batch)
{
    auto encoderState = this->encoder_->Forward(batch);

    // [dimBatch, dimWords, dimModel] 
    // auto encoderContext = encoderState->getContext();
    // [dimBatch, 1, 1, dimWords] - > [0., 0., 0., -inf]
    // auto encoderMask = 

	
	/*

    // [:, 0, :] -> [dimBatch, dimModel] 
    int dimBatch = encoderContext->shape()[-3];
    int dimWords = encoderContext->shape()[-2];
    int dimModel = encoderContext->shape()[-1];
    auto reshapeEncoderContext = HUTensorUtil::Reshape(encoderContext, {dimBatch * dimWords, dimModel});
    std::vector<size_t> clsIndices(dimBatch, 0);
    for (int i = 0; i < dimBatch; i++) {
        clsIndices[i] = (size_t) i * dimWords;
    }
    auto clsContext = HUTensorUtil::CopyRows(reshapeEncoderContext, clsIndices, this->memPool_);
    this->memPool_->free(reshapeEncoderContext->memory());


    // forward ouput layer 
    auto logits = this->output_->Forward(clsContext);
#ifdef DEBUG_MOD
    LOG(info, "[TenTrans][Classifier][Forward] {}", logits->debug());
#endif
    this->memPool_->free(clsContext->memory());
    this->memPool_->free(logits->memory());
	*/

}

}
