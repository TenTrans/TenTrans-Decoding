
#include "HUEncoderDecoder.h"
#include <iostream>
namespace TenTrans{

HUEncoderDecoder::HUEncoderDecoder(HUPtr<HUConfig> options, HUPtr<HUMemPool> memoryPool, HUPtr<HUDevice> device, cnpy::npz_t modelNpz, bool isEncoder)
	: HUBaseLayer(options, memoryPool, device, modelNpz, isEncoder)
{
	this->encoder_ = HUNew<HUEncoder>(options, memoryPool, device, modelNpz);
	this->decoder_ = HUNew<HUDecoder>(options, memoryPool, device, modelNpz);
	this->output_ = HUNew<HUOutputLayer>(options, memoryPool, device, modelNpz);
}

HUEncoderDecoder::~HUEncoderDecoder()
{

}

void HUEncoderDecoder::Init()
{
	encoder_->Init();
	decoder_->Init();
	output_->Init();
}


HUPtr<HUDecoderState> HUEncoderDecoder::PrepareForDecoding(HUPtr<HUBatch> batch)
{
    cudaEvent_t start, stop;
    float elapsedTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

	auto encoderState = this->encoder_->Forward(batch);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Encoder Layer Time Cost(ms): " << elapsedTime << std::endl;

	return this->decoder_->PrepareForDecoding(batch, encoderState);

}

HUPtr<HUDecoderState> HUEncoderDecoder::Step(HUPtr<HUDecoderState> state, size_t* hypIndices, int selIdxSize, std::vector<size_t>& embIndices, int beamSize, int realDimBatch, uint8_t* isAllDone, const std::vector<size_t>& tgtEmbIdx, bool isFirstStep)
{
    // std::cout << "ABC" << std::endl;
	// state = hypIndices.empty() ? state : state->select(hypIndices, isAllDone, beamSize, this->memPool_);
    // state = isFirstStep ? state : state->select(hypIndices, isAllDone, beamSize, this->memPool_);
    state = (selIdxSize == 0) ? state : state->select(hypIndices, selIdxSize, isAllDone, beamSize, this->memPool_);

#ifdef DECODER_DEBUG
    std::cout << "[HUEncoderDecoder][Step] embIndices " << std::endl;
    for(int i = 0; i < embIndices.size(); i++)
    {
        std::cout << embIndices[i] << " ";
    }
    std::cout << std::endl;
#endif

	this->decoder_->EmbeddingsFromPrediction(state, embIndices, state->getEncoderState(), beamSize);

	States decoderStates;
	auto decoderContext = this->decoder_->Step(state, decoderStates, realDimBatch, isAllDone);

#ifdef TOPK_SOFTMAX_FUSION
    HUPtr<HUTensor> logLogits = this->output_->MultiplyW(decoderContext);
    this->memPool_->free(decoderContext->memory());

    int dimBatchBeam = logLogits->shape()[-3];   // logLogits->shape()[-2]==1
    int dimTgtWords = logLogits->shape()[-1];
    logLogits = HUTensorUtil::Reshape(logLogits, {dimBatchBeam, dimTgtWords});

#else   // TOPK_SOFTMAX_FUSION
    HUPtr<HUTensor> logits;
    if (beamSize > 1) {
        //// logits = this->output_->Forward(decoderContext);
        //// logits = this->output_->MultiplyW(decoderContext);

#ifdef TOPK_FUSION
        logits = this->output_->MultiplyW(decoderContext, tgtEmbIdx, isFirstStep);
#else
        logits = this->output_->Forward(decoderContext);
#endif
    }
    else
    {
        logits = this->output_->MultiplyW(decoderContext, tgtEmbIdx, isFirstStep);
    }
	this->memPool_->free(decoderContext->memory());
#ifdef DECODER_DEBUG
    LOG(trace, "[TenTrans][HUEncoderDecoder] Step, logits {}", logits->debug());
#endif
	//std::cout << logits->debug() << std::endl;

    /* [dimBatch*dimBeam, 1, dimTgtWords]*/ 
    // auto logLogits = HUTensorUtil::LogSoftmax(logits, this->memPool_, this->device_);
    // int dimBatchBeam = logLogits->shape()[-3];
    // int dimTgtWords = logLogits->shape()[-1];
    // logLogits = HUTensorUtil::Reshape(logLogits, {dimBatchBeam, dimTgtWords});
	// this->memPool_->free(logits->memory());

    HUPtr<HUTensor> logLogits;
    if (beamSize > 1)
    {
        //// logLogits = HUTensorUtil::LogSoftmax(logits, this->memPool_, this->device_);
        //// logLogits = HUTensorUtil::AddBiasLogSoftmax(logits, this->GetOutputLayerBias(), this->memPool_, this->device_);

#ifdef TOPK_FUSION
        logLogits = HUTensorUtil::AddBiasLogSoftmax(logits, this->GetOutputLayerBias(), realDimBatch, isAllDone, this->memPool_, this->device_);
#else
        logLogits = HUTensorUtil::LogSoftmax(logits, this->memPool_, this->device_);
#endif

        int dimBatchBeam = logLogits->shape()[-3];
        int dimTgtWords = logLogits->shape()[-1];
        logLogits = HUTensorUtil::Reshape(logLogits, {dimBatchBeam, dimTgtWords});
        this->memPool_->free(logits->memory());
    }
    else // if beamSize=1, no need LogSoftmax
    {
        logLogits = logits;
        int dimBatchBeam = logLogits->shape()[-3];
        int dimTgtWords = logLogits->shape()[-1];
        logLogits = HUTensorUtil::Reshape(logLogits, {dimBatchBeam, dimTgtWords});
    }
#endif // TOPK_SOFTMAX_FUSION


#ifdef DECODER_DEBUG
    LOG(trace, "[TenTrans][HUEncoderDecoder] Step, logLogits {}", logLogits->debug());
#endif

	HUPtr<HUDecoderState> nextState = HUNew<HUDecoderState>(decoderStates, logLogits, state->getEncoderState(), state->getBatch(), state->getHeads(), this->memPool_);
	//std::cout << "decoderStates contents" << std::endl;
	//for(int i=0; i< decoderStates.size(); i++)
	//	std::cout << "decoderLayerState.output "<< i << " " << decoderStates[i].output->debug() << std::endl;

	nextState->setPosition(state->getPosition() + 1);
    int startPos = (int)nextState->getPosition();
#ifdef DECODER_DEBUG
    LOG(trace, "[TenTrans][HUEncoderDecoder] Step, nextState.startPos {}", startPos);
#endif

	return nextState;
}

}
