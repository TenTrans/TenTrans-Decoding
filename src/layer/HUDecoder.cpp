
#include "HUDecoder.h"
#include <iostream>
namespace TenTrans{
	
HUDecoder::HUDecoder(HUPtr<HUConfig> options, HUPtr<HUMemPool> memoryPool, HUPtr<HUDevice> device, cnpy::npz_t modelNpz, bool isEncoder)
    : HUBaseLayer(options, memoryPool, device, modelNpz, isEncoder)
{
    LOG(debug, "[TenTrans][HUDecoder] Loading Target Embedding Layer ...");
    this->embedding_ = HUNew<HUEmbeddingLayer>(options, memoryPool, device, modelNpz, isEncoder);
    this->layers_ = options->get<int>("dec-depth");
    this->heads_ = options->get<int>("transformer-heads");
    for(int i = 0; i < this->layers_; i++) {
        LOG(debug, "[TenTrans][HUDecoder] Loading Decoder Layer {}", i);
        HUPtr<HUDecoderLayer> layer = HUNew<HUDecoderLayer>(options, memoryPool, device, modelNpz, i);
        decoderLayers_.push_back(layer);
    }
}

HUDecoder::~HUDecoder()
{
    this->memPool_->free(this->lengths_->memory());
}

void HUDecoder::Init()
{
    LOG(debug, "[TenTrans][HUDecoder] Initialize Target Embedding Layer ...");
    this->embedding_->Init();
    for(int i = 0; i < this->layers_; i++)
    { 
        LOG(debug, "[TenTrans][HUDecoder] Initialize Decoder Layer {}", i);
        decoderLayers_[i]->Init();
    }
}

/*
HUPtr<HUTensor> HUDecoder::Forward(int position, HUPtr<HUTensor> encoderOutput, std::vector<size_t> &embIdx)
{
	LOG(debug, "[TenTrans][HUDecoder] Forward Target Embedding at Position {}...", position);
	auto output = this->embedding_->ForwardDecoder(encoderOutput, embIdx);
	 
	return output;
}*/

HUPtr<HUDecoderState> HUDecoder::PrepareForDecoding(HUPtr<HUBatch> batch, HUPtr<HUEncoderState> encoderState)
{
#ifdef DECODER_DEBUG
    LOG(debug, "[TenTrans][HUDecoder] Prepare for Decoding ...");
#endif
    States startStates;

    // [dimBatch, dimWords, dimEmb]
    HUPtr<HUTensor> encoderContext = encoderState->getContext();
#ifdef DECODER_DEBUG
    LOG(trace, "[TenTrans][HUDecoder][StartDecode] encoderContext {}", encoderContext->debug());
#endif

    /* [dimBatch, dimHeads broadcast=1, dimWords broadcast=1, dimWords] -> [0., 0., 0., -inf] */
    HUPtr<HUTensor> encoderMask = encoderState->getMask();
#ifdef DECODER_DEBUG
    LOG(trace, "[TenTrans][HUDecoder][StartDecode] encoderMask {}", encoderMask->debug());
#endif

    this->encoderContext_ = encoderContext;
    this->encoderMask_ = encoderMask;

    auto sourceLengths = batch->lengths();
    auto lengthsMem = this->memPool_->alloc<float>(sourceLengths.size());
    auto lengthsShape = HUShape({sourceLengths.size()});
    this->lengths_ = HUNew<HUTensor>(lengthsMem, lengthsShape, this->device_);
    this->lengths_->set(sourceLengths);

    return HUNew<HUDecoderState>(startStates, nullptr, encoderState, batch, this->heads_, this->memPool_);
}

HUPtr<HUDecoderState> HUDecoder::StartDecode_v2(HUPtr<HUBatch> batch, HUPtr<HUEncoderState> encoderState)
{
	LOG(debug, "[TenTrans][HUDecoder] Start Decoder");
	States startStates;
	int dimBatch = batch->batchSize();

    /* [dimBatch, dimWords, dimEmb] */
	HUPtr<HUTensor> encoderContext = encoderState->getContext();
#ifdef DECODER_DEBUG
    LOG(trace, "[TenTrans][HUDecoder][StartDecode] encoderContext {}", encoderContext->debug());
#endif

    /* [dimBatch, 1, 1, dimWords] -> [0., 0., 0., -inf] */
    HUPtr<HUTensor> encoderMask = encoderState->getMask();
#ifdef DECODER_DEBUG
    LOG(trace, "[TenTrans][HUDecoder][StartDecode] encoderMask {}", encoderMask->debug());
#endif

    /*
    int dimBatch = encoderContext->shape()[-3];
    int dimWords = encoderContext->shape()[-2];
    int dimModel = encoderContext->shape()[-1];
    */
    // auto beamEncoderContext = HUTensorUtil::Reshape(encoderContext, {});


    //[-4: beam depth=1, -3: batch size, -2: max length, -1: vector dim]
    auto encoderContext1 = HUTensorUtil::TransposeTimeBatch(encoderContext, this->memPool_, device_);
	//this->memPool_->free(encoderContext->memory());
    //std::cout << "test4" << std::endl;
    //std::cout << "transpose " << encoderContext1->debug() << std::endl;
    encoderContext = encoderContext1;

    int dimSrcWords = encoderContext1->shape()[-2];
    encoderMask = HUTensorUtil::AtLeastNd(encoderMask, 4);
	//std::cout << "in StartDecode 1" << encoderMask->debug() << std::endl;
    auto encoderMask1 = HUTensorUtil::Reshape(HUTensorUtil::TransposeTimeBatch(encoderMask, this->memPool_, device_), {1, dimBatch, 1, dimSrcWords});    //this->memPool_->free(encoderMask->memory());
	//std::cout << "in StartDecode 2 " << encoderMask1->debug() << std::endl;
	//[-4: batch size, -3: 1, -2: vector dim=1, -1: max length]
	auto encoderMask2 = HUTensorUtil::TransposedLogMask(encoderMask1, this->memPool_, device_);
	//std::cout << "in StartDecode 3 " << encoderMask2->debug() << std::endl;
	this->memPool_->free(encoderMask1->memory());

    encoderMask = encoderMask2;
	this->encoderContext_ = encoderContext;
	this->encoderMask_ = encoderMask;
    //encoderMask = HUTensorUtil::TransposedLogMask(encoderMask1, this->memPool_, device_);
    //this->memPool_->free(encoderMask1->memory());
    //std::cout << encoderMask

    /*if(dimBeam > 1)
    {
        auto encoderMask2 = HUTensorUtil::Repeat(encoderMask, dimBeam, -4, this->memPool_);
        //this->memPool_->free(encoderMask->memory());
        encoderMask = encoderMask2;
    }*/

	return HUNew<HUDecoderState>(startStates, nullptr, encoderState, batch, this->heads_, this->memPool_);
}

void HUDecoder::EmbeddingsFromPrediction(HUPtr<HUDecoderState> state, std::vector<size_t>& embIdx, HUPtr<HUEncoderState> encState, int beamSize)
{
	auto encoderOutput = encState->getContext();
#ifdef DECODER_DEBUG
    LOG(trace, "[TenTrans][HUDecoder][EmbeddingsFromPrediction] encoderOutput {}", encoderOutput->debug());
#endif

    size_t startPos = state->getPosition();
	auto selectedEmbs = this->embedding_->ForwardDecoder(encoderOutput, embIdx, beamSize, startPos);
	state->setTargetEmbeddings(selectedEmbs);
}

HUPtr<HUTensor> HUDecoder::Step(HUPtr<HUDecoderState> state, States& decoderStates, int realDimBatch, uint8_t* isAllDone)
{
	//[dimBatch * dimBeam, 1, dimTrgEmb]
	auto addPosEmbedding = state->getTargetEmbeddings();
    int startPos = (int)state->getPosition();

	// int dimEmb = embeddings->shape()[-1];
    // int dimBeam = embeddings->shape()[-3];
    /*
	int dimBeam = 1;
    if(embeddings->shape().size() > 3)
		dimBeam = embeddings->shape()[-3];
    */

    /* wheather scale ? */
    /*
    bool useEmbedScale = this->options_->get<bool>("use-emb-scale");
    if (useEmbedScale) {
        HUTensorUtil::ScaleAndShift(embeddings, std::sqrt((float)dimEmb), 0.0);
#ifdef DECODER_DEBUG
        LOG(trace, "[TenTrans][HUEmbeddingLayer][Forward] ScaleAndShift {}", embeddings->debug());
#endif
    } */

    /*
	int startPos = (int)state->getPosition();
    // std::cout << "[HUDecoder][Step] startPos: " << startPos << std::endl;
	auto addPosEmbedding = this->embedding_->AddPositinoalEmbeddings(embeddings, startPos);
	this->memPool_->free(embeddings->memory());
    */
#ifdef DECODER_DEBUG
    LOG(trace, "[TenTrans][HUDecoder][Step] addPosEmbedding {}", addPosEmbedding->debug());
#endif
    auto lnEmbeddings = this->embedding_->ForwardLayerNorm(addPosEmbedding);
    this->memPool_->free(addPosEmbedding->memory());

#ifdef DECODER_DEBUG
    LOG(trace, "[TenTrans][HUDecoder][Step] lnEmbedding {}", lnEmbeddings->debug());
#endif

    /* [dimBatch * dimBeam, 1, dimTrgEmb]*/
    auto query = lnEmbeddings;
    // [dimBatch * dimBeam, 1, 1] -> [batch*beam, broadcast dimHeads=1, 1, 1] */
    auto selfMask_tmp = HUTensorUtil::Ones({query->shape()[-3], 1, 1}, this->memPool_, device_);
#ifdef DECODER_DEBUG
    LOG(trace, "[TenTrans][HUDecoder][Step] selfMask_tmp {}", selfMask_tmp->debug());
#endif
    auto selfMask = HUTensorUtil::TransposedLogMask(selfMask_tmp, this->memPool_, device_);
#ifdef DECODER_DEBUG
    LOG(trace, "[TenTrans][HUDecoder][Step] selfMask {}", selfMask->debug());
#endif
    this->memPool_->free(selfMask_tmp->memory());

    /* [dimBatch, 1, 1, dimWords] */
	auto encoderMask = this->encoderMask_;
	if(startPos > 0)
	{
        int dimBatch = encoderMask->shape()[-4];
        int dimSrcWords = encoderMask->shape()[-1];
        /* [dimBatch, dimHeads broadcast=1, curStep=1, dimSrcWords]  -> [dimBatch*dimBeam, 1, curStep=1, dimSrcWords] */

        int dimBeam = query->shape()[-3] / dimBatch;
		auto encoderMask2 = HUTensorUtil::Reshape(HUTensorUtil::Repeat(this->encoderMask_, dimBeam, -3, this->memPool_), 
                {dimBatch*dimBeam, 1, 1, dimSrcWords});
		//std::cout << "in Step 1 " << encoderMask2->debug() << std::endl;
		//this->memPool_->free(encoderMask->memory());
		encoderMask = encoderMask2;
	}
	
	States prevDecoderStates = state->getStates();

	// apply decoder layers
	auto decDepth = this->options_->get<int>("dec-depth");
	for(int i = 0; i < decDepth; i++)
	{
		State prevDecoderState;
		if(prevDecoderStates.size() > 0) {
        	prevDecoderState = prevDecoderStates[i];
        }

		State decoderState;
#ifdef DECODER_DEBUG
		LOG(debug, "[TenTrans][HUDecoder] Forward Decoder Layer {}", i);
#endif
		query = decoderLayers_[i]->Forward(query, selfMask, decoderState, prevDecoderState, startPos, encoderContext_, encoderMask, this->lengths_, realDimBatch, isAllDone);
		decoderStates.push_back(decoderState);
	}
    this->memPool_->free(selfMask->memory());
    // std::cout << "[selfMask] free ..." << std::endl;
    
	
	if(startPos > 0) {
		this->memPool_->free(encoderMask->memory());
        // std::cout << "[encoderMask] free ..." << std::endl;
    }
    	
    return query;
}

}

