/*
 * Author: Danielkxwu
 * E-mial: danielkxwu@tencent.com
 * Created Date: 2021/4/8
 *
 */

#include "HUEmbeddingLayer.h"

namespace TenTrans
{

HUEmbeddingLayer::HUEmbeddingLayer(HUPtr<HUConfig> options, HUPtr<HUMemPool> memoryPool, HUPtr<HUDevice> device, cnpy::npz_t modelNpz, bool isEncoder): 
    HUBaseLayer(options, memoryPool, device, modelNpz, isEncoder)
{
    this->lnLayer_ = HUNew<HULayerNorm>(options, memoryPool, device, modelNpz, isEncoder, -1, false);
    this->useEmbedScale_ = this->options_->get<bool>("use-emb-scale");
    this->isSharedEmbed_ = this->options_->get<bool>("share-all-embed");
}

void HUEmbeddingLayer::Init()
{
    /* initialization word embedding */
    string prefix = this->isEncoder_ ? "encoder.embedding.weight" : "decoder.embedding.weight";
    if (!this->isEncoder_ && this->isSharedEmbed_) {
        prefix = "encoder.embedding.weight";
    }

    auto np = modelNpz_[prefix];
    HUPtr<HUShape> embShape = GetShapeByModel(prefix, this->modelNpz_);
    auto mem = this->memPool_->alloc<float>(embShape->elements()); 
    this->wordEmbedding_ = HUNew<HUTensor>(mem, *embShape, device_);
#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][EmbeddingLayer] Loading [{}] parameters, {}", prefix, embShape->toString());
#endif

    /* loading word embedding data */
    size_t size = 1; 
    for(size_t dim : np->shape) {
        size *= dim;
    }
    this->wordEmbedding_->set((float*)np->data(), (float*)np->data() + size);
#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][{}] parameters, {}", prefix, (this->wordEmbedding_)->debug());
#endif
    
    /* initialization positional embedding */
    InitPosEmbeddings();

    /* initialization for layer normalization */
    this->lnLayer_->Init();
}

void HUEmbeddingLayer::InitPosEmbeddings() {
    bool isLearnedPos = this->options_->get<bool>("learned-pos");
    
    if (isLearnedPos)     
    {
        string prefix = this->isEncoder_ ? "encoder.pe.pe.weight" : "decoder.pe.pe.weight";
        auto np = modelNpz_[prefix];
        HUPtr<HUShape> embShape;
        embShape = GetShapeByModel(prefix, this->modelNpz_);
        auto mem = this->memPool_->alloc<float>(embShape->elements());
        this->posEmbedding_ = HUNew<HUTensor>(mem, *embShape, device_);

        size_t size = 1;
        for(size_t dim : np->shape) {
            size *= dim;
        }
        this->posEmbedding_->set((float*)np->data(), (float*)np->data() + size);
#ifdef DEBUG_MOD
        LOG(trace, "[TenTrans][EmbeddingLayer] Loading Learned PostionalEmbedding [{}] parameters, {}, {}", \
                prefix, embShape->toString(), (this->posEmbedding_)->debug());
#endif
    }
    else 
    {
        /*
         * PE(pos, 2i)   = sin(pos / exp(10000, 2i/dimEmb))
         * PE(pos, 2i+1) = cos(pos / exp(10000, 2i/dimEmb))
         * ==> for example, dimEmb=512, i in [0, dimEmb/2]
         * PE(1) = [sin(1/exp(10000, 0/512)), cos(1/exp(10000, 0/512)), sin(1/exp(10000, 2/512)), cos(1/exp(10000, 2/512))]
         */
        int dimEmb = this->wordEmbedding_->shape()[-1];
        int maxSeqLength = this->options_->get<size_t>("max-seq-length");
        float num_timescales = dimEmb / 2.0f;
        float log_timescale_increment = std::log(10000.f) / num_timescales;

        std::vector<float> vPos(maxSeqLength * dimEmb, 0.f);
        for (int pos = 0; pos < maxSeqLength; ++pos)
        {
            for (int i = 0; i < (int)num_timescales; ++i)
            {
                float v = pos * std::exp(i * -log_timescale_increment);
                vPos[pos*dimEmb + 2*i] = std::sin(v);
                vPos[pos*dimEmb + 2*i+1] = std::cos(v);
            }
        }

        HUShape embShape = HUShape({maxSeqLength, dimEmb});
        auto mem = this->memPool_->alloc<float>(embShape.elements());
        this->posEmbedding_ = HUNew<HUTensor>(mem, embShape, device_);
        this->posEmbedding_->set(vPos);
#ifdef DEBUG_MOD
        LOG(trace, "[TenTrans][EmbeddingLayer] Loading Sinusoidal PostionalEmbedding, {}, {}", \
                embShape.toString(), (this->posEmbedding_)->debug());
#endif
    }
}

HUEmbeddingLayer::~HUEmbeddingLayer() 
{
    this->memPool_->free(wordEmbedding_->memory());
    this->memPool_->free(posEmbedding_->memory());
}

void HUEmbeddingLayer::Forward(HUPtr<HUBatch> batch, HUPtr<HUTensor> &batchEmb, HUPtr<HUTensor> &batchMask)
{
    int dimBatch = (int)batch->batchSize();
    int dimWords = (int)batch->batchWidth();
    int dimEmb = this->wordEmbedding_->shape()[-1];

    // start from 0
    auto scaledEmbeddings = HUTensorUtil::StartIdEmbeddingLookUpPositionEncoding(this->wordEmbedding_, 
            this->posEmbedding_, batch->data(), dimBatch, this->useEmbedScale_, this->memPool_, this->device_); 
    /* get word embedding */
    /*
    auto indices = batch->data();
    auto chosenEmbeddings = HUTensorUtil::CopyRows(this->wordEmbedding_, indices, this->memPool_);   // [dimBatch*dimWords, dimEmb]
#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUEmbeddingLayer][Forward] CopyRows {}", chosenEmbeddings->debug());
#endif
    auto batchEmbeddings = HUTensorUtil::Reshape(chosenEmbeddings, { dimBatch, dimWords, dimEmb });  // [dimBatch, dimWords, dimEmb]
#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUEmbeddingLayer][Forward] Reshape {}", batchEmbeddings->debug());
#endif
    */

    /* batch mask for handling padding tokens */
    HUShape maskShape = HUShape({dimBatch, 1, dimWords});
    auto maskMem = this->memPool_->alloc<float>(maskShape.elements());
    auto maskEmbeddings = HUNew<HUTensor>(maskMem, maskShape, device_);
    maskEmbeddings->set((float*)batch->mask().data(), (float*)batch->mask().data() + batch->mask().size());	
#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUEmbeddingLayer][Forward] maskEmbeddings {}", maskEmbeddings->debug());
#endif

    /* wheather scale ? */
    // bool useEmbedScale = this->options_->get<bool>("use-emb-scale");
    /*
    if (this->useEmbedScale_) 
    { 
        HUTensorUtil::ScaleAndShift(batchEmbeddings, std::sqrt((float)dimEmb), 0.0);
#ifdef DEBUG_MOD
        LOG(trace, "[TenTrans][HUEmbeddingLayer][Forward] ScaleAndShift {}", batchEmbeddings->debug());
#endif
    } */

    /* add positional embeddings */
    /*
    auto scaledEmbeddings = AddPositinoalEmbeddings(batchEmbeddings);
    this->memPool_->free(batchEmbeddings->memory());
#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUEmbeddingLayer][Forward] AddPositinoalEmbeddings {}", scaledEmbeddings->debug());
#endif
     */

    /* layer normalization */
    auto layerNormEmbeddings = this->lnLayer_->Forward(scaledEmbeddings, 1e-12);
    this->memPool_->free(scaledEmbeddings->memory());
#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUEmbeddingLayer][Forward] Layer Normlization {}", layerNormEmbeddings->debug());
#endif
    
    // [dimBatch, dimWords, dimEmb]
    batchEmb = layerNormEmbeddings;
#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUEmbeddingLayer][Forward] batchEmb {}", batchEmb->debug());	
#endif

    // [-4: dimBatch, -3: numHeads broadcast=1, -2: dimWords broadcast=1, -1: dimWords] -> [0., 0., 0., -inf]
    auto transposedLayerMask = HUTensorUtil::TransposedLogMask(maskEmbeddings, this->memPool_, device_);
    this->memPool_->free(maskEmbeddings->memory());

    batchMask = transposedLayerMask;
#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUEmbeddingLayer][Forward] batchMask {}", batchMask->debug());
#endif
}

HUPtr<HUTensor> HUEmbeddingLayer::AddPositinoalEmbeddings(HUPtr<HUTensor> input, int start)
{
    int dimWords = input->shape()[-2];
    int dimEmb = input->shape()[-1];

    std::vector<size_t> lengthIndices(dimWords, start);
    for(size_t i = 0; i < (size_t)dimWords; i++) {
        lengthIndices[i] = start + i;
    }

    /* [dimWords, dimEmb] */
    auto chosenEmbeddings = HUTensorUtil::CopyRows(this->posEmbedding_, lengthIndices, this->memPool_);
#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUEmbeddingLayer][AddLearnedPositinoalEmbeddings] {}", chosenEmbeddings->debug());
#endif

    auto signal = HUTensorUtil::Reshape(chosenEmbeddings, { 1, dimWords, dimEmb });
#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUEmbeddingLayer][AddLearnedPositinoalEmbeddings] {}", signal->debug());
#endif

    auto c = HUTensorUtil::Plus(input, signal, this->memPool_, device_);
    this->memPool_->free(signal->memory());

    return c;
}

HUPtr<HUTensor> HUEmbeddingLayer::ForwardLayerNorm(HUPtr<HUTensor> input)
{
    auto layerNormEmbeddings = this->lnLayer_->Forward(input, 1e-12);
    return layerNormEmbeddings;
}


HUPtr<HUTensor> HUEmbeddingLayer::ForwardDecoder(HUPtr<HUTensor> encoderOutput, std::vector<size_t> &embIdx, int beamSize, size_t startPos)
{
    int dimBatch = encoderOutput->shape()[-3];
    int dimTrgEmb = encoderOutput->shape()[-1];
    int dimBeam = beamSize;

    // std::cout << "[test4]" << std::endl;
    HUPtr<HUTensor> selectedEmbs;
    if (embIdx.empty())   // first time
    {
        embIdx.resize(dimBatch, BOS_ID);
        selectedEmbs = HUTensorUtil::EmbeddingLookUpPositionEncoding(this->wordEmbedding_, this->posEmbedding_, 
                embIdx, 0, this->useEmbedScale_, this->memPool_, this->device_);
        // selectedEmbs = HUTensorUtil::Reshape(selectedEmbs, {dimBatch, 1, dimTrgEmb});
    }
    else
    {
        selectedEmbs = HUTensorUtil::EmbeddingLookUpPositionEncoding(this->wordEmbedding_, this->posEmbedding_, 
                embIdx, startPos, this->useEmbedScale_, this->memPool_, this->device_);
        // selectedEmbs = HUTensorUtil::Reshape(selectedEmbs, {dimBatch*dimBeam, 1, dimTrgEmb});
    }
#ifdef DECODER_DEBUG
    LOG(trace, "[TenTrans][HUEmbeddingLayer][ForwardDecoder] selectedEmbs {}", selectedEmbs->debug());
#endif

    return selectedEmbs;
}

HUPtr<HUTensor> HUEmbeddingLayer::ForwardDecoder_V2(HUPtr<HUTensor> encoderOutput, std::vector<size_t> &embIdx, int beamSize)
{
    int dimBatch = encoderOutput->shape()[-3];
    int dimTrgEmb = encoderOutput->shape()[-1];
    int dimBeam = beamSize;

    HUPtr<HUTensor> selectedEmbs;
    if (embIdx.empty())   // first time
    {
        embIdx.resize(dimBatch, BOS_ID);
        selectedEmbs = HUTensorUtil::CopyRows(this->wordEmbedding_, embIdx, this->memPool_);
        selectedEmbs = HUTensorUtil::Reshape(selectedEmbs, {dimBatch, 1, dimTrgEmb});
    }
    else
    {
        selectedEmbs = HUTensorUtil::CopyRows(this->wordEmbedding_, embIdx, this->memPool_);
        selectedEmbs = HUTensorUtil::Reshape(selectedEmbs, { dimBatch*dimBeam, 1, dimTrgEmb });
    }
#ifdef DECODER_DEBUG
    LOG(trace, "[TenTrans][HUEmbeddingLayer][ForwardDecoder] selectedEmbs {}", selectedEmbs->debug());
#endif

    return selectedEmbs;
}

}
