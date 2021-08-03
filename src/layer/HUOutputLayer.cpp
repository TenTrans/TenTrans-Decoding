/*
 * Author: Danielkxwu
 * E-mial: danielkxwu@tencent.com
 * Created Date: 2021/4/9
 *
 */

#include "HUOutputLayer.h"
#include "HUTensorUtil.h"

namespace TenTrans{

HUOutputLayer::HUOutputLayer(HUPtr<HUConfig> options, HUPtr<HUMemPool> memoryPool, HUPtr<HUDevice> device, cnpy::npz_t modelNpz, bool isEncoder): 
    HUBaseLayer(options, memoryPool, device, modelNpz, isEncoder)
{
#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][HUOutputLayer] Loading OutputLayer ...\n");
#endif
    if (isEncoder) {
		this->prefix_ = "classification_output_layer.";
    }
    else {
		this->prefix_ = "decoder.output_layer.";
    }

    this->isSharedAllEmbed_ = this->options_->get<bool>("share-all-embed");
    this->isSharedOutEmbed_ = this->options_->get<bool>("share-out-embed");
}

HUOutputLayer::~HUOutputLayer()
{
    this->memPool_->free(this->W_->memory());
    this->memPool_->free(this->b_->memory());
}

void HUOutputLayer::Init()
{
	string WParam = this->prefix_ + "weight";

    if (this->isSharedOutEmbed_) {
        WParam = "decoder.embedding.weight"; 
    }

    if (this->isSharedAllEmbed_) {
        WParam = "encoder.embedding.weight";
    }

	string bParam = this->prefix_ + "bias";
	auto WNp = modelNpz_[WParam];
	auto bNp = modelNpz_[bParam];

	auto WShape = GetShapeByModel(WParam, this->modelNpz_);
	auto bShape = GetShapeByModel(bParam, this->modelNpz_);

	auto WMem = this->memPool_->alloc<float>(WShape->elements());
	auto bMem = this->memPool_->alloc<float>(bShape->elements());

	this->W_ = HUNew<HUTensor>(WMem, *WShape, device_);
	this->b_ = HUNew<HUTensor>(bMem, *bShape, device_);

	size_t size = 1; 
    for(size_t dim: WNp->shape) {
        size *= dim;
    }
    this->W_->set((float*)WNp->data(), (float*)WNp->data() + size);

	size = 1;
	for(size_t dim: bNp->shape) {
		size *= dim;
    }
	this->b_->set((float*)bNp->data(), (float*)bNp->data() + size);

#ifdef DEBUG_MOD
	LOG(trace, "[TenTrans][OutputLayer] Loading {} parameters, {}", WParam, this->W_->debug());
	LOG(trace, "[TenTrans][OutputLayer] Loading {} parameters, {}", bParam, this->b_->debug());
#endif
}

HUPtr<HUTensor> HUOutputLayer::Forward(HUPtr<HUTensor> input)
{
    bool transA = false, transB = false;
    if (this->isSharedOutEmbed_ || this->isSharedAllEmbed_) {
        transB = true;
    }
	auto output = HUTensorUtil::Affine(input, this->W_, this->b_, this->memPool_, this->device_, transA, transB);
#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][OutputLayer] Forward, {}", output->debug());
#endif

	return output;
}
HUPtr<HUTensor> HUOutputLayer::MultiplyW(HUPtr<HUTensor> input)
{
    bool transA = false, transB = false;
    if (this->isSharedOutEmbed_ || this->isSharedAllEmbed_) {
        transB = true;
    }
    auto output = HUTensorUtil::Multiply(input, this->W_, this->memPool_, this->device_, transA, transB);
#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][OutputLayer] Forward, {}", output->debug());
#endif

    return output;
}

}
