
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
    this->useShortList_ = this->options_->get<bool>("use-shortlist");
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

	auto WMem = this->memPool_->alloc<TT_DATA_TYPE>(WShape->elements());
	auto bMem = this->memPool_->alloc<TT_DATA_TYPE>(bShape->elements());

	this->W_ = HUNew<HUTensor>(WMem, *WShape, device_);
	this->b_ = HUNew<HUTensor>(bMem, *bShape, device_);

	size_t size = 1; 
    for(size_t dim: WNp->shape) {
        size *= dim;
    }
    this->W_->set((TT_DATA_TYPE*)WNp->data(), (TT_DATA_TYPE*)WNp->data() + size);

	size = 1;
	for(size_t dim: bNp->shape) {
		size *= dim;
    }
	this->b_->set((TT_DATA_TYPE*)bNp->data(), (TT_DATA_TYPE*)bNp->data() + size);

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

HUPtr<HUTensor> HUOutputLayer::MultiplyW(HUPtr<HUTensor> input, const std::vector<size_t> &tgtEmbIdx, bool isFirstTime)
{
    bool transA = false, transB = false;
    if (this->isSharedOutEmbed_ || this->isSharedAllEmbed_) 
    {

        transB = true;
        if (this->useShortList_ && isFirstTime)   // copy by rows
        {
            this->new_W_ = HUTensorUtil::CopyRows(this->W_, tgtEmbIdx, this->memPool_);
            this->new_b_ = HUTensorUtil::CopyRows(this->b_, tgtEmbIdx, this->memPool_);
        }
    } 
    else 
    {
        if (this->useShortList_ && isFirstTime)   // copy by cols 
        {
            this->new_W_ = HUTensorUtil::CopyCols(this->W_, tgtEmbIdx, this->memPool_);
            this->new_b_ = HUTensorUtil::CopyCols(this->b_, tgtEmbIdx, this->memPool_);

            /*
            auto tmp_w = HUTensorUtil::Transpose(this->W_, {1, 0}, this->memPool_, this->device_); 
            auto tmp_w1 = HUTensorUtil::CopyRows(tmp_w, tgtEmbIdx, this->memPool_);
            this->new_W_ = HUTensorUtil::Transpose(tmp_w1, {1, 0}, this->memPool_, this->device_);
             
            auto tmp_b = HUTensorUtil::Transpose(this->b_, {1, 0}, this->memPool_, this->device_);           
            auto tmp_b1 = HUTensorUtil::CopyRows(tmp_b, tgtEmbIdx, this->memPool_);
            this->new_b_ = HUTensorUtil::Transpose(tmp_b1, {1, 0}, this->memPool_, this->device_);
            */

            // LOG(trace, "[TenTrans][OutputLayer] Forward W_, {}", this->W_->debug());
            // LOG(trace, "[TenTrans][OutputLayer] Forward new_W_, {}", this->new_W_->debug());

            // LOG(trace, "[TenTrans][OutputLayer] Forward b_, {}", this->b_->debug());
            // LOG(trace, "[TenTrans][OutputLayer] Forward new_b_, {}", this->new_b_->debug());
        }
    }

    auto tmp_W = this->useShortList_ ? this->new_W_ : this->W_;
    auto output = HUTensorUtil::Multiply(input, tmp_W, this->memPool_, this->device_, transA, transB);
    /*
    if (this->useShortList_) {
        auto output = HUTensorUtil::Multiply(input, this->new_W_, this->memPool_, this->device_, transA, transB);
    }
    else {
        auto output = HUTensorUtil::Multiply(input, this->W_, this->memPool_, this->device_, transA, transB);
    }
    */
#ifdef DEBUG_MOD
    LOG(trace, "[TenTrans][OutputLayer] Forward, {}", output->debug());
#endif

    return output;
}

}
