
#pragma once
#include "HUTensor.h"
#include "HUGlobal.h"
#include "HUData.h"
#include <vector>

namespace TenTrans{
	
class HUEncoderState {
private:
  HUPtr<HUTensor> context_;
  HUPtr<HUTensor> mask_;
  HUPtr<HUBatch> batch_;

public:
  HUEncoderState(HUPtr<HUTensor> context, HUPtr<HUTensor> mask, HUPtr<HUBatch> batch)
      : context_(context), mask_(mask), batch_(batch) {}
  HUEncoderState() {}

  virtual HUPtr<HUTensor> getContext() { return context_; }
  virtual HUPtr<HUTensor> getMask() { return mask_; }
  virtual const std::vector<size_t>& getSourceWords() { return batch_->data(); }
  // virtual const std::vector<int>& getLengths() { return batch_->lengths(); }
};

}
