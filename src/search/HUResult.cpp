#include "HUResult.h"

namespace TenTrans{

HUResult::HUResult(HUPtr<HUConfig> options, HUPtr<HUVocab> vocab)
{
	this->vocab_ = vocab;
	this->reverse_ = options->get<bool>("right-left");
	this->nbest_ = options->get<bool>("n-best") ? options->get<int>("beam-size") : 0;
	this->maxId_ = -1;
}

void HUResult::GetTransResult(HUPtr<HUHistory> history, string& best1, string& bestn, int& token_num)
{
	const auto& nbl = history->NBest(nbest_);
	
	//@TODO nbest
	
	auto result = history->Top();
	auto words = std::get<0>(result);
    token_num = words.size();

	if(reverse_)
      std::reverse(words.begin(), words.end());
	
	auto wordsStr = vocab_->Decode(words);
	best1 = StringUtil::Join(wordsStr);
}

void HUResult::Add(long sourceId, const std::string& best1, const std::string& bestn)
{
	std::lock_guard<std::mutex> lock(mutex_);
	LOG(info, "[TenTrans] Best translation {} : {}", sourceId, best1);
	outputs_[sourceId] = std::make_pair(best1, bestn);
	if(maxId_ <= sourceId)
		maxId_ = sourceId;
}

std::vector<std::string> HUResult::Collect(bool nbest)
{
	std::vector<std::string> outputs;
	for(int id = 0; id <= maxId_; ++id)
		outputs.emplace_back(nbest ? outputs_[id].second : outputs_[id].first);
	return outputs;
}

}
