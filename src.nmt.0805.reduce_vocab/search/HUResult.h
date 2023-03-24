#pragma once
#include<iostream>
#include "HUVocab.h"
#include "HUHistory.h"
#include <mutex>
#include "HUConfig.h"
#include "HUUtil.h"
#include <map>
namespace TenTrans{

class HUResult{
public:
	HUResult(HUPtr<HUConfig> options, HUPtr<HUVocab> vocab);
	void GetTransResult(HUPtr<HUHistory> history, const std::vector<size_t>& newVocab, string& best1, string& bestn, int& token_num);
	void Add(long sourceId, const std::string& best1, const std::string& bestn);
	std::vector<std::string> Collect(bool nbest=false);

private:
	long maxId_;
	std::mutex mutex_;
	HUPtr<HUVocab> vocab_;

	bool reverse_{false};
	size_t nbest_{0};
	typedef std::map<long, std::pair<std::string, std::string>> Outputs;
	Outputs outputs_;
};

}
