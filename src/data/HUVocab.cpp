#include "HUVocab.h"
#include "Logging.h"
#include "HUGlobal.h"
#include "HUUtil.h"
#include <boost/filesystem.hpp> 
#include <iostream>
#include <algorithm>
#include "yaml-cpp/yaml.h"

namespace TenTrans{

HUVocab::HUVocab()
{}

int HUVocab::Load(const std::string& vocabPath, int max) {
	LOG(info, "[TenTrans][data] Loading vocabulary from {}", vocabPath);
	
	ABORT_IF(!boost::filesystem::exists(vocabPath), "[TenTrans] Vocabulary {} does not exits", vocabPath);

	YAML::Node vocab = YAML::LoadFile(vocabPath);

	for(auto&& pair : vocab) {
    	auto str = pair.first.as<std::string>();
    	auto id = pair.second.as<size_t>();

    	if(!max || id < (size_t)max) {
      		str2id_[str] = id;
      		if(id >= id2str_.size())
        		id2str_.resize(id + 1);
      		id2str_[id] = str;
    	}
  	}
	
	ABORT_IF(id2str_.empty(), "[TenTrans] Empty vocabulary: ", vocabPath);
	
	id2str_[PAD_ID] = PAD_STR;
	id2str_[UNK_ID] = UNK_STR;
	id2str_[BOS_ID] = BOS_STR;
  	id2str_[EOS_ID] = EOS_STR;

	LOG(debug, "[TenTrans][vocab] id2str.size() {}", std::to_string(id2str_.size()));
	
	return std::max((int)id2str_.size(), max);
}

std::vector<size_t> HUVocab::Encode(const std::string& line, bool addBOS, bool addEOS) const
{
	std::vector<std::string> tokens;
	StringUtil::split(line, " ", tokens);
	return this->Encode(tokens, addBOS, addEOS);
}

std::vector<size_t> HUVocab::Encode(const std::vector<std::string>& tokens, bool addBOS, bool addEOS) const
{
	std::vector<size_t> words(tokens.size());
	std::transform(tokens.begin(), tokens.end(), words.begin(), [&](const std::string& w) { return (*this)[w]; });
	if (addBOS) {
		words.insert(words.begin(), BOS_ID);
	}
	if (addEOS) {
    	words.push_back(EOS_ID);
	}
  	return words;
}

size_t HUVocab::operator[](const std::string& word) const {
	auto it = str2id_.find(word);
  	if(it != str2id_.end())
    	return it->second;
  	else
    	return UNK_ID;
}

const std::string& HUVocab::operator[](size_t id) const {
  ABORT_IF(id >= id2str_.size(), "Unknown word id: ", id);
  return id2str_[id];
}

std::vector<std::string> HUVocab::Decode(const std::vector<int>& sentence, bool ignoreEOS) const
{
	std::vector<std::string> decoded;
	for(size_t i = 0; i < sentence.size(); ++i) {
		if((sentence[i] != EOS_ID || !ignoreEOS)) {
			decoded.push_back((*this)[(size_t)sentence[i]]);
		}
	}
	return decoded;
}

}
