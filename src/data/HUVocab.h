#pragma once

#include <iostream>
#include <map>
#include <vector>
#include <algorithm>
using namespace std;

// This code draws on Vocab part of Marain project

namespace TenTrans{

class HUVocab{
public:
	HUVocab();
	int Load(const std::string& vocabPath, int max=0);
	size_t operator[](const std::string& word) const;
	const std::string& operator[](size_t id) const;
	std::vector<size_t> Encode(const std::string& line, bool addBOS=true, bool addEOS=true) const;
	std::vector<size_t> Encode(const std::vector<std::string>& tokens, bool addBOS=true, bool addEOS=true) const;
	std::vector<std::string> Decode(const std::vector<int>& sentence, bool ignoreEOS=true) const;

private:
	typedef std::map<std::string, size_t> Str2Id;
	Str2Id str2id_;

	typedef std::vector<std::string> Id2Str;
	Id2Str id2str_;

};
}
