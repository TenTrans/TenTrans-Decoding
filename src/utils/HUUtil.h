
#pragma once
#include<boost/algorithm/string.hpp>
#include <fstream>
#include <string>
#include <sstream>
#include<ctype.h>

#if defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
#else
#include <locale>
#include <cstdlib>
#endif

using namespace std;

#include <vector>

bool endswith(const std::string& str, const std::string& end);

void TrimLine(std::string & line);

void SplitUTF8Line(std::string & line, std::vector<std::string> & charVec);

int splitString(const std::string &srcStr,const std::string &splitStr,std::vector<std::string> &destVec);

string vec_to_str(vector<string> vec);

bool IsEnglishWord(string str);

class CodeConvertUtil
{
public:
	static wstring UTF82Unicode(string & line);
	static wstring UTF82Unicode(const char * line);
	static string Unicode2UTF8(wstring & line);
	static wstring GB2Unicode(string & line);
	static wstring GB2Unicode(const char * line);
	static string Unicode2GB(wstring & line);
};

class StringUtil
{
	public:
		static string LeftTrim(const string & str);
		static wstring LeftTrim(const wstring & str);
		static string RightTrim(const string & str);
		static wstring RightTrim(const wstring & str);
		static string Trim(string & str);
		static wstring Trim(wstring & str);

		static void split(string str, string delim, vector<string> & ret);
		static void split(wstring str, wstring delim, vector<wstring> & ret);
		static std::vector<std::string> split(const std::string input, const std::string chars);

		static int Wstring2Int(const wstring & str);
		static std::string Join(const std::vector<std::string>& words, const std::string& del = " ", bool reverse = false);

};
