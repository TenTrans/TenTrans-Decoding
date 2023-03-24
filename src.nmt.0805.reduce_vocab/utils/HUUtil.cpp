#include "HUUtil.h"
#include <sstream>
#include<string>

bool endswith(const std::string& str, const std::string& end)
{
    int srclen = str.size();
    int endlen = end.size();
    if(srclen >= endlen)
    {   
        string temp = str.substr(srclen - endlen, endlen);
        if(temp == end)
            return true;
    }   

    return false;    
}

void TrimLine(std::string & line)
{
    line.erase(0,line.find_first_not_of(" \t\r\n"));
    line.erase(line.find_last_not_of(" \t\r\n")+1); 
}

void SplitUTF8Line(std::string & line, std::vector<std::string> & charVec)
{
    charVec.clear();
    TrimLine(line);
    for (size_t i = 0; i < line.length();)
    {
        string myChar = line.substr(i, 1);
        unsigned char myCh1 = (unsigned char)myChar.at(0);
        if (myCh1 < 128)
        {
            i ++;
        }
        else if (myCh1 < 224)
        {
            myChar = line.substr(i, 2);
            i += 2;
        }
        else if (myCh1 < 240)
        {
            myChar = line.substr(i, 3);
            i += 3;
        }
        else
        {
            myChar = line.substr(i, 4);                 
            i += 4;
        }
            charVec.push_back(myChar);
    }
}

int splitString(const std::string &srcStr,const std::string &splitStr,std::vector<std::string> &destVec) 
{   
    if(srcStr.size() == 0)
    {    
        return 0;
    }    
    size_t oldPos,newPos;
    oldPos = 0; 
    newPos = 0; 
    std::string tempData;
    while(1)
    {    
        newPos = srcStr.find(splitStr,oldPos);
        if(newPos != std::string::npos)
        {    
            tempData = srcStr.substr(oldPos,newPos-oldPos);
            destVec.push_back(tempData);
            oldPos = newPos + splitStr.size();
        }    
        else if(oldPos <= srcStr.size())
        {    
            tempData = srcStr.substr(oldPos);
            destVec.push_back(tempData);
            break;
        }                       
        else 
        {    
            break;
        }    
    }    
    return 0;
}

string vec_to_str(vector<string> vec)
{
    string raw_str = ""; 
    for(int i = 0;i < vec.size();i++){
        raw_str = raw_str + vec[i] + " ";
    }   
    raw_str = raw_str.substr(0,raw_str.length() - 1); 
    return raw_str;
}

bool IsEnglishWord(string str)
{
	int len = str.length();
	for(int i=0; i< len; i++)
	{
		if( !isalpha(str[i]))
			return false;
	}
	return true;
}

wstring CodeConvertUtil::UTF82Unicode(string & line)
	{
#if defined(_WIN32) || defined(_WIN64)
		size_t size = MultiByteToWideChar(CP_UTF8, 0, line.c_str(), -1, NULL, 0);
		wchar_t * wcstr = new wchar_t[size];
		if (!wcstr)
			return L"";
		MultiByteToWideChar(CP_UTF8, 0, line.c_str(), -1, wcstr, size);
#else
		setlocale(LC_ALL, "zh_CN.UTF-8");
		size_t size = mbstowcs(NULL, line.c_str(), 0);
		wchar_t * wcstr = new wchar_t[size + 1];
		if (!wcstr)
			return L"";
		mbstowcs(wcstr, line.c_str(), size + 1);
#endif
		wstring retrunStr(wcstr);
		delete[] wcstr;

		return retrunStr;
	}

	wstring CodeConvertUtil::UTF82Unicode(const char * line)
	{
#if defined(_WIN32) || defined(_WIN64)
		size_t size = MultiByteToWideChar(CP_UTF8, 0, line, -1, NULL, 0);
		wchar_t * wcstr = new wchar_t[size];
		if (!wcstr)
			return L"";
		MultiByteToWideChar(CP_UTF8, 0, line, -1, wcstr, size);
#else
		setlocale(LC_ALL, "zh_CN.UTF-8");
		size_t size = mbstowcs(NULL, line, 0);
		wchar_t * wcstr = new wchar_t[size + 1];
		if (!wcstr)
			return L"";
		mbstowcs(wcstr, line, size + 1);
#endif
		wstring retrunStr(wcstr);
		delete[] wcstr;

		return retrunStr;
	}

	string CodeConvertUtil::Unicode2UTF8(wstring & line)
	{
#if defined(_WIN32) || defined(_WIN64)
		size_t size = WideCharToMultiByte(CP_UTF8, 0, line.c_str(), -1, NULL, 0, NULL, NULL);
		char * mbstr = new char[size];
		if (!mbstr)
			return "";
		WideCharToMultiByte(CP_UTF8, 0, line.c_str(), -1, mbstr, size, NULL, NULL);
#else
		setlocale(LC_ALL, "zh_CN.UTF-8");
		size_t size = wcstombs(NULL, line.c_str(), 0);
		char * mbstr = new char[size + 1];
		if (!mbstr)
			return "";
		wcstombs(mbstr, line.c_str(), size + 1);
#endif
		string returnStr(mbstr);
		delete[] mbstr;

		return returnStr;
	}

	wstring CodeConvertUtil::GB2Unicode(string & line)
	{
#if defined(_WIN32) || defined(_WIN64)
		size_t size = MultiByteToWideChar(CP_ACP, 0, line.c_str(), -1, NULL, 0);
		wchar_t* wcstr = new wchar_t[size];
		if (!wcstr)
			return L"";
		MultiByteToWideChar(CP_ACP, 0, line.c_str(), -1, wcstr, size);
#else
		setlocale(LC_ALL, "zh_CN.GB2312");
		size_t size = mbstowcs(NULL, line.c_str(), 0);
		wchar_t* wcstr = new wchar_t[size + 1];
		if (!wcstr)
			return L"";
		mbstowcs(wcstr, line.c_str(), size + 1);
#endif
		wstring returnStr(wcstr);
		delete[] wcstr;

		return returnStr;
	}

	wstring CodeConvertUtil::GB2Unicode(const char * line)
	{
#if defined(_WIN32) || defined(_WIN64)
		size_t size = MultiByteToWideChar(CP_ACP, 0, line, -1, NULL, 0);
		wchar_t* wcstr = new wchar_t[size];
		if (!wcstr)
			return L"";
		MultiByteToWideChar(CP_ACP, 0, line, -1, wcstr, size);
#else
		setlocale(LC_ALL, "zh_CN.GB2312");
		size_t size = mbstowcs(NULL, line, 0);
		wchar_t* wcstr = new wchar_t[size + 1];
		if (!wcstr)
			return L"";
		mbstowcs(wcstr, line, size + 1);
#endif
		wstring returnStr(wcstr);
		delete[] wcstr;

		return returnStr;
	}

	string CodeConvertUtil::Unicode2GB(wstring & line)
	{
#if defined(_WIN32) || defined(_WIN64)
		size_t size = WideCharToMultiByte(CP_ACP, 0, line.c_str(), -1, NULL, 0, NULL, NULL);
		char* mbstr = new char[size];
		if (!mbstr)
			return "";
		WideCharToMultiByte(CP_ACP, 0, line.c_str(), -1, mbstr, size, NULL, NULL);
#else
		setlocale(LC_ALL, "zh_CN.GB2312");
		size_t size = wcstombs(NULL, line.c_str(), 0);
		char* mbstr = new char[size + 1];
		if (!mbstr)
			return "";
		wcstombs(mbstr, line.c_str(), size + 1);
#endif
		string returnStr(mbstr);
		delete[] mbstr;

		return returnStr;
	}

string StringUtil::LeftTrim(const string & str)
	{
		if (str.find_first_not_of(" \n\r\t") == string::npos)
			return str;
		return str.substr(str.find_first_not_of(" \n\r\t"));
	}

	wstring StringUtil::LeftTrim(const wstring & str)
	{
		if (str.find_first_not_of(L" \n\r\t　") == wstring::npos)
			return str;
		return str.substr(str.find_first_not_of(L" \n\r\t　"));
	}

	string StringUtil::RightTrim(const string & str)
	{
		if (str.find_last_not_of(" \n\r\t") == string::npos)
			return str;
		return str.substr(0, str.find_last_not_of(" \n\r\t") + 1);
	}

	wstring StringUtil::RightTrim(const wstring & str)
	{
		if (str.find_last_not_of(L" \n\r\t　") == wstring::npos)
			return str;
		return str.substr(0, str.find_last_not_of(L" \n\r\t　") + 1);
	}

	string StringUtil::Trim(string & str)
	{
		str = LeftTrim(RightTrim(str));
		return str;
	}

	wstring StringUtil::Trim(wstring & str)
	{
		str = StringUtil::LeftTrim(StringUtil::RightTrim(str));
		return str;
	}

	void StringUtil::split(string str, string delim, vector<string> & ret)
	{
		size_t last = 0;
		size_t index = str.find_first_of(delim, last);
		while (index != std::string::npos)
		{
			ret.push_back(str.substr(last, index - last));
			last = index + 1;
			index = str.find_first_of(delim, last);
		}
		if (index - last>0)
		{
			ret.push_back(str.substr(last, index - last));
		}
	}

    void StringUtil::split_v2(const std::string & input, const std::string & sep, std::vector<std::string> & vec)
    {
        std::string str = input;
        std::string substring;
        std::string::size_type start = 0;
        std::string::size_type index = 0;
        std::string::size_type separator_len = sep.size();
        while (index != std::string::npos && start < input.size())
        {
            index = input.find(sep, start);
            if (index == 0)
            {
                if (start == 0) {
                    vec.push_back("");
                }
                start = start + separator_len;
                continue;
            }
            if (index == std::string::npos)
            {
                vec.push_back(input.substr(start));
                break;
            }
            vec.push_back(input.substr(start, index - start));
            start = index + separator_len;
        }
    }

	void StringUtil::split(wstring str, wstring delim, vector<wstring> & ret)
	{
		size_t last = 0;
		size_t index = str.find_first_of(delim, last);
		while (index != std::wstring::npos)
		{
			ret.push_back(str.substr(last, index - last));
			last = index + 1;
			index = str.find_first_of(delim, last);
		}
		if (index - last>0)
		{
			ret.push_back(str.substr(last, index - last));
		}
	}

	std::vector<std::string> StringUtil::split(const std::string input, const std::string chars)
	{
		std::vector<std::string> output;
		boost::split(output, input, boost::is_any_of(chars));
		return output;
	}

	int StringUtil::Wstring2Int(const wstring & str)
	{
		int base = 0;
		wstring::size_type index = 0;

		bool isNegative = str[index] == L'-' ? ++index, true : false;

		for (; index < str.length(); index++)
		{
			if (str[index] >= L'0' && str[index] <= L'9')
				base = base * 10 + str[index] - L'0';
			else
				break;
		}

		if (isNegative)
			base *= -1;

		return base;
	}

/*
	string StringUtil::Join(std::vector<std::string> strList)
	{
		std::stringstream ss;
		for (size_t i = 0; i < strList.size(); i++)
		{
			if (i != 0)
				ss << " ";
			ss << strList[i];
		}
		string result = StringUtil::Trim(ss.str());
		return result;
	}
*/

std::string StringUtil::Join(const std::vector<std::string>& words, const std::string& del /*= " "*/, bool reverse /*= false*/) {
  std::stringstream ss;
  if(words.empty()) {
    return "";
  }

  if(reverse) {
    for(size_t i = words.size() - 1; i > 0; --i) {
      ss << words[i] << del;
    }
    ss << words[0];
  } else {
    ss << words[0];
    for(size_t i = 1; i < words.size(); ++i) {
      ss << del << words[i];
    }
  }
  return ss.str();
}
