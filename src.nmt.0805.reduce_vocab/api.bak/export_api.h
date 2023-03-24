#pragma once
#include <vector>
#include <iostream>
// using namespace std;

void * __init(std::string configPath);
std::vector<std::string> __translate(void * handle, std::vector<std::string> sources);
void   __destroy(void * handle);

