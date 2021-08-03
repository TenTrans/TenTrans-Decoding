#include "HUConfig.h"
#include "cnpy.h"
#include "HUUtil.h"
//using namespace std;

namespace TenTrans{

//HUConfig::HUConfig(const HUConfig& other) : config_(YAML::Clone(other.config_)) {}

void HUConfig::Init(int argc, char** argv, HUConfigMode mode){

	//Step 1. Convert command line args to YAML Config
	auto parser = HUConfigParser(argc, argv, mode);
  	this->config_ = parser.getConfig();
	//this->devices_ = parser.getDevices();

  	createLoggers(this);

	this->devices_ = parser.getDevices();

  	// echo version and command line
  	std::string cmdLine;
  	for (int i = 0; i < argc; i++) {
    	std::string arg = argv[i];
    	std::string quote; // attempt to quote special chars
    	if (arg.empty() || arg.find_first_of(" #`\"'\\${}|&^?*!()%><") != std::string::npos)
      		quote = "'";
    	arg = boost::regex_replace(arg, boost::regex("'"), "'\\''");
    	if (!cmdLine.empty())
      		cmdLine.push_back(' ');
    	cmdLine += quote + arg + quote;
  	}

	//load model parameters
	if(mode == HUConfigMode::translating)
	{
		auto model = get<std::vector<std::string>>("models")[0];
		try {
			if(!get<bool>("ignore-model-config"))
				loadModelParameters(model);
    	} catch(std::runtime_error& ) {
			//LOG(info, "[config] No model configuration found in model file");
    	}
  	}

	Log();
}

void HUConfig::loadModelParameters(const std::string& name){
	YAML::Node config;
	GetYamlFromNpz(config, "special:model.yml", name);
	override(config);

}

void HUConfig::override(const YAML::Node& params) {
	for(auto& it : params) {
		//cout << it.first.as<std::string>() << "\t" << it.second << endl;
		config_[it.first.as<std::string>()] = it.second;
	}
}

void HUConfig::GetYamlFromNpz(YAML::Node& yaml, const std::string& varName, const std::string& fName) {
	yaml = YAML::Load(cnpy::npz_load(fName, varName)->data());
}

bool HUConfig::has(const std::string& key) const {
  if(config_[key])
	  return true;
  else 
	  return false;
}

void HUConfig::Log(){
	YAML::Emitter out;
  	OutputYaml(config_, out);
  	std::string configString = out.c_str();

  	// print YAML prepending each line with [config]
  	std::vector<std::string> results;
	StringUtil::split(configString, "\n", results);
  	for(auto& r : results)
    	LOG(info, "[TenTrans][config] {}", r);
}

}
