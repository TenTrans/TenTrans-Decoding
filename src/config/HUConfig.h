#pragma once
#include "yaml-cpp/yaml.h"
#include <iostream>
#include <boost/regex.hpp>
#include <vector>
#include "HUGlobal.h"
#include "HUConfigParser.h"
#include "Logging.h"

//using namespace std;

namespace TenTrans{


// This code draws on Config/ConfigParser part of Marain project
class HUConfig{

public:
  //static size_t seed;

	typedef YAML::Node YamlNode;

  /*HUConfig(int argc,
         char** argv,
         string mode,
         bool validate = true);*/

	HUConfig(){}
  //HUConfig(const HUConfig& other);
  //Config(const Options& options);

	void Init(int argc, char** argv, HUConfigMode mode);

	bool has(const std::string& key) const;

  //YAML::Node operator[](const std::string& key) const;
  //YAML::Node get(const std::string& key) const;

	template <typename T>
	T get(const std::string& key) const {
		return config_[key].as<T>();
	}

	void override(const YAML::Node& params);
	void Log();

  /*template <typename T>
  T get(const std::string& key, const T& dflt) const {
    if(has(key))
      return config_[key].as<T>();
    else
      return dflt;
  }

  const YAML::Node& get() const;
  YAML::Node& get();

  template <typename T>
  void set(const std::string& key, const T& value) {
    config_[key] = value;
  }*/

  //YAML::Node getModelParameters();
	void loadModelParameters(const std::string& name);
	void GetYamlFromNpz(YAML::Node& yaml, const std::string& varName, const std::string& fName);
	const std::vector<DeviceId>& getDevices() {return devices_; }

	
  //void save(const std::string& name);

  /*friend std::ostream& operator<<(std::ostream& out, const Config& config) {
    YAML::Emitter outYaml;
    cli::OutputYaml(config.get(), outYaml);
    out << outYaml.c_str();
    return out;
  }*/


private:
	YAML::Node config_;
	std::vector<DeviceId> devices_;

  // Add options overwritting values for existing ones
  //void override(const YAML::Node& params)
	
};
}
