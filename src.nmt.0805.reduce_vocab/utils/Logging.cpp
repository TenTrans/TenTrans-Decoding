#include <iostream>
#include "Logging.h"
#include "spdlog/sinks/null_sink.h"
#include "HUConfig.h"

namespace TenTrans
{

std::shared_ptr<spdlog::logger> stderrLogger(const std::string& name, const std::string& pattern, bool quiet) 
{

  std::vector<spdlog::sink_ptr> sinks;
  auto stderr_sink = spdlog::sinks::stderr_sink_mt::instance();

  if(!quiet) { 
      sinks.push_back(stderr_sink);
  }

  auto logger = std::make_shared<spdlog::logger>(name, begin(sinks), end(sinks));
  spdlog::register_logger(logger);
  logger->set_pattern(pattern);

  return logger;
}

void createLoggers(const HUConfig* options) 
{
    bool quiet = options && options->get<bool>("quiet");
    Logger general{ stderrLogger("TenTransLog", "[%Y-%m-%d %T] %v", quiet) };

    if(options && options->has("log-level")) 
    {
        std::string loglevel = options->get<std::string>("log-level");
        setLoggingLevel(*general, loglevel);
    }
}

bool setLoggingLevel(spdlog::logger& logger, std::string const level) 
{
  if(level == "trace") {
      logger.set_level(spdlog::level::trace);
  }
  else if(level == "debug") {
      logger.set_level(spdlog::level::debug);
  }
  else if(level == "info") {
      logger.set_level(spdlog::level::info);
  }
  else if(level == "warn") { 
      logger.set_level(spdlog::level::warn);
  }
  else if(level == "err" or level == "error") {
      logger.set_level(spdlog::level::err);
  }
  else if(level == "critical") {
      logger.set_level(spdlog::level::critical);
  }
  else if(level == "off") {
      logger.set_level(spdlog::level::off);
  }
  else {
      logger.warn("Unknown log level '{}' for logger '{}'", level.c_str(), logger.name().c_str());
      return false;
  }
  return true;
}
}
