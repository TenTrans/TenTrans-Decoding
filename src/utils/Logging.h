
// note: This code draws on Logging part of Marain project
#pragma once
#include <iostream>
#include "spdlog/spdlog.h"

/*
 * Prints logging message into stderr and a file specified with `--log` option.
 *
 * Example usage: `LOG(info, "[data] Vocab size: {}", vocabSize)`
 *
 * A good practice is to put `[namespace]` at the beginning of the message.
 *
 * @param level Logging level: trace, debug, info, warn, error, critical
 * @param ... Message text and variables
 */

#define LOG(level, ...) TenTrans::checkedLog("TenTransLog", #level, __VA_ARGS__)

#ifdef __GNUC__
#define FUNCTION_NAME __PRETTY_FUNCTION__
#else
#ifdef _WIN32
#define FUNCTION_NAME __FUNCTION__
#else
#define FUNCTION_NAME "???"
#endif
#endif

/*
 * Prints critical error message and causes abnormal program termination by
 * calling std::abort().
 *
 * @param ... Message text and variables
 */

#define ABORT(...)                                                          \
  do {                                                                      \
	checkedLog("TenTransLog", "critical", __VA_ARGS__);                     \
    std::cerr << "Aborted from " << FUNCTION_NAME << " in " << __FILE__     \
              << ": " << __LINE__ << std::endl;                             \
    std::abort();                                                           \
  } while(0)

/*
 * Prints critical error message and causes abnormal program termination if
 * conditions is true.
 *
 * @param condition Condition expression
 * @param ... Message text and variables
 *
 * @see \def ABORT(...)
 */

#define ABORT_IF(condition, ...) \
  do {                           \
    if(condition) {              \
      ABORT(__VA_ARGS__);        \
    }                            \
  } while(0)


namespace TenTrans
{

class HUConfig;
void createLoggers(const HUConfig* options = nullptr);
typedef std::shared_ptr<spdlog::logger> Logger;
Logger stderrLogger(const std::string&, const std::string&,  bool quiet = false);
bool setLoggingLevel(spdlog::logger& logger, std::string const level);

template <class... Args>
void checkedLog(std::string logger, std::string level, Args... args) 
{
    Logger log = spdlog::get(logger);
    if(!log) 
    {
        if(level == "critical") 
        {
            auto errlog = stderrLogger("error", "Error: %v - aborting");
            errlog->critical(args...);
        }
        return ;
    }
    
    if(level == "trace") {
        log->trace(args...);
    }
    else if(level == "debug") { 
        log->debug(args...);
    }
    else if(level == "info") {
        log->info(args...);
    }
    else if(level == "warn") {
        log->warn(args...);
    }
    else if(level == "error") {
        log->error(args...);
    }
    else if(level == "critical") {
        log->critical(args...);
    }
    else {
        log->warn("Unknown log level '{}' for logger '{}'", level, logger);
    }
}
}

