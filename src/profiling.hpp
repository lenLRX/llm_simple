#pragma once

#include "acl_util.hpp"
#include <string>
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>


using json = nlohmann::json;

class AppProfiler {
public:
  bool StartLogging(const std::string &log_path);
  void RecordEvent(json jevent);
  void Finish();
private:
  std::ofstream profile_file;
  bool first_record;
  uint64_t start_us_count;
};

class AppProfileGuard {
public:
  AppProfileGuard(const char *name, const char *info, aclrtStream stream, AppProfiler* profiler, const char *fname, int lineno, bool is_profiling);
  ~AppProfileGuard();
  void AddBeginRecord();
  void AddEndRecord();

private:
  void AddRecord(const char *name, const char *info, aclrtStream stream, AppProfiler* profiler, const char *fname, int lineno) const;
  std::string record_name;
  std::string record_info;
  aclrtStream stream;
  AppProfiler* profiler;
  const char *record_file_name;
  int record_file_lineno;
  bool is_profiling;
  std::chrono::time_point<std::chrono::steady_clock, std::chrono::microseconds>
      start_us;
  aclrtEvent event;
};

#define _CONCAT_(x, y) x##y
#define __CONCAT__(x, y) _CONCAT_(x, y)

#define APP_PROFILE(name, info, stream, profiler, isprofiling)                           \
    AppProfileGuard __CONCAT__(temp_perf_obj_, __LINE__)(name, info, stream, profiler, __FILE__, __LINE__, isprofiling)
