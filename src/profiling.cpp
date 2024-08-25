#include <atomic>
#include <chrono>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>

#include "profiling.hpp"



bool AppProfiler::StartLogging(const std::string &log_path) {
    profile_file.open(log_path.c_str());
    if (!profile_file) {
        return false;
    }
    profile_file << "[\n";
    first_record = true;
    auto start_tp = std::chrono::steady_clock::now();
    auto start_us_tp = std::chrono::time_point_cast<std::chrono::microseconds>(start_tp);
    start_us_count = start_us_tp.time_since_epoch().count();
    return true;
}

void AppProfiler::RecordEvent(json jevent) {
    if (first_record) {
        first_record = false;
    } else {
        profile_file << ",\n";
    }
    jevent["ts"] = jevent["ts"].get<uint64_t>() - start_us_count;
    profile_file << jevent;
}

void AppProfiler::Finish() {
  profile_file << "]\n";
  profile_file.close();
}

AppProfileGuard::AppProfileGuard(const char *name,
                                 const char *info,
                                 aclrtStream stream,
                                 AppProfiler* profiler,
                                 const char *fname, int lineno,
                                 bool is_profiling)
    : record_name(name), record_info(info), record_file_name(fname), record_file_lineno(lineno),
      stream(stream), profiler(profiler), is_profiling(is_profiling) {
  if (is_profiling) {
    CHECK_ACL(aclrtCreateEvent(&event));
    CHECK_ACL(aclrtRecordEvent(event, stream));
    AddBeginRecord();
  }
}

AppProfileGuard::~AppProfileGuard() {
  if (is_profiling) {
    AddEndRecord();
  }
}

void AppProfileGuard::AddBeginRecord() {
  auto current_tp = std::chrono::steady_clock::now();
  start_us =
      std::chrono::time_point_cast<std::chrono::microseconds>(current_tp);
}

void AppProfileGuard::AddEndRecord() {
  AddRecord(record_name.c_str(), record_info.c_str(), stream, profiler, record_file_name, record_file_lineno);
}

void AppProfileGuard::AddRecord(const char *name, const char *info, aclrtStream stream, AppProfiler* profiler, const char *fname, int lineno) const {
  aclrtEvent end_event;
  CHECK_ACL(aclrtCreateEvent(&end_event));
  CHECK_ACL(aclrtRecordEvent(end_event, stream));
  CHECK_ACL(aclrtSynchronizeStream(stream));
  float duration_ms;
  CHECK_ACL(aclrtEventElapsedTime(&duration_ms, event, end_event));
  CHECK_ACL(aclrtDestroyEvent(event));
  CHECK_ACL(aclrtDestroyEvent(end_event));

  auto current_tp = std::chrono::steady_clock::now();
  auto us_tp =
      std::chrono::time_point_cast<std::chrono::microseconds>(current_tp);
  auto start_count = start_us.time_since_epoch().count();
  auto end_count = us_tp.time_since_epoch().count();

  json record{{"name", name},
              {"ph", "X"},
              {"pid", 0},
              {"ts", start_count},
              {"dur", duration_ms * 1000},
              {"args", {{"file", fname}, {"lineno", lineno}, {"info", info}}}};

  profiler->RecordEvent(record);
}