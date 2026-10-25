#include "rcd/progress.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <ostream>

namespace rcd {

AlgorithmTimer::AlgorithmTimer()
    : started_(std::chrono::steady_clock::now()) {}

void AlgorithmTimer::pause_for_output() {
    if (!paused_) {
        paused_at_ = std::chrono::steady_clock::now();
        paused_ = true;
    }
}

void AlgorithmTimer::resume_after_output() {
    if (paused_) {
        excluded_output_ += std::chrono::steady_clock::now() - paused_at_;
        paused_ = false;
    }
}

double AlgorithmTimer::elapsed_seconds() const {
    const auto end = std::chrono::steady_clock::now();
    const auto excluded = excluded_output_ +
        (paused_ ? end - paused_at_ : std::chrono::steady_clock::duration{});
    return std::chrono::duration<double>(end - started_ - excluded).count();
}

void print_progress_line(
    std::ostream& os,
    const std::string& stage,
    double completed_percent) {
    const auto now = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());
    std::tm local_time{};
#if defined(_WIN32)
    localtime_s(&local_time, &now);
#else
    localtime_r(&now, &local_time);
#endif

    const double percent = std::clamp(completed_percent, 0.0, 100.0);
    os << "[Progress] timestamp=" << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S")
       << " stage=" << stage
       << " completed=" << std::fixed << std::setprecision(2) << percent << "%\n"
       << std::defaultfloat << std::flush;
}

}  // namespace rcd
