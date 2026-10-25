#pragma once

#include <iosfwd>
#include <string>
#include <chrono>
#include <functional>

namespace rcd {

using ProgressCallback = std::function<void(const std::string&, double)>;

// Emit a timestamped, flushed progress record suitable for nohup logs.
void print_progress_line(
    std::ostream& os,
    const std::string& stage,
    double completed_percent);

class AlgorithmTimer {
public:
    AlgorithmTimer();
    void pause_for_output();
    void resume_after_output();
    double elapsed_seconds() const;

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point started_;
    Clock::time_point paused_at_;
    std::chrono::duration<double> excluded_output_{};
    bool paused_ = false;
};

}  // namespace rcd
