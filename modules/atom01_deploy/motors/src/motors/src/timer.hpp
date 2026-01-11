#pragma once

#include <chrono>
#include <thread>

class Timer {
   private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_t_;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_t_;

    std::chrono::milliseconds step_;

   public:
    Timer(int step) : step_(step) {}
    ~Timer() {}

    inline void ThreadSleepUntil() { std::this_thread::sleep_until(end_t_); }

    inline void UpdateNext() {
        start_t_ = std::chrono::high_resolution_clock::now();
        end_t_ = start_t_ + step_;
    }

    static inline void ThreadSleepFor(int num_steps) {
        std::this_thread::sleep_for(std::chrono::milliseconds(num_steps));
    }
    static inline void ThreadSleepForUs(int num_steps) {
        std::this_thread::sleep_for(std::chrono::microseconds(num_steps));
    }
};
