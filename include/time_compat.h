// Portability shim: provide clock_gettime(CLOCK_MONOTONIC, ...) on platforms
// (e.g. MinGW/Windows) where it is not available. No-op on POSIX systems that
// already provide it.
#ifndef VIBEASR_TIME_COMPAT_H
#define VIBEASR_TIME_COMPAT_H

#include <time.h>

#if defined(_WIN32) && !defined(CLOCK_MONOTONIC)

#include <chrono>

#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif

static inline int vibeasr_clock_gettime(int /*clk_id*/, struct timespec *ts) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    const auto secs = std::chrono::duration_cast<std::chrono::seconds>(now);
    const auto nsecs = std::chrono::duration_cast<std::chrono::nanoseconds>(now - secs);
    ts->tv_sec = static_cast<time_t>(secs.count());
    ts->tv_nsec = static_cast<long>(nsecs.count());
    return 0;
}

#define clock_gettime(clk, ts) vibeasr_clock_gettime((clk), (ts))

#endif // _WIN32 && !CLOCK_MONOTONIC

#endif // VIBEASR_TIME_COMPAT_H
