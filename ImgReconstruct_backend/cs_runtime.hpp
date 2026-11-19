#ifndef CS_RUNTIME_HPP
#define CS_RUNTIME_HPP

// Small helpers shared by the encoder and the decoder: little-endian fields, the CS_PROFILE phase log, side threads.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <thread>
#include <utility>

inline uint32_t cs_get_le32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
inline void cs_put_le32(uint8_t* p, uint32_t v) {
    for (int b = 0; b < 4; ++b) p[b] = (uint8_t)(v >> (8 * b));
}

// CS_PROFILE=1 (anything but empty or "0").
inline bool cs_profile_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("CS_PROFILE");
        return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
    }();
    return on;
}

// Opt-in phase profiler: with CS_PROFILE=1 a scope prints one "profile[<tag>]: phase=ms ... total=ms" line on stderr
// when it ends, each phase the milliseconds since the previous mark. A scope without marks prints nothing. When off,
// the cost is two clock reads.
struct CsPhaseLog {
    const char* tag;
    const bool on = cs_profile_enabled();
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now(), last = start;
    std::string line;

    explicit CsPhaseLog(const char* t) : tag(t) {}
    CsPhaseLog(const CsPhaseLog&) = delete;
    CsPhaseLog& operator=(const CsPhaseLog&) = delete;

    void mark(const char* name) {
        if (!on) return;
        const auto now = std::chrono::steady_clock::now();
        char b[64];
        std::snprintf(b, sizeof(b), " %s=%.1f", name, std::chrono::duration<double, std::milli>(now - last).count());
        line += b;
        last = now;
    }

    ~CsPhaseLog() {
        if (on && !line.empty())
            std::fprintf(stderr, "profile[%s]:%s total=%.1f\n", tag, line.c_str(),
                std::chrono::duration<double, std::milli>(last - start).count());
    }
};

// A side thread joined when its scope ends (also on exceptions). What it runs must not throw.
struct CsJoinedThread {
    std::thread t;
    ~CsJoinedThread() { if (t.joinable()) t.join(); }
};

// A side thread whose work may throw: wait() joins it and rethrows what it threw. Joined when its scope ends too
// (the exception is then dropped: the scope is already unwinding). Not movable once started.
class CsSideTask {
    std::thread t;
    std::exception_ptr error;

public:
    CsSideTask() = default;
    CsSideTask(const CsSideTask&) = delete;
    CsSideTask& operator=(const CsSideTask&) = delete;
    ~CsSideTask() { if (t.joinable()) t.join(); }

    template <class F>
    void start(F&& f) {
        t = std::thread([this, f = std::forward<F>(f)]() mutable {
            try { f(); }
            catch (...) { error = std::current_exception(); }
        });
    }

    void wait() {
        if (t.joinable()) t.join();
        if (error) std::rethrow_exception(std::exchange(error, nullptr));
    }
};

#endif
