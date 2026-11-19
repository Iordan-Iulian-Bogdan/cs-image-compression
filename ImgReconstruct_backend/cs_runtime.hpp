#ifndef CS_RUNTIME_HPP
#define CS_RUNTIME_HPP

// Small helpers shared by the encoder and the decoder: little-endian fields, side threads.

#include <cstdint>
#include <exception>
#include <thread>
#include <utility>

inline uint32_t cs_get_le32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
inline void cs_put_le32(uint8_t* p, uint32_t v) {
    for (int b = 0; b < 4; ++b) p[b] = (uint8_t)(v >> (8 * b));
}

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
