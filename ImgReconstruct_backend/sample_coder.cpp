#include "sample_coder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace {
constexpr int NPASS = 11;
const int PASS_T[NPASS] = { 1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64 };   // pass p holds Bayer ranks [T[p-1], T[p])
constexpr int KNB = 6;          // neighbours per prediction
constexpr int MAXOFF = 128;     // candidate offsets per (pass, phase)
constexpr int NACT = 12, NSUP = 6;
constexpr int CTX_P0 = NACT * NSUP, CTX_NONE = CTX_P0 + 1, NCTX = CTX_NONE + 1;
constexpr int PROB_BITS = 12, PROB = 1 << PROB_BITS;
constexpr uint32_t RANS_L = 1u << 23;
const int ACT_T[NACT - 1] = { 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128 };   // activity bins, 1/16 levels

struct Off { int8_t dy, dx; uint16_t d2; };
struct Tables {
    uint8_t bayer[64];
    uint8_t pass_of[64];
    std::vector<Off> off[NPASS][64];   // per pass (>= 1) and position phase: earlier-pass lattice offsets, nearest first
    Tables() {
        for (int y = 0; y < 8; ++y)
            for (int x = 0; x < 8; ++x) {
                int r = 0;
                for (int b = 0; b < 3; ++b)
                    r |= ((((x ^ y) >> b) & 1) << (2 * (2 - b) + 1)) | (((y >> b) & 1) << (2 * (2 - b)));
                bayer[y * 8 + x] = (uint8_t)r;
            }
        for (int i = 0; i < 64; ++i) {
            int p = 0;
            while (bayer[i] >= PASS_T[p]) ++p;
            pass_of[i] = (uint8_t)p;
        }
        for (int p = 1; p < NPASS; ++p) {
            const double f = PASS_T[p - 1] / 64.0;
            const int R = (std::min)(51, (int)std::ceil(std::sqrt(MAXOFF / (3.14159265 * f))) + 1);
            for (int ph = 0; ph < 64; ++ph) {
                std::vector<Off>& o = off[p][ph];
                for (int dy = -R; dy <= R; ++dy)
                    for (int dx = -R; dx <= R; ++dx) {
                        if (!dy && !dx) continue;
                        if (bayer[(((ph >> 3) + dy) & 7) * 8 + (((ph & 7) + dx) & 7)] >= PASS_T[p - 1]) continue;
                        o.push_back({ (int8_t)dy, (int8_t)dx, (uint16_t)(dy * dy + dx * dx) });
                    }
                std::sort(o.begin(), o.end(), [](const Off& a, const Off& b) {
                    return a.d2 != b.d2 ? a.d2 < b.d2 : a.dy != b.dy ? a.dy < b.dy : a.dx < b.dx; });
                if (o.size() > MAXOFF) o.resize(MAXOFF);
            }
        }
    }
};
const Tables& tables() { static const Tables t; return t; }

// Prediction and context of the sample at (y, x) of pass p (>= 1) from the earlier passes' samples in val.
inline void predict(const Tables& T, const uint8_t* val, const uint8_t* mask, int rows, int cols, int p,
    int y, int x, int amax, uint8_t& pred, uint8_t& ctx) {
    int v[KNB]; uint32_t w[KNB];
    int n = 0, d2min = 0;
    uint64_t sw = 0, swv = 0;
    for (const Off& o : T.off[p][(y & 7) * 8 + (x & 7)]) {
        const int yy = y + o.dy, xx = x + o.dx;
        if ((unsigned)yy >= (unsigned)rows || (unsigned)xx >= (unsigned)cols) continue;
        const size_t j = (size_t)yy * cols + xx;
        if (!mask[j]) continue;
        if (n == 0) d2min = o.d2;
        else if (n >= 3 && o.d2 > 9 * d2min) break;   // far samples add nothing once near ones exist
        v[n] = val[j]; w[n] = (1u << 20) / o.d2;
        sw += w[n]; swv += (uint64_t)w[n] * v[n];
        if (++n == KNB) break;
    }
    if (n == 0) { pred = (uint8_t)((amax + 1) >> 1); ctx = CTX_NONE; return; }
    const int p16 = (int)((16 * swv + sw / 2) / sw);
    uint64_t sa = 0;
    for (int i = 0; i < n; ++i) sa += (uint64_t)w[i] * (uint32_t)std::abs(16 * v[i] - p16);
    const int act = (int)(sa / sw);
    int ab = 0;
    while (ab < NACT - 1 && act >= ACT_T[ab]) ++ab;
    const int sup = n < 3 ? 5 : d2min <= 1 ? 0 : d2min <= 2 ? 1 : d2min <= 5 ? 2 : d2min <= 16 ? 3 : 4;
    pred = (uint8_t)(std::min)(amax, (p16 + 8) >> 4);
    ctx = (uint8_t)(sup * NACT + ab);
}

// Sample positions (row-major) per pass.
void pass_lists(const Tables& T, const uint8_t* mask, int rows, int cols, std::vector<uint32_t> pos[NPASS]) {
    for (int y = 0; y < rows; ++y) {
        const uint8_t* m = mask + (size_t)y * cols;
        const uint8_t* po = T.pass_of + (y & 7) * 8;
        for (int x = 0; x < cols; ++x)
            if (m[x]) pos[po[x & 7]].push_back((uint32_t)((size_t)y * cols + x));
    }
}

inline int to_sym(int v, int pred, int A) {      // wrapped residual, zigzag
    const int r = ((v - pred + A / 2) & (A - 1)) - A / 2;
    return r >= 0 ? 2 * r : -2 * r - 1;
}
inline int from_sym(int z, int pred, int A) {
    const int r = (z & 1) ? -((z + 1) >> 1) : (z >> 1);
    return (pred + r) & (A - 1);
}

// counts -> frequencies summing to PROB, every used symbol >= 1.
void normalize(const uint32_t* cnt, int n, uint16_t* fr) {
    uint64_t tot = 0;
    for (int i = 0; i < n; ++i) tot += cnt[i];
    int sum = 0, big = 0;
    for (int i = 0; i < n; ++i) {
        fr[i] = cnt[i] ? (uint16_t)(std::max<uint64_t>)(1, (uint64_t)cnt[i] * PROB / tot) : 0;
        sum += fr[i];
        if (fr[i] > fr[big]) big = i;
    }
    // The rounding error goes to the largest; if that would empty it (many rare symbols), shave the others.
    int diff = PROB - sum;
    while (diff < 0 && -diff >= fr[big]) {
        for (int i = 0; i < n && diff < 0; ++i)
            if (i != big && fr[i] > 1) { --fr[i]; ++diff; }
    }
    fr[big] = (uint16_t)(fr[big] + diff);
}
}

std::vector<uint8_t> cs_plane_encode(const uint8_t* val, const uint8_t* mask, int rows, int cols, int bits) {
    const Tables& T = tables();
    const int A = 1 << bits;
    std::vector<uint32_t> pos[NPASS];
    pass_lists(T, mask, rows, cols, pos);
    size_t n = 0;
    for (int p = 0; p < NPASS; ++p) n += pos[p].size();
    std::vector<uint8_t> out;
    if (n == 0) return out;
    out.push_back(0);   // stream type: 0 = coded, 1 = bit-packed (below)
    std::vector<uint8_t> ctx(n), sym(n);
    size_t base = 0;
    int prev = A / 2;
    for (size_t i = 0; i < pos[0].size(); ++i) {   // pass 0: DPCM along the scan
        const int v = val[pos[0][i]];
        ctx[i] = CTX_P0; sym[i] = (uint8_t)to_sym(v, prev, A);
        prev = v;
    }
    base = pos[0].size();
    for (int p = 1; p < NPASS; ++p) {
        const std::vector<uint32_t>& ps = pos[p];
        const long long cnt = (long long)ps.size();
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < cnt; ++i) {
            const int y = (int)(ps[i] / (uint32_t)cols), x = (int)(ps[i] % (uint32_t)cols);
            uint8_t pr, c;
            predict(T, val, mask, rows, cols, p, y, x, A - 1, pr, c);
            ctx[base + i] = c; sym[base + i] = (uint8_t)to_sym(val[ps[i]], pr, A);
        }
        base += ps.size();
    }
    // Tables: per context [u16 n = highest used symbol + 1 (0 = unused)] then n x 12-bit frequencies (mod 4096).
    std::vector<uint32_t> cnt((size_t)NCTX * 256, 0);
    for (size_t i = 0; i < n; ++i) cnt[(size_t)ctx[i] * 256 + sym[i]]++;
    std::vector<uint16_t> fr((size_t)NCTX * 256, 0), st((size_t)NCTX * 256, 0);
    for (int c = 0; c < NCTX; ++c) {
        int nu = 0;
        for (int s = 0; s < A; ++s) if (cnt[(size_t)c * 256 + s]) nu = s + 1;
        out.push_back((uint8_t)nu); out.push_back((uint8_t)(nu >> 8));
        if (!nu) continue;
        uint16_t* f = &fr[(size_t)c * 256];
        normalize(&cnt[(size_t)c * 256], nu, f);
        uint32_t acc = 0; int nb = 0, s0 = 0;
        for (int s = 0; s < nu; ++s) {
            st[(size_t)c * 256 + s] = (uint16_t)s0; s0 += f[s];
            acc |= (uint32_t)(f[s] & (PROB - 1)) << nb; nb += PROB_BITS;
            while (nb >= 8) { out.push_back((uint8_t)acc); acc >>= 8; nb -= 8; }
        }
        if (nb) out.push_back((uint8_t)acc);
    }
    // rANS, symbols in reverse; the bytes come out back to front.
    std::vector<uint8_t> rb;
    rb.reserve(n / 2 + 16);
    uint32_t x = RANS_L;
    for (size_t i = n; i-- > 0;) {
        const uint32_t f = fr[(size_t)ctx[i] * 256 + sym[i]], s = st[(size_t)ctx[i] * 256 + sym[i]];
        const uint32_t xmax = ((RANS_L >> PROB_BITS) << 8) * f;
        while (x >= xmax) { rb.push_back((uint8_t)x); x >>= 8; }
        x = ((x / f) << PROB_BITS) + (x % f) + s;
    }
    for (int b = 0; b < 4; ++b) out.push_back((uint8_t)(x >> (24 - 8 * b)));
    out.insert(out.end(), rb.rbegin(), rb.rend());
    // A plane the model does not fit (few samples: the tables dominate; noise) ships bit-packed in scan order,
    // so a stream is never more than one byte above the packed size.
    const size_t packed = (n * (size_t)bits + 7) / 8;
    if (out.size() > 1 + packed) {
        out.assign(1 + packed, 0);
        out[0] = 1;
        size_t bit = 8;
        for (size_t j = 0; j < (size_t)rows * cols; ++j) {
            if (!mask[j]) continue;
            for (int b = bits - 1; b >= 0; --b, ++bit) out[bit >> 3] |= (uint8_t)(((val[j] >> b) & 1) << (7 - (bit & 7)));
        }
    }
    return out;
}

bool cs_plane_decode(const uint8_t* data, size_t len, const uint8_t* mask, int rows, int cols, int bits, uint8_t* val) {
    const Tables& T = tables();
    const int A = 1 << bits;
    std::vector<uint32_t> pos[NPASS];
    pass_lists(T, mask, rows, cols, pos);
    size_t n = 0, maxp = 0;
    for (int p = 0; p < NPASS; ++p) { n += pos[p].size(); maxp = (std::max)(maxp, pos[p].size()); }
    if (n == 0) return true;
    if (len < 1 || data[0] > 1) return false;
    if (data[0] == 1) {   // bit-packed in scan order
        if (len != 1 + (n * (size_t)bits + 7) / 8) return false;
        size_t bit = 8;
        for (size_t j = 0; j < (size_t)rows * cols; ++j) {
            if (!mask[j]) continue;
            int v = 0;
            for (int b = 0; b < bits; ++b, ++bit) v = (v << 1) | ((data[bit >> 3] >> (7 - (bit & 7))) & 1);
            val[j] = (uint8_t)v;
        }
        return true;
    }
    const uint8_t* d = data + 1; const uint8_t* end = data + len;
    std::vector<uint16_t> fr((size_t)NCTX * 256, 0), st((size_t)NCTX * 256, 0);
    std::vector<uint8_t> slot((size_t)NCTX * PROB, 0);
    for (int c = 0; c < NCTX; ++c) {
        if (end - d < 2) return false;
        const int nu = d[0] | (d[1] << 8); d += 2;
        if (!nu) continue;
        if (nu > A || (size_t)(end - d) < ((size_t)nu * PROB_BITS + 7) / 8) return false;
        uint32_t acc = 0; int nb = 0, s0 = 0;
        uint16_t* f = &fr[(size_t)c * 256];
        for (int s = 0; s < nu; ++s) {
            while (nb < PROB_BITS) { acc |= (uint32_t)*d++ << nb; nb += 8; }
            f[s] = (uint16_t)(acc & (PROB - 1)); acc >>= PROB_BITS; nb -= PROB_BITS;
            s0 += f[s];
        }
        if (s0 == 0) f[nu - 1] = PROB;   // a single symbol: 4096 stored as 0
        else if (s0 != PROB) return false;
        s0 = 0;
        for (int s = 0; s < nu; ++s) {
            st[(size_t)c * 256 + s] = (uint16_t)s0;
            if (f[s]) std::memset(&slot[(size_t)c * PROB + s0], s, f[s]);
            s0 += f[s];
        }
    }
    if (end - d < 4) return false;
    uint32_t x = ((uint32_t)d[0] << 24) | ((uint32_t)d[1] << 16) | ((uint32_t)d[2] << 8) | d[3];
    d += 4;
    const auto get = [&](int c) -> int {   // -1: the encoder never used this context (corrupt stream)
        const uint32_t sl = x & (PROB - 1);
        const int s = slot[(size_t)c * PROB + sl];
        const uint32_t f = fr[(size_t)c * 256 + s];
        if (f == 0) return -1;
        x = f * (x >> PROB_BITS) + sl - st[(size_t)c * 256 + s];
        while (x < RANS_L) x = (x << 8) | (d < end ? *d++ : 0);
        return s;
    };
    int prev = A / 2;
    for (uint32_t ix : pos[0]) {
        const int z = get(CTX_P0);
        if (z < 0) return false;
        prev = from_sym(z, prev, A);
        val[ix] = (uint8_t)prev;
    }
    std::vector<uint8_t> pr(maxp), cx(maxp);
    for (int p = 1; p < NPASS; ++p) {
        const std::vector<uint32_t>& ps = pos[p];
        const long long cnt = (long long)ps.size();
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < cnt; ++i) {
            const int y = (int)(ps[i] / (uint32_t)cols), xx = (int)(ps[i] % (uint32_t)cols);
            predict(T, val, mask, rows, cols, p, y, xx, A - 1, pr[i], cx[i]);
        }
        for (long long i = 0; i < cnt; ++i) {
            const int z = get(cx[i]);
            if (z < 0) return false;
            val[ps[i]] = (uint8_t)from_sym(z, pr[i], A);
        }
    }
    return true;
}
