#include "helper_functions.hpp"
#include "fista_kernels.hpp"
#include "container_format.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>


namespace {
// container pixels (3 bytes each) holding `bytes`
int cs_container_pixels(size_t bytes) {
    return (int)((bytes + 2) / 3);
}
} // namespace

cv::Mat cs_pack_container_body(const cv::Mat& raw,
    const std::vector<cs_body_section>& sections) {
    const size_t raw_bytes = raw.total() * raw.elemSize();
    if (sections.empty()) {
        return raw.clone();
    }
    const size_t first = sections[0].offset;
    if (first > raw_bytes) {
        throw std::runtime_error("packed container: section starts past end of body");
    }
    bool any_packed = false;
    size_t packed_len = first;
    for (const auto& s : sections) {
        if (!cs_sample_bits_valid(s.bits)) {
            throw std::runtime_error("sample-bits out of range [1, 8]");
        }
        if (s.offset < first || s.offset > raw_bytes || s.count > raw_bytes - s.offset) {
            throw std::runtime_error("packed container: sections out of order");
        }
        any_packed = any_packed || s.bits < 8;
        packed_len += cs_packed_bytes(s.count, s.bits);
    }
    if (!any_packed) return raw.clone();
    const int total_p = cs_container_pixels(packed_len);
    cv::Mat out(1, total_p, CV_8UC3, cv::Scalar(0, 0, 0));
    std::memcpy(out.data, raw.data, first);
    size_t woff = first;
    for (const auto& s : sections) {
        cs_pack_samples(raw.data + s.offset, s.count, s.bits, out.data + woff);
        woff += cs_packed_bytes(s.count, s.bits);
    }
    return out;
}

size_t cs_unpacked_body_bytes(const std::vector<cs_body_section>& sections) {
    if (sections.empty()) return 0;
    size_t len = sections[0].offset;
    for (const auto& s : sections) len += s.count;
    return len;
}

void cs_unpack_container_body(const uint8_t* packed, size_t packed_bytes,
    const std::vector<cs_body_section>& sections, uint8_t* raw) {
    if (sections.empty()) return;
    const size_t first = sections[0].offset;
    size_t needed = first;
    for (const auto& s : sections) {
        if (!cs_sample_bits_valid(s.bits)) {
            throw std::runtime_error("sample-bits out of range [1, 8]");
        }
        if (s.offset < first) {
            throw std::runtime_error("invalid packed container section");
        }
        needed += cs_packed_bytes(s.count, s.bits);
    }
    if (needed > packed_bytes) {
        throw std::runtime_error("packed container: body shorter than section table");
    }
    std::memcpy(raw, packed, first);
    size_t roff = first;
    for (const auto& s : sections) {
        cs_unpack_samples(packed + roff, s.count, s.bits, raw + s.offset);
        roff += cs_packed_bytes(s.count, s.bits);
    }
}

// Generic initial solution of a tile: a neutral gray plane per channel in the solver's
// coefficient domain (DCT / 10).
std::vector<cv::Mat> createRefSolutions(const int& rows, const int& cols) {
    const cv::Mat ref(rows, cols, CV_8UC3, cv::Scalar(128, 128, 128));
    std::vector<cv::Mat> c;
    cv::split(ref, c);

    for (int i = 0; i < 3; i++) {
        c[i].convertTo(c[i], CV_32F);
        c[i] = c[i] / 255.0f;
        cs_dct_forward(c[i]);
        c[i] = c[i] / 10.0f;
    }

    return c;
}

// ---------------------------------------------------------------------------
// FISTA + reweighted L1 (DCT)
// ---------------------------------------------------------------------------
// Under DCT the forward operator A = P * IDCT is a row-selected orthonormal
// transform, so ||A|| = 1 and the data term f(x) = ||Ax - b||^2 has
// Lipschitz constant L = 2 exactly. That makes FISTA's proximal step exact
// and cheap: one IDCT + one DCT per iteration with no line search.
// Reweighting (Candes et al.) runs 2 outer passes with normalized
// weights w = eps/(|x|+eps) so large coefficients are protected while small
// ones are pushed harder toward zero.

float cs_auto_reg_coefficient(float ratio, int num_tiles) {
    float r = ratio < 0.05f ? 0.05f : (ratio > 1.0f ? 1.0f : ratio);
    const int nt = num_tiles > 0 ? num_tiles : 1;
    // Tile-grid factor below ratio 0.5: small tiles hold fewer pixels per
    // solve and need a stronger prior.
    float tf = 1.0f;
    if (r < 0.5f) {
        tf = std::pow((float)nt / 64.0f, 0.15f);
        if (tf < 0.5f) tf = 0.5f;
        if (tf > 1.2f) tf = 1.2f;
    }
    // 0.0225/r tracked the sweep optimum before the anchors; with them, half
    // of it won at r=0.3..0.9: +0.18..0.31 dB, dE-p95 better, SSIM flat.
    // The per-tile chroma solve then halves it (coef_c = 0.5*coef).
    const float base = 0.01125f / r;
    float coef = base * tf;
    if (coef < 0.01f) coef = 0.01f;
    if (coef > 0.3f) coef = 0.3f;
    return coef;
}

// Whole-plane orthonormal 2D DCT of a CV_32F plane, in place. All solver-domain
// conversions (warm starts, tails) use these so analysis and synthesis agree.
void cs_dct_forward(cv::Mat& plane) {
    cv::dct(plane, plane, 0);
}

void cs_dct_inverse(cv::Mat& plane) {
    cv::dct(plane, plane, cv::DCT_INVERSE);
}

// Per-coefficient soft-threshold weights: highest_weight^f with f in [0,1] the
// normalized 2D frequency (DC -> 1, highest AC -> highest_weight); highest_weight
// <= 1 keeps uniform weights.
cs_dct_basis cs_make_dct_basis(int rows, int cols, float highest_weight) {
    cs_dct_basis basis;
    basis.wscale.assign(static_cast<size_t>(rows) * cols, 1.0f);
    if (highest_weight <= 1.0f) return basis;
    const double logw = std::log(static_cast<double>(highest_weight));
    for (int r = 0; r < rows; ++r) {
        const double fn_r = rows > 1 ? (double)r / (rows - 1) : 0.0;
        for (int c = 0; c < cols; ++c) {
            const double fn_c = cols > 1 ? (double)c / (cols - 1) : 0.0;
            const double f = 0.5 * (fn_r + fn_c);
            basis.wscale[(size_t)r * cols + c] = static_cast<float>(std::exp(logw * f));
        }
    }
    return basis;
}

// Gradient of the smooth part in the coefficient domain:
//   g = 2*IDCT^T(scatter(IDCT(y) - b)). pix/sc are scratch (size n).
static void cs_fista_grad(const float* y, float* pix, float* sc, float* g,
    const float* b, const int* rix, const int* riy, int m, int rows, int cols)
{
    const int n = rows * cols;
    std::memcpy(pix, y, sizeof(float) * (size_t)n);
    cv::Mat P(rows, cols, CV_32F, pix);
    cs_dct_inverse(P);
    std::memset(sc, 0, sizeof(float) * (size_t)n);
    for (int k = 0; k < m; ++k) {
        const int idx = rix[k] * cols + riy[k];
        sc[idx] = pix[idx] - b[k];
    }
    cv::Mat S(rows, cols, CV_32F, sc);
    cs_dct_forward(S);
    for (int i = 0; i < n; ++i) g[i] = 2.0f * sc[i];
}

// One weighted-L1 FISTA run: min ||Ax-b||^2 + lambda*sum(w|x|) with the exact
// fixed step 1/lip. x is the warm start in, solution out. y/x_prev/grad/pix/sc
// are scratch.
void cs_fista_core_single(float* x, const float* b, const int* rix, const int* riy, int m,
    int rows, int cols, const float* w, float lambda, int iters,
    const cs_dct_basis& bx,
    std::vector<float>& y, std::vector<float>& x_prev,
    std::vector<float>& grad, std::vector<float>& pix, std::vector<float>& sc)
{
    const int n = rows * cols;
    const float L = bx.lip;
    std::memcpy(y.data(), x, sizeof(float) * (size_t)n);
    std::memcpy(x_prev.data(), x, sizeof(float) * (size_t)n);
    std::vector<float> z((size_t)n);
    float* yy = y.data();
    float t = 1.0f;
    for (int k = 0; k < iters; ++k) {
        cs_fista_grad(yy, pix.data(), sc.data(), grad.data(), b, rix, riy, m, rows, cols);
        const float step = 1.0f / L;
        const float base = lambda * step;
        // simd-safe: z is local; yy/grad/w/wscale are distinct inputs.
        #pragma omp simd
        for (int i = 0; i < n; ++i) {
            const float v = yy[i] - step * grad[i];
            const float thr = base * w[i] * bx.wscale[i];
            const float av = std::fabs(v);
            z[(size_t)i] = (av > thr) ? ((v > 0.0f ? 1.0f : -1.0f) * (av - thr)) : 0.0f;
        }
        const float t_new = 0.5f * (1.0f + std::sqrt(1.0f + 4.0f * t * t));
        const float mom = (t - 1.0f) / t_new;
        // simd-safe: yy/x_prev are distinct caller vectors, z is local.
        #pragma omp simd
        for (int i = 0; i < n; ++i) yy[i] = z[(size_t)i] + mom * (z[(size_t)i] - x_prev[(size_t)i]);
        std::memcpy(x_prev.data(), z.data(), sizeof(float) * (size_t)n);
        t = t_new;
    }
    std::memcpy(x, z.data(), sizeof(float) * (size_t)n);
}

// Per-pass FISTA step count: fista_iters when positive, else mapped from the
// iteration budget (iterations * 4 in [16, 40]).
int cs_fista_inner_iters(int iterations, int fista_iters) {
    if (fista_iters > 0) {
        if (fista_iters > 500) fista_iters = 500;
        return fista_iters;
    }
    int inner = iterations * 4;
    if (inner < 16) inner = 16;
    if (inner > 40) inner = 40;
    return inner;
}

void cs_extract_channel_measurements(const cv::Mat& pixel_measurements, int channel,
    int m, std::vector<float>& b)
{
    b.assign((size_t)m, 0.0f);
    const int avail = (int)pixel_measurements.total() - CS_HEADER_PIXELS;
    const int cnt = avail < m ? (avail < 0 ? 0 : avail) : m;
    for (int i = 0; i < cnt; ++i) {
        b[(size_t)i] = pixel_measurements.at<cv::Vec3b>(i + CS_HEADER_PIXELS)[channel] / 255.0f;
    }
}

void cs_point_sample_fill(const cv::Mat& pixel_measurements, int channel, int rows, int cols,
    const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    std::vector<float>& b, std::vector<float>& fill)
{
    const int n = rows * cols;
    const int m = (int)ri_x.size();
    cs_extract_channel_measurements(pixel_measurements, channel, m, b);
    // Gaussian-weighted average of the samples (normalized convolution), width
    // scaled to the sample spacing; the global mean where no sample is near.
    cv::Mat val = cv::Mat::zeros(rows, cols, CV_32F), msk = cv::Mat::zeros(rows, cols, CV_32F);
    double mean = 0.0;
    for (int k = 0; k < m; ++k) {
        const int idx = ri_x[(size_t)k] * cols + ri_y[(size_t)k];
        val.ptr<float>(0)[idx] += b[(size_t)k];
        msk.ptr<float>(0)[idx] += 1.0f;
        mean += b[(size_t)k];
    }
    mean = m > 0 ? mean / m : 0.5;
    const float rho = (std::max)(1e-3f, (float)m / (float)n);
    const double sigma = (std::min)(12.0, (std::max)(1.0, 1.5 / std::sqrt((double)rho)));
    cv::Mat num, den;
    cv::GaussianBlur(val, num, cv::Size(0, 0), sigma);
    cv::GaussianBlur(msk, den, cv::Size(0, 0), sigma);
    fill.resize((size_t)n);
    for (int i = 0; i < n; ++i) {
        const float dn = den.ptr<float>(0)[i];
        fill[(size_t)i] = dn > 1e-4f ? num.ptr<float>(0)[i] / dn : (float)mean;
    }
}

void reconstruct_color_channel_fista(const cv::Mat& pixel_measurements, const int& channel, const float& param_c,
    const int& rows, const int& cols, const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    const int& iterations, cv::Mat& ref, bool copy_next_ref, cv::Mat& next_ref,
    int reweights, int fista_iters, float wscale)
{
    if (!ref.isContinuous()) ref = ref.clone();
    const int m = (int)ri_x.size();
    if (reweights < 1) reweights = 1;
    if (reweights > 5) reweights = 5;
    const int n = rows * cols;
    const cs_dct_basis bx = cs_make_dct_basis(rows, cols, wscale);

    std::vector<float> b;
    cs_extract_channel_measurements(pixel_measurements, channel, m, b);

    std::vector<float> w((size_t)n, 1.0f);
    std::vector<float> y((size_t)n), x_prev((size_t)n), grad((size_t)n), pix((size_t)n), sc((size_t)n);
    const int inner = cs_fista_inner_iters(iterations, fista_iters);
    float* x = (float*)ref.data;
    for (int r = 0; r < reweights; ++r) {
        cs_fista_core_single(x, b.data(), ri_x.data(), ri_y.data(), m, rows, cols,
            w.data(), param_c, inner, bx, y, x_prev, grad, pix, sc);
        if (r + 1 < reweights) {
            float mx = 0.0f;
            for (int i = 0; i < n; ++i) {
                const float av = std::fabs(x[i]);
                if (av > mx) mx = av;
            }
            float eps = 0.02f * mx;
            if (eps < 1e-3f) eps = 1e-3f;
            for (int i = 0; i < n; ++i) w[(size_t)i] = eps / (std::fabs(x[i]) + eps);
        }
    }

    if (copy_next_ref && !next_ref.empty()) {
        if (!next_ref.isContinuous()) next_ref = next_ref.clone();
        std::memcpy(next_ref.data, ref.data, sizeof(float) * (size_t)n);
    }
    cv::Mat AtAxb2(rows, cols, CV_32F, (float*)ref.data);
    cs_dct_inverse(AtAxb2);
    AtAxb2 = AtAxb2 * 255.0f;
}

double cs_estimate_noise_sigma(const cv::Mat& bgr)
{    if (bgr.empty() || bgr.type() != CV_8UC3) return 0.0;
    if (bgr.rows < 4 || bgr.cols < 4) return 0.0;
    cv::Mat ycc;
    cv::cvtColor(bgr, ycc, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> chs;
    cv::split(ycc, chs);
    cv::Mat yf;
    chs[0].convertTo(yf, CV_32F);
    // Immerkaer Laplacian (valid interior only: 1px border excluded), then
    // median block sigma so texture does not pose as grain (see header).
    static const float kdata[9] = { 1, -2, 1, -2, 4, -2, 1, -2, 1 };
    const cv::Mat kernel(3, 3, CV_32F, (void*)kdata);
    cv::Mat lo;
    cv::filter2D(yf, lo, CV_32F, kernel, cv::Point(-1, -1), 0.0,
        cv::BORDER_REPLICATE);
    cv::Mat ab;
    cv::absdiff(lo, cv::Scalar::all(0), ab);
    static const double kScale = 1.2533141373155001 / 6.0;  // sqrt(pi/2)/6
    constexpr int BS = 32;
    std::vector<double> blocks;
    blocks.reserve((size_t)((yf.rows + BS - 1) / BS) * ((yf.cols + BS - 1) / BS));
    for (int y0 = 1; y0 + 1 < yf.rows; y0 += BS) {
        for (int x0 = 1; x0 + 1 < yf.cols; x0 += BS) {
            const int w = (std::min)(BS, yf.cols - 1 - x0);
            const int h = (std::min)(BS, yf.rows - 1 - y0);
            if (w < 8 || h < 8) continue;
            blocks.push_back(kScale * cv::mean(ab(cv::Rect(x0, y0, w, h)))[0]);
        }
    }
    if (blocks.empty()) return 0.0;
    std::nth_element(blocks.begin(), blocks.begin() + blocks.size() / 2, blocks.end());
    return blocks[blocks.size() / 2];
}


cv::Mat cs_error_anchor_flags(const cv::Mat& orig_bgr, const cv::Mat& dec_bgr, int count) {
    if (orig_bgr.empty() || orig_bgr.type() != CV_8UC3 ||
        dec_bgr.size() != orig_bgr.size() || dec_bgr.type() != CV_8UC3)
        return cv::Mat();
    constexpr int BS = CS_ANCHOR_BLOCK;
    const int bh = orig_bgr.rows / BS, bw = orig_bgr.cols / BS;
    if (bh == 0 || bw == 0) return cv::Mat();
    // Mean squared error per 16x16 block (all channels), then flag the
    // worst 5% of blocks: the closed-loop choice that the measured
    // anchor gains were obtained with. More anchors than 74% of those
    // blocks' pixels (production's fill) flag more blocks, at that fill.
    std::vector<float> score((size_t)bh * bw);
    for (int by = 0; by < bh; ++by) {
        for (int bx = 0; bx < bw; ++bx) {
            double s = 0.0;
            for (int y = by * BS; y < by * BS + BS; ++y) {
                const cv::Vec3b* o = orig_bgr.ptr<cv::Vec3b>(y);
                const cv::Vec3b* d = dec_bgr.ptr<cv::Vec3b>(y);
                for (int x = bx * BS; x < bx * BS + BS; ++x) {
                    for (int c = 0; c < 3; ++c) {
                        const int e = (int)o[x][c] - (int)d[x][c];
                        s += (double)(e * e);
                    }
                }
            }
            score[(size_t)by * bw + bx] = (float)(s / (BS * BS * 3));
        }
    }
    std::vector<float> sorted = score;
    const size_t nfill = (size_t)std::ceil((double)count / (0.74 * BS * BS));
    const size_t nflag = (std::min)(sorted.size(), (std::max)((std::max<size_t>)(1, (size_t)std::llround(0.05 * (double)sorted.size())), nfill));
    std::nth_element(sorted.begin(), sorted.begin() + (sorted.size() - nflag), sorted.end());
    const float thr = sorted[sorted.size() - nflag];
    cv::Mat flags(bh, bw, CV_8U);
    for (int by = 0; by < bh; ++by)
        for (int bx = 0; bx < bw; ++bx)
            flags.at<uint8_t>(by, bx) = score[(size_t)by * bw + bx] >= thr ? 1 : 0;
    return flags;
}

// K distinct pixels of the flagged blocks: candidates in block-raster then
// pixel-raster order, partial Fisher-Yates from the anchor-tagged seed (same
// generator convention as the sample draws), returned ascending.
static std::vector<uint32_t> cs_anchor_positions(const cv::Mat& flags, int width,
    int count, const uint8_t seed[32]) {
    constexpr int BS = CS_ANCHOR_BLOCK;
    std::vector<uint32_t> pool;
    for (int by = 0; by < flags.rows; ++by) {
        for (int bx = 0; bx < flags.cols; ++bx) {
            if (!flags.at<uint8_t>(by, bx)) continue;
            for (int y = by * BS; y < by * BS + BS; ++y)
                for (int x = bx * BS; x < bx * BS + BS; ++x)
                    pool.push_back((uint32_t)y * (uint32_t)width + (uint32_t)x);
        }
    }
    const int n = (int)pool.size();
    if (count > n) count = n;
    uint8_t s2[32];
    std::memcpy(s2, seed, 32);
    s2[0] ^= 0x48;   // the anchors' own stream, apart from the sample draws
    std::seed_seq seq(s2, s2 + 32);
    std::mt19937 generator(seq);
    for (int k = 0; k < count; ++k) {
        std::uniform_int_distribution<int> pick(k, n - 1);
        std::swap(pool[(size_t)k], pool[(size_t)pick(generator)]);
    }
    pool.resize((size_t)count);
    std::sort(pool.begin(), pool.end());
    return pool;
}

static void put_u32(std::vector<uint8_t>& v, uint32_t x) {
    for (int b = 0; b < 4; ++b) v.push_back((uint8_t)((x >> (b * 8)) & 0xff));
}
static uint32_t get_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

std::vector<uint8_t> cs_anchor_payload(const CsAnchorPlan& plan, const uint8_t seed[32]) {
    const int W = plan.luma.cols, H = plan.luma.rows;
    if (plan.count <= 0 || plan.luma.type() != CV_8U || plan.flags.type() != CV_8U ||
        plan.flags.rows != H / CS_ANCHOR_BLOCK || plan.flags.cols != W / CS_ANCHOR_BLOCK) {
        throw std::runtime_error("anchor payload: inconsistent plan (internal error)");
    }
    const std::vector<uint32_t> pos = cs_anchor_positions(plan.flags, W,
        (std::min)(plan.count, CS_HRDETAIL_MAX_SAMPLES), seed);
    if (pos.empty()) throw std::runtime_error("anchor payload: no flagged blocks");
    std::vector<uint8_t> png;
    if (!cv::imencode(".png", plan.flags, png, { cv::IMWRITE_PNG_COMPRESSION, 9 }) || png.empty()) {
        throw std::runtime_error("anchor payload: flag map encode failed");
    }
    std::vector<uint8_t> out;
    out.reserve(18 + png.size() + pos.size());
    put_u32(out, (uint32_t)pos.size());
    out.push_back((uint8_t)(CS_ANCHOR_BLOCK & 0xff));
    out.push_back((uint8_t)(CS_ANCHOR_BLOCK >> 8));
    put_u32(out, (uint32_t)W);
    put_u32(out, (uint32_t)H);
    put_u32(out, (uint32_t)png.size());
    out.insert(out.end(), png.begin(), png.end());
    for (uint32_t p : pos) out.push_back(plan.luma.data[(size_t)(p / W) * plan.luma.step + p % W]);
    return out;
}

bool cs_read_anchor_payload(const uint8_t* src, size_t len, const uint8_t seed[32], std::vector<CsHrSample>& out) {
    out.clear();
    if (len < 18) return false;
    const uint8_t* buf = src;
    const uint32_t K = get_u32(buf);
    const int bs = buf[4] | (buf[5] << 8);
    const uint32_t W = get_u32(buf + 6), H = get_u32(buf + 10);
    const uint32_t L = get_u32(buf + 14);
    bool ok = bs == CS_ANCHOR_BLOCK && K > 0 && K <= CS_HRDETAIL_MAX_SAMPLES &&
        W >= (uint32_t)bs && H >= (uint32_t)bs && W <= (1u << 16) && H <= (1u << 16) &&
        (size_t)18 + L + K == len;
    cv::Mat flags;
    if (ok) {
        flags = cv::imdecode(std::vector<uint8_t>(buf + 18, buf + 18 + L), cv::IMREAD_GRAYSCALE);
        ok = !flags.empty() && flags.rows == (int)(H / bs) && flags.cols == (int)(W / bs);
    }
    std::vector<uint32_t> pos;
    if (ok) {
        pos = cs_anchor_positions(flags, (int)W, (int)K, seed);
        ok = pos.size() == K;
    }
    if (ok) {
        const uint8_t* y = buf + 18 + L;
        out.resize(K);
        for (uint32_t k = 0; k < K; ++k) {
            out[k].idx = pos[k];
            out[k].y = y[k];
        }
    }
    return ok;
}

size_t cs_apply_hr_detail(cv::Mat& bgr, const std::vector<CsHrSample>& s) {
    if (bgr.empty() || bgr.type() != CV_8UC3 || s.empty()) return 0;
    const int W = bgr.cols, H = bgr.rows;
    const size_t total = (size_t)W * H;
    cv::Mat ycc;
    cv::cvtColor(bgr, ycc, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> chs;
    cv::split(ycc, chs);
    const cv::Mat orig_y = chs[0].clone();
    size_t in_range = 0;
    for (const CsHrSample& sm : s) {
        if ((size_t)sm.idx >= total) continue;
        chs[0].at<uint8_t>((int)(sm.idx / (uint32_t)W), (int)(sm.idx % (uint32_t)W)) = sm.y;
        ++in_range;
    }
    if (in_range == 0) return 0;
    cv::merge(chs, ycc);
    cv::cvtColor(ycc, bgr, cv::COLOR_YCrCb2BGR);
    // Gamut guard: exact Y plus solver chroma can fall outside BGR gamut
    // (clipped on the way back, Y drifts). Revert anchors that did not
    // survive the roundtrip instead of injecting clip artifacts.
    cv::Mat chk;
    cv::cvtColor(bgr, chk, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> chk_chs;
    cv::split(chk, chk_chs);
    size_t applied = 0;
    for (const CsHrSample& sm : s) {
        if ((size_t)sm.idx >= total) continue;
        const int y = (int)(sm.idx / (uint32_t)W), x = (int)(sm.idx % (uint32_t)W);
        if (std::abs((int)chk_chs[0].at<uint8_t>(y, x) - (int)sm.y) <= 1) {
            ++applied;
        } else {
            chs[0].at<uint8_t>(y, x) = orig_y.at<uint8_t>(y, x);
        }
    }
    if (applied < in_range) {
        cv::merge(chs, ycc);
        cv::cvtColor(ycc, bgr, cv::COLOR_YCrCb2BGR);
    }
    return applied;
}
