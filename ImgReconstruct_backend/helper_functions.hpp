#ifndef IMGRECONSTRUCT_BACKEND_HELPER_FUNCTIONS_HPP_
#define IMGRECONSTRUCT_BACKEND_HELPER_FUNCTIONS_HPP_

#include "image_tiles.hpp"
#include "container_format.hpp"

#include <opencv2/core.hpp>

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// A tile's sample positions (row, column lists).
struct indices {
    std::vector<int> ri_x_g, ri_y_g;
};

// The FISTA basis: whole-plane DCT, its data-term Lipschitz constant and the
// per-coefficient soft-threshold weights (cs_make_dct_basis).
struct cs_dct_basis {
    float lip = 2.0f;
    std::vector<float> wscale;
};
cs_dct_basis cs_make_dct_basis(int rows, int cols, float highest_weight);

// Noise sigma estimate of a BGR tile in gray levels: Immerkaer Laplacian
// (sigma = sqrt(pi/2) * mean|Lo| / 6) per 32px block, MEDIAN over blocks.
// The median separates grain from texture — texture raises some blocks
// while flat blocks stay at the noise floor, so textured-but-clean tiles
// read low and uniformly grainy tiles read high. Single block for small
// tiles. Returns 0 on degenerate input. Pure function (deterministic).
double cs_estimate_noise_sigma(const cv::Mat& bgr);

// Bit-packing of the samples (the encoder's in-memory anchor-pass container; bit
// format in container_format.hpp): sections record raw offset/count and their
// plane depth in ascending order; everything before the first section copies
// verbatim (header + prefix); each section packs independently, whole-byte
// padded. Unpack mirrors it; both throw std::runtime_error on truncation.
cv::Mat cs_pack_container_body(const cv::Mat& raw,
    const std::vector<cs_body_section>& sections);
size_t cs_unpacked_body_bytes(const std::vector<cs_body_section>& sections);
void cs_unpack_container_body(const uint8_t* packed, size_t packed_bytes,
    const std::vector<cs_body_section>& sections, uint8_t* raw);

std::vector<cv::Mat> createRefSolutions(const int& rows, const int& cols);

/** @brief single-channel reweighted-L1 FISTA over the DCT (exact proximal soft-threshold).
Each outer reweight gets cs_fista_inner_iters(iterations, fista_iters) FISTA steps;
`reweights` counts outer passes (first unweighted); wscale is the highest-frequency
threshold weight. DCT warm start in, pixel plane 0..255 out after the tail scale.
*/
// Default sink for the optional next_ref chain output (a temporary cannot
// bind to the non-const reference; never written unless copy_next_ref).
inline cv::Mat& cs_null_mat() { static cv::Mat m; return m; }
void reconstruct_color_channel_fista(const cv::Mat& pixel_measurements, const int& channel, const float& param_c,
    const int& rows, const int& cols, const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    const int& iterations, cv::Mat& ref, bool copy_next_ref = false, cv::Mat& next_ref = cs_null_mat(),
    int reweights = 2, int fista_iters = 0, float wscale = 2.0f);

/** @brief samples of one channel in [0,1] (b) and a smooth starting image (fill)
built from them by normalized Gaussian convolution (the tile solves' warm start). */
void cs_point_sample_fill(const cv::Mat& pixel_measurements, int channel, int rows, int cols,
    const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    std::vector<float>& b, std::vector<float>& fill);

// Automatic luma coef of the tile solve (parameter sweep): 0.01125/ratio, times the
// tile-grid factor (num_tiles/64)^0.15 in [0.5, 1.2] below ratio 0.5. Ratio is
// clamped to [0.05, 1.0], the coef to [0.01, 0.3].
float cs_auto_reg_coefficient(float ratio, int num_tiles);

// Whole-plane 2D DCT on a CV_32F plane, in place (forward and inverse are exact adjoints).
void cs_dct_forward(cv::Mat& plane);
void cs_dct_inverse(cv::Mat& plane);

/** @brief HR-anchor context for the CS-SR fused solve.
Exact native-luma anchors (global output-geometry indices) enter the HR
coefficient solve as extra data terms beside the LR samples, so the anchor
neighborhoods improve too — not just the anchor pixels (post-hoc
replacement still pins those exactly afterwards). Anchors outside this
tile map to nothing here (neighbor tiles own them). Empty anchors, zero
stride disable the term. */
struct CsHrSrCtx {
    const std::vector<CsHrSample>* anchors = nullptr;
    int stride = 0;     // output-geometry row stride (image width)
    int ox = 0, oy = 0; // this tile's origin in output geometry
};

/** @brief CS super-resolution 2x refinement of one solved LR tile (CPU, DCT).
Luminance-only PnP-ADMM (rho 0.1, admm_iters outer passes, clamped [1, 50]):
the x-update is a sparse DCT FISTA solve so the 2x2-box downsample of the HR
luma fits the LR luma samples (BGR samples combined with the BGR2YCrCb
weights) and the exact-HR anchors in hr_sr; the z-update is a sigma-matched
fastNlMeans prox, or the identity with prox_off (the decoder's). The warm start and the Cr/Cb planes are the AVIR
upscale of the tile, so one HR solve runs instead of three. Falls back to
AVIR when the sample set is empty or the tile is degenerate. lr_tile is the
solved CV_8UC3 tile; hr_out is the 2x CV_8UC3 result. */
void cs_sr_upscale_tile_2x(const cv::Mat& lr_tile, const cv::Mat& pixel_measurements,
    const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    float coef, int iterations, int fista_iters, float wscale, int admm_iters, bool prox_off,
    cv::Mat& hr_out, const CsHrSrCtx& hr_sr = CsHrSrCtx());

/** Compact HR anchors (mode 19): which 16x16 blocks hold anchors, how many,
and the exact luma they come from. Positions are not stored; both sides draw
them from (flags, seed). */
#define CS_ANCHOR_BLOCK 16
struct CsAnchorPlan {
    cv::Mat flags;   // CV_8U, (H / 16) x (W / 16), 1 = block holds anchors
    cv::Mat luma;    // CV_8U Y plane at decode output geometry (W x H)
    int count = 0;   // anchors to draw (clamped to the flagged pixel count)
};

/** @brief closed-loop block choice: flag the worst 5% of 16x16 blocks by
MSE between the original and a decode of its container (same size, BGR
8-bit), or more when `count` anchors would fill them past 74% (production's
fill: 450k anchors on 5% of a 12.2 MP photo's blocks), so the share grows
with the count. Empty Mat when the inputs mismatch or are
smaller than a block. */
cv::Mat cs_error_anchor_flags(const cv::Mat& orig_bgr, const cv::Mat& dec_bgr, int count = 0);

/** @brief mode-19 anchor payload:
[u32 K][u16 block][u32 W][u32 H][u32 L][L bytes PNG of flags][K luma bytes].
Positions are K distinct pixels of the flagged blocks drawn from the seed
(cs_anchor_positions), so only the luma bytes grow with K. Throws on an
empty plan or a PNG encode failure. */
std::vector<uint8_t> cs_anchor_payload(const CsAnchorPlan& plan, const uint8_t seed[32]);

/** @brief parse a mode-19 payload and regenerate the anchors
(idx in W x H output geometry, ascending). False on any inconsistency. */
bool cs_read_anchor_payload(const uint8_t* src, size_t len, const uint8_t seed[32], std::vector<CsHrSample>& out);

/** @brief apply HR-detail anchors to a BGR image in place: converts to
YCrCb, replaces the Y plane at each in-range anchor index with the stored
value, converts back. Out-of-range indices are skipped (e.g. a half-res
decode of a half-res-pipeline container). Anchors whose exact Y plus the
image chroma falls outside BGR gamut (clipped on the way back, Y drifts)
are reverted instead of injecting clip artifacts. No-op on empty
input/payload. Returns the number of anchors applied. */
size_t cs_apply_hr_detail(cv::Mat& bgr, const std::vector<CsHrSample>& s);

#endif  // IMGRECONSTRUCT_BACKEND_HELPER_FUNCTIONS_HPP_
