#ifndef ENCRYPT_IMAGE_HPP
#define ENCRYPT_IMAGE_HPP

#include "CS_encryption.hpp"
#include <functional>

class encrypt_image : CSencryption
{
private:
    float bm = 0.0f;
    // encrypt_ycc420_segmask_ids keeps its sampled planes (Y, 4:2:0 Cr / Cb) and label PNG for a same_as_last call.
    cv::Mat seg_y, seg_cr, seg_cb;
    std::vector<uint8_t> seg_png;

    // Container build steps (see .cpp).
    uint8_t* alloc_container(int total);
    static void stamp_sample_bits(uint8_t* buf, int sample_bits, int& chroma_bits);
    void finish_container(int sample_bits, int chroma_bits, const std::vector<cs_body_section>& sections);
    void code_container_body(size_t y0, int mY, int mC, int crows, int ccols, size_t anchor_len, int sample_bits,
        int chroma_bits);

public:
    encrypt_image(const cv::Mat& input, bool global_image);

    encrypt_image() {}

    // Native size parity recorded in the container (the input was padded to even by one replicated
    // row / column per odd side; decrypt crops it back).
    void set_odd_parity(int p) { odd_parity = p; }

    void get_mat(cv::Mat& dest);

    // Runs `decode` on this object's container (the closed loop's anchor pass) with its sample draws on offer to
    // the decoder (CSencryption::draw_hint), so that decode does not regenerate them.
    cv::Mat decode_own_container(const std::function<cv::Mat(const cv::Mat&)>& decode);

    cv::Mat get_mat();

    // Per-tile mini-container (header + the samples at the given positions) for the decoder's tile solves.
    void encrypt(const std::vector<int>& ri_x_g, const std::vector<int>& ri_y_g);

    // Per-segment luma + uniform 4:2:0 chroma (mode 11) from resolved ids:
    // rates[c] in [0,1] is the luma sampling rate of class c, scaled by pixel_p.
    // Body: [seg prefix + label PNG][Y x mY][Cr x mC][Cb x mC]; decrypt
    // regenerates the positions from the shipped prefix. A non-null anchor plan
    // switches to CS_MODE_YCC420_SEG_HRDETAIL (mode 19) with the compact anchor
    // payload (see cs_anchor_payload). chroma_masked (multi-scale layers): chroma
    // drawn inside the label support (seg prefix version 2). tcluster: luma drawn as T-tetromino groups
    // (--sampling tcluster; prefix version 3 with chroma_masked, else 4). coded (--sample-coding): the three
    // planes entropy-coded instead of bit-packed (CS_OFF_SAMPLE_CODING).
    void encrypt_ycc420_segmask_ids(const float& pixel_p,
        const cv::Mat& ids, const std::vector<float>& rates,
        int sample_bits = 8, int chroma_bits = 0, const CsAnchorPlan* anchors = nullptr, bool chroma_masked = false,
        bool tcluster = false, bool coded = false, bool same_as_last = false);

    // Build source | binary mask | cyan overlay from the indices used by the
    // last encrypt_* call. Returns false when no indices are available.
    bool build_sampling_mask_view(cv::Mat& out) const;
    // The luma sample positions of the last encrypt_* call (rows, cols).
    void sample_positions(std::vector<int>& rows, std::vector<int>& cols) const;

    // --show-mask: write <output>.mask.png and open a window until a key is
    // pressed (skipped when headless/no window support). output_path may be
    // empty (in-memory encrypt); then only the window is used.
    void present_sampling_mask(const std::string& output_path) const;

    /** @brief encrypts a photo into the single-scale production container (mode 19) and writes it to disk:
    --seg-auto classes split into --seg-tile detail levels, 4:2:0 chroma, and HR anchors placed by a closed loop.
    @param compression_ratio : --ratio in (0.001, 1.0]
    @param seg_auto : k-means classes (2..32)
    @param hr_detail : anchor count (1..CS_HRDETAIL_MAX_SAMPLES)
    @param sample_bits, chroma_bits : --sample-bits; sample_bits 0 = the default 6-bit luma / 8-bit chroma at the
        8-bit byte budget of --ratio
    @param show_mask : after encrypt render and show the sampling mask
    @param anchor_decode : decodes a container to BGR at decrypt output geometry; the anchors go to the worst
        blocks of that decode
    */
    static int encrypt_image_tiled(const std::string& input_path, const std::string& output_path,
        float compression_ratio, int seg_auto, int hr_detail, int sample_bits,
        int chroma_bits, bool show_mask, const std::function<cv::Mat(const cv::Mat&)>& anchor_decode,
        bool tcluster = false, bool sample_coding = false);
};

// Built-in scene segmentation (--seg-auto): k-means in Lab on a capped
// working copy, classes sorted by size desc, tiny components reassigned to
// the neighborhood mode, upscaled NEAREST to rows x cols. Deterministic
// (fixed OpenCV RNG seed, state restored). No models or downloads.
void cs_seg_auto_ids(const cv::Mat& bgr, int classes, int rows, int cols, cv::Mat& ids);
// Auto per-class rates from mean Sobel gradient energy (detail proxy):
// r = 0.08 + 0.92*(d/dmax)^0.75 over the classes present in ids; flat
// images (dmax == 0) get uniform 1.0.
std::vector<float> cs_seg_auto_rates(const cv::Mat& bgr, const cv::Mat& ids, int classes);
// Multi-scale container (mode 20, the default production encode): the encoder plans every 120 px tile's scale
// (2x..6x) and luma rate for the budget of a production container at ratio, and writes one seg layer (mode 11) per
// scale in use. hr_detail with anchor_decode: the 2x layer gets the production closed-loop anchors, hr_detail x its
// share of the tiles. Writes the container to output_path; returns 0 on success.
// show_mask (--show-mask): every layer's luma samples on the 2x layer's grid (W/2 x H/2; a sample of the s-x layer
// at its pixel's centre, so coarse tiles show sparse dots), as <output>.mask.png and a window like single-scale.
int cs_encrypt_multiscale(const std::string& input_path, const std::string& output_path,
    int hr_detail, const std::function<cv::Mat(const cv::Mat&)>& anchor_decode, float ratio, bool show_mask = false,
    bool tcluster = false, bool sample_coding = false, float coarse = 0.0f);

// Per-tile detail split (--seg-tile, the "seg8-tile" container): every 64x64
// tile gets an LOD from its mean |Laplacian| (min-max to [32, 255]) bucketed
// into 8 levels; label = class * 8 + level, rate = class rate x level weight
// (256 + bin-centre LOD) / the class's pixel-weighted mean weight, so each
// class keeps its share. The rates are then scaled by one factor s, solved so
// that sum(pixels x min(1, s x rate)) = pixel_p x all pixels: the same luma
// budget as a uniform/adaptive container at that ratio (busy labels cap at 1
// and the rest keep rising). ids (K -> 8K labels) and rates (final, encrypt
// at ratio 1) in place. Needs classes <= 32.
void cs_seg_tile_refine(const cv::Mat& bgr, cv::Mat& ids, std::vector<float>& rates, int classes,
    float pixel_p);

#endif
