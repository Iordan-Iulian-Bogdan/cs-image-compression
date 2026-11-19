#ifndef ENCODE_IMAGE_HPP
#define ENCODE_IMAGE_HPP

#include "cs_sampler.hpp"
#include <functional>

class encode_image : CsSampler
{
private:
    // encode_ycc420_segmask_ids keeps its sampled planes (Y, 4:2:0 Cr / Cb) and label PNG for a same_as_last call.
    cv::Mat seg_y, seg_cr, seg_cb;
    std::vector<uint8_t> seg_png;

    // Container build steps (see .cpp).
    uint8_t* alloc_container(int total);
    void code_container_body(size_t y0, int mY, int mC, int crows, int ccols, size_t anchor_len, int sample_bits,
        int chroma_bits);

public:
    encode_image(const cv::Mat& input, bool global_image);

    encode_image() {}

    void get_mat(cv::Mat& dest);

    // Runs `decode` on this object's container (the closed loop's anchor pass) with its sample draws on offer to
    // the decoder (CsSampler::draw_hint), so that decode does not regenerate them.
    cv::Mat decode_own_container(const std::function<cv::Mat(const cv::Mat&)>& decode);

    cv::Mat get_mat();

    // Per-tile mini-container (header + the samples at the given positions) for the decoder's tile solves.
    void encode(const std::vector<int>& ri_x_g, const std::vector<int>& ri_y_g);

    // A layer (mode 11) from a label map: rates[c] in [0,1] is the luma sampling rate of class c; 6-bit luma,
    // 8-bit 4:2:0 chroma drawn inside the label support. Body: [seg prefix + label PNG][Y x mY][Cr x mC][Cb x mC];
    // decode regenerates the positions from the shipped prefix. A non-null anchor plan switches to
    // CS_MODE_YCC420_SEG_HRDETAIL (mode 19) with the compact anchor payload (see cs_anchor_payload). coded: the
    // three planes entropy-coded (CS_OFF_SAMPLE_CODING), else bit-packed. same_as_last: this object's previous call
    // had the same labels and rates; its draws and sampled planes are reused.
    void encode_ycc420_segmask_ids(const cv::Mat& ids, const std::vector<float>& rates,
        const CsAnchorPlan* anchors = nullptr, bool coded = true, bool same_as_last = false);

    // The luma sample positions of the last encode_* call (rows, cols).
    void sample_positions(std::vector<int>& rows, std::vector<int>& cols) const;
};

// Encodes a photo into a multi-scale container (mode 20): plans every 120 px tile's scale (2x..6x) and luma rate
// for the byte budget of ratio, and writes one seg layer (mode 11) per scale in use. The 2x layer gets closed-loop
// anchors (mode 19): anchor_decode decodes its container, and hr_detail x its share of the tiles anchors go to the
// worst 16x16 blocks of that decode. Writes the container to output_path; returns 0 on success.
// show_mask (--show-mask): every layer's luma samples on the 2x layer's grid (W/2 x H/2; a sample of the s-x layer
// at its pixel's centre, so coarse tiles show sparse dots) as <output>.mask.png, and in a window when mask_window.
int cs_encode_multiscale(const std::string& input_path, const std::string& output_path,
    int hr_detail, const std::function<cv::Mat(const cv::Mat&)>& anchor_decode, float ratio, bool show_mask = false,
    bool mask_window = true, float coarse = 0.0f);

#endif
