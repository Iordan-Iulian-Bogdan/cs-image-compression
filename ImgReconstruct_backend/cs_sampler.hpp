#ifndef CS_SAMPLER_HPP
#define CS_SAMPLER_HPP


#include "helper_functions.hpp"
#include "container_format.hpp"

// Chroma support of a label map: CV_8U at the 4:2:0 chroma geometry, 255 where
// any luma pixel of the chroma pixel's block has a label with rate > 0. Labels
// at luma (pipeline) geometry rows x cols.
cv::Mat cs_seg_chroma_support(const cv::Mat& labels, const std::vector<uint8_t>& rates_q);

class CsSampler {
protected:
    cv::Mat encoded_img, input_img;
    cv::Size org_size;
    int odd_parity = 0;   // native size parity (cs_build_metadata): bit 0 odd height, bit 1 odd width
    int rows, cols, m, n;
    std::vector<int> ri_x, ri_y;

    // Every draw is seeded with cs_sample_seed (container_format.hpp).
    // Per-segment sampling: class ids from an 8-bit label map (xm x ym),
    // per-class sample counts from quantized rates
    // (count_c = round(pix_c * q_c / 255), ids >= rates_q.size() score 0).
    // One shared RNG stream draws classes in ascending id order (partial
    // Fisher-Yates over each class's pixel pool), then one global shuffle
    // fixes the measurement order — fully deterministic from (seed, labels,
    // rates) so encode and decode agree.
    void returnSegMaskIndices(std::vector<int>& ri_x, std::vector<int>& ri_y, int xm, int ym,
        const cv::Mat& labels, const std::vector<uint8_t>& rates_q, const uint8_t seed[32]);

    // Sampling mode state (decode side: restored from the container header;
    // encode side: set by encode_*). CS_MODE_* from container_format.hpp.
    int sampling_mode = 0;

    // Mode 19 (CS_MODE_YCC420_SEG_HRDETAIL): exact native-luma anchors carried
    // as a trailing body section. Filled by the decode-side
    // payload parse; enter the cs-sr solve as data terms.
    std::vector<CsHrSample> hr_samples;
    bool hr_detail = false;
    uint32_t hr_count = 0;

    // 4:2:0 chroma sampling positions on the half-resolution chroma grid +
    // per-plane chroma sample count, filled by returnYcc420ChromaIndices;
    // ri_x/ri_y keep the full-res luma positions.
    std::vector<int> ri_cx1, ri_cy1, ri_cx2, ri_cy2;
    int m_chroma = 0;
    // CV_8U at the chroma geometry: the chroma draws pick only its nonzero
    // pixels (cs_seg_chroma_support).
    cv::Mat chroma_support;

    // Chroma draws: fills ri_cx1/ri_cy1 (Cr), ri_cx2/ri_cy2 (Cb) and m_chroma
    // from tagged seed copies (see cs_ycc420_channel_seed). Fully deterministic
    // from (seed, m_luma, geometry, chroma_support) so encode and decode agree.
    void returnYcc420ChromaIndices(int xm, int ym, int m_luma, const uint8_t seed[32]);

    // The encoder's anchor pass decodes the container it has just written, on the same thread: while that decode
    // runs, draw_hint() points at the encoder, and the decoder copies its luma / chroma draws (the same seed, labels
    // and rates give the same lists) instead of regenerating them. Null otherwise.
    static const CsSampler*& draw_hint() {
        static thread_local const CsSampler* hint = nullptr;
        return hint;
    }
    void copy_luma_draw(const CsSampler& o) { ri_x = o.ri_x; ri_y = o.ri_y; }
    void copy_chroma_draw(const CsSampler& o) {
        ri_cx1 = o.ri_cx1; ri_cy1 = o.ri_cy1; ri_cx2 = o.ri_cx2; ri_cy2 = o.ri_cy2;
        m_chroma = o.m_chroma; chroma_support = o.chroma_support;
    }
};

#endif
