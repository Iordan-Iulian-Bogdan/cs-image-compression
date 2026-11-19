#ifndef DECODE_IMAGE_HPP
#define DECODE_IMAGE_HPP

#include "cs_sampler.hpp"
#include "photo_upscaler.hpp"

#include <functional>
#include <vector>

class decode_image : CsSampler {
private:
    /** @brief header parsing.
    Throws std::runtime_error on containers of another version or a corrupt
    header. On success fills m/rows/cols/org_size and the sampling fields.
    */
    void parse_container_header(const cv::Mat& input);

    // Sample depths restored from the header
    int sample_bits = 8;
    int sample_bits_chroma = 8;

    // Restores the raw measurement layout from a bit-packed container (the
    // encoder's in-memory anchor pass): each section unpacks independently into a fresh raw-size Mat
    // that replaces the packed container, so every downstream reader runs
    // unchanged. Sections are (raw byte offset, value count) in ascending
    // order. No-op at 8 bits.
    void unpack_container_measurements(const std::vector<cs_body_section>& sections);
    // The same for an entropy-coded body (CS_OFF_SAMPLE_CODING, every written container): opens and decodes the three
    // planes and lays their dequantized values out in draw order, followed by the anchor payload.
    void decode_coded_measurements(size_t y0, int mY, int mC, int crows, int ccols);

    // Decodes + parses the seg-mask body prefix, caches seg_rates_q/seg_ids and
    // fills ri_x/ri_y.
    CsSegPrefix open_segmask_prefix(const std::function<void()>& after_labels = {}, const CsSampler* draws = nullptr);

    // Shipped per-class quantized rates of the seg container; cached during
    // sampling so the per-segment decode can scale each tile's coef by its
    // class's sampled rate and break mixed-tile votes toward the higher one.
    std::vector<uint8_t> seg_rates_q;

    // Solve-geometry label map of the seg container; cached during sampling.
    cv::Mat seg_ids;

public:
    decode_image() {}

    decode_image(cv::Mat input);

    // A container file as a 1 x N BGR Mat (its bytes zero-padded to whole pixels); empty if unreadable.
    static cv::Mat read_container(const std::string& path);

    float get_compression_ratio();

    cv::Size get_org_size();

    // Crops the 2x output back to the native size when the encoder padded an odd side
    // (cs_build_metadata parity). No-op for even images.
    void crop_odd_parity(cv::Mat& img) const {
        if (odd_parity == 0 || img.rows < 2 || img.cols < 2) return;
        img = img(cv::Rect(0, 0, img.cols - ((odd_parity >> 1) & 1), img.rows - (odd_parity & 1))).clone();
    }

    int get_sampling_mode() const { return sampling_mode; }
    const std::vector<uint8_t>& get_seg_rates_q() const { return seg_rates_q; }
    const cv::Mat& get_seg_ids() const { return seg_ids; }
    // Rectangular tile grid chosen in auto mode (rows x cols, 0 = the square num_tiles grid).
    int grid_rows = 0, grid_cols = 0;
    int get_sample_bits() const { return sample_bits; }
    int get_sample_bits_chroma() const { return sample_bits_chroma; }

    /** @brief unpacks a production container (seg luma, 4:2:0 chroma; modes 11 and 19) into
    per-plane sampled images + binary masks: luma at full resolution (CV_8U), each
    chroma plane on its half-resolution grid (CV_8U). Masks hold 0/1.
    Regenerates the identical index sets from (seed, m, geometry, label map);
    mode 19 also reads its HR anchors into hr_samples.
    */
    void get_sampled_ycc420(cv::Mat& y, cv::Mat& y_mask,
        cv::Mat& cr, cv::Mat& cr_mask, cv::Mat& cb, cv::Mat& cb_mask);

    /** @brief decodes a container (mode 20 = its layers composited; a layer, mode 11 or 19) with
    automatic parameters (iterations, tile grid, threads, coef) and writes the result to
    output_path, or to decoded_out when given (in memory, padded geometry).
    @param show_preview : live tile preview window
    @param fista_iters : FISTA steps per pass, 0 = auto-map from the iterations (the encoder's anchor pass: 12)
    @param photo_up : the tile overlap and the cs-sr stage's options
    */
    static int decode_image_tiled(const cv::Mat& encoded_img_in, const std::string& output_path, bool show_preview, int fista_iters, const CsPhotoUpscalerOptions& photo_up, cv::Mat* decoded_out = nullptr);

    static int decode_image_tiled(const std::string& input_path, const std::string& output_path, bool show_preview, int fista_iters, const CsPhotoUpscalerOptions& photo_up);

    /** @brief the tile pipeline of decode_image_tiled. dimgs is the already-parsed container
    (seed cached); remaining args are the resolved parameters. Solves luma at full tile resolution
    and each chroma plane natively on its half-resolution grid, merges YCrCb -> BGR per tile,
    upscales every tile 2x with cs-sr, composites, then applies the anchors.
    */
    static int decode_image_tiled_ycc420(decode_image& dimgs, const std::string& output_path, int num_tiles, int overlap, int iterations, int num_threads, float coef, bool show_preview, int fista_iters, const CsPhotoUpscalerOptions& photo_up, cv::Mat* decoded_out = nullptr);

};

#endif
