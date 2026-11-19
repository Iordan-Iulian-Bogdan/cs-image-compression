// Backend tests: encoder geometry.
#include "test_common.hpp"

// Measured 24.3..24.8 dB at ratio 1.0 (2026-11-09); the alignment check below is the real test.
static const double GEOMETRY_MIN_PSNR = 22.0;

// Encoder geometry: the container's samples must come from the image at the positions the
// decoder puts them. Sizes not divisible by 8 used to be stretched to a multiple of 8 for the
// sampling pixels while the sample positions stayed on the unstretched grid, shifting content
// by up to 4 px (5-9 dB lower decode PSNR on real photos).
void test_encoder_geometry() {
    // Sharp-edged content makes any shift cost PSNR.
    // 100x76 and 98x74 are not multiples of 8; 96x72 is the control
    for (const cv::Size sz : { cv::Size(100, 76), cv::Size(98, 74), cv::Size(96, 72) }) {
        cv::Mat img(sz, CV_8UC3, cv::Scalar(90, 100, 110));
        cv::rectangle(img, cv::Rect(sz.width / 5, sz.height / 5, sz.width / 2, sz.height / 2), cv::Scalar(30, 40, 230), cv::FILLED);
        cv::rectangle(img, cv::Rect(sz.width * 3 / 5, sz.height * 3 / 5, sz.width / 4, sz.height / 4), cv::Scalar(230, 220, 40), cv::FILLED);
        cv::circle(img, cv::Point(sz.width * 3 / 4, sz.height / 4), sz.height / 8, cv::Scalar(20, 230, 20), cv::FILLED);
        const std::string tmp_in = ".cs_test_geom_in.png", tmp_out = ".cs_test_geom_out.png";
        cv::imwrite(tmp_in, img);
        const std::string tmp_enc = ".cs_test_geom_in.cs";
        const int rc = prod_encode(tmp_in, tmp_enc, 1.0f);
        char name[96];
        std::snprintf(name, sizeof(name), "geometry %dx%d: encode", sz.width, sz.height);
        check(rc == 0, name);
        const int rd = rc == 0 ? prod_decode(read_container(tmp_enc), tmp_out) : -1;
        std::snprintf(name, sizeof(name), "geometry %dx%d: decode", sz.width, sz.height);
        check(rd == 0, name);
        cv::Mat dec = cv::imread(tmp_out, cv::IMREAD_COLOR);
        std::snprintf(name, sizeof(name), "geometry %dx%d: output has the input size", sz.width, sz.height);
        check(dec.size() == img.size(), name);
        if (dec.size() == img.size()) {
            // ratio 1.0: the decode is the input up to the half-res solve and the 2x cs-sr stage
            const double p = cs_quality::psnr(img, dec);
            // the decode must not be closer to the input shifted by 1..4 px than to the input itself
            double best_shift = 1e9, aligned = 1e9;
            for (int dy = -4; dy <= 4; ++dy)
                for (int dx = -4; dx <= 4; ++dx) {
                    const cv::Rect a(std::max(0, dx), std::max(0, dy), sz.width - std::abs(dx), sz.height - std::abs(dy));
                    const cv::Rect b(std::max(0, -dx), std::max(0, -dy), a.width, a.height);
                    const double e = cv::norm(img(a), dec(b), cv::NORM_L1) / ((double)a.area() * 3);
                    if (dx == 0 && dy == 0) aligned = e;
                    else best_shift = std::min(best_shift, e);
                }
            std::printf("       geometry %dx%d: PSNR %.2f dB, error aligned %.3f vs best shifted %.3f\n",
                sz.width, sz.height, p, aligned, best_shift);
            std::snprintf(name, sizeof(name), "geometry %dx%d: decode aligned with the input (PSNR > %.0f)", sz.width, sz.height, GEOMETRY_MIN_PSNR);
            check(p > GEOMETRY_MIN_PSNR, name);
            std::snprintf(name, sizeof(name), "geometry %dx%d: no shifted copy matches better", sz.width, sz.height);
            check(aligned <= best_shift, name);
        }
        std::remove(tmp_in.c_str());
        std::remove(tmp_enc.c_str());
        std::remove(tmp_out.c_str());
    }
}
