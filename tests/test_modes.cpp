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

void test_chroma_draw_v2() {
    // Seg prefix version 5: chroma draw v2 (a count floor, stratified on coarse layers); version 2 keeps the old draw.
    const auto parse = [](uint8_t v, CsSegPrefix& p) {
        const uint8_t body[12] = { 0, v, 0, 1, 128, 3, 0, 0, 0, 1, 2, 3 };
        return cs_segmask_parse_prefix(body, sizeof body, p);
    };
    CsSegPrefix p;
    check(parse(5, p) && p.chroma_draw_v2, "chroma v2: version 5 uses chroma draw v2");
    check(parse(2, p) && !p.chroma_draw_v2, "chroma v2: version 2 keeps the old draw");
    check(!parse(3, p) && !parse(6, p), "chroma v2: other versions rejected");

    // The draw on a full 64 x 64 chroma support (128 x 128 luma): 256 luma samples give 64 chroma samples (1.6%).
    struct Draw : CsSampler {
        std::vector<int> run(int scale, bool v2, int& count) {
            chroma_support = cv::Mat(64, 64, CV_8U, cv::Scalar(255));
            layer_scale = scale;
            chroma_draw_v2 = v2;
            returnYcc420ChromaIndices(128, 128, 256, cs_sample_seed);
            count = m_chroma;
            std::vector<int> per(64, 0);   // samples per aligned 8 x 8 block (Cr)
            for (size_t k = 0; k < ri_cx1.size(); ++k) ++per[(size_t)(ri_cx1[k] / 8 * 8 + ri_cy1[k] / 8)];
            return per;
        }
    } d;
    int n_old = 0, n2 = 0, n4 = 0;
    const std::vector<int> old = d.run(4, false, n_old), s2 = d.run(2, true, n2), s4 = d.run(4, true, n4);
    check(n_old == 64, "chroma v2: the old draw keeps the luma-derived count");
    check(n2 == 164 && n4 == 164, "chroma v2: count raised to 4% of the support");
    check(*std::min_element(s4.begin(), s4.end()) >= 2, "chroma v2: a coarse layer leaves no 8x8 block below 2 samples");
    check(*std::min_element(old.begin(), old.end()) == 0, "chroma v2: (the old random draw leaves empty blocks)");
    int again = 0;
    check(d.run(4, true, again) == s4, "chroma v2: deterministic");

    // Encode a layer (version 5), coded and bit-packed with anchors below the floor; both decode.
    cv::Mat img = make_test_image(128, 96);
    cv::Mat ids(96, 128, CV_8U, cv::Scalar(0));
    encode_image enc(img, true);
    enc.encode_ycc420_segmask_ids(ids, { 0.05f });
    const cv::Mat c = enc.get_mat();
    check(c.data[CS_HEADER_BYTES] == 0 && c.data[CS_HEADER_BYTES + 1] == 5, "chroma v2: a layer writes version 5");
    cv::Mat y, ym, cr, crm, cb, cbm;
    decode_image dec(c);
    dec.get_sampled_ycc420(y, ym, cr, crm, cb, cbm);
    cv::Mat ycc, cr_ref;
    cv::cvtColor(img, ycc, cv::COLOR_BGR2YCrCb);
    cv::extractChannel(ycc, cr_ref, 1);
    cv::resize(cr_ref, cr_ref, cr.size(), 0, 0, cv::INTER_AREA);
    bool match = cv::countNonZero(crm) >= (int)std::ceil(0.04 * crm.total());
    for (int r = 0; r < crm.rows && match; ++r)
        for (int col = 0; col < crm.cols; ++col)
            if (crm.at<uint8_t>(r, col) && cr.at<uint8_t>(r, col) != cr_ref.at<uint8_t>(r, col)) { match = false; break; }
    check(match, "chroma v2: container roundtrip returns the encoded Cr samples, at least 4% of them");
    CsAnchorPlan ap;
    cv::Mat big;
    cv::resize(img, big, cv::Size(256, 192));
    cv::cvtColor(big, ycc, cv::COLOR_BGR2YCrCb);
    cv::extractChannel(ycc, ap.luma, 0);
    ap.flags = cv::Mat(12, 16, CV_8U, cv::Scalar(1));
    ap.count = 500;
    encode_image enc2(img, true);
    enc2.encode_ycc420_segmask_ids(ids, { 0.02f }, &ap, false);
    bool ok = true;
    try {
        decode_image dec2(enc2.get_mat());
        dec2.get_sampled_ycc420(y, ym, cr, crm, cb, cbm);
        ok = cv::countNonZero(crm) >= (int)std::ceil(0.04 * crm.total());
    } catch (const std::exception&) { ok = false; }
    check(ok, "chroma v2: bit-packed anchored layer below the floor decodes its anchors");
}
