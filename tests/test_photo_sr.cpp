// Backend tests: HR anchors (apply, CS-SR data terms).
#include "test_common.hpp"

// ---------------------------------------------------------------------------
// HR anchors (mode 19): exact native luma applied to the decoded image.
// ---------------------------------------------------------------------------
void test_hr_detail() {
    // Apply: exact luma replacement at anchor positions, OOB skipped.
    // Neutral-gray canvas admits every Y (no gamut clipping).
    cv::Mat canvas(24, 32, CV_8UC3, cv::Scalar(100, 100, 100));
    const std::vector<CsHrSample> put = { {0, 200}, {100, 50}, {(uint32_t)(24 * 32 - 1), 128},
        {(uint32_t)(24 * 32 + 5), 200} };
    const size_t applied = cs_apply_hr_detail(canvas, put);
    check(applied == 3, "hr-detail: apply skips out-of-range");
    cv::Mat canvas_ycc;
    cv::cvtColor(canvas, canvas_ycc, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> canvas_chs;
    cv::split(canvas_ycc, canvas_chs);
    check(canvas_chs[0].at<uint8_t>(0, 0) == 200 &&
        canvas_chs[0].at<uint8_t>(100 / 32, 100 % 32) == 50 &&
        canvas_chs[0].at<uint8_t>(23, 31) == 128,
        "hr-detail: apply sets exact luma");
}

// ---------------------------------------------------------------------------
// HR anchors inside the CS-SR solve: exact-HR data terms pull
// anchor pixels AND their neighborhoods, beyond post-hoc replacement.
// ---------------------------------------------------------------------------
static double hr_anchor_mean_err(const cv::Mat& hr_bgr, const cv::Mat& truth_gray,
    const std::vector<CsHrSample>& anchors, int stride, bool centers_only) {
    cv::Mat ycc;
    cv::cvtColor(hr_bgr, ycc, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> chs;
    cv::split(ycc, chs);
    double sum = 0.0;
    int cnt = 0;
    const int W = chs[0].cols, H = chs[0].rows;
    for (const CsHrSample& s : anchors) {
        const int gx = (int)(s.idx % (uint32_t)stride);
        const int gy = (int)(s.idx / (uint32_t)stride);
        for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
                if (centers_only && (dx != 0 || dy != 0)) continue;
                if (!centers_only && dx == 0 && dy == 0) continue;
                const int x = gx + dx, y = gy + dy;
                if (x < 0 || y < 0 || x >= W || y >= H) continue;
                if (x >= truth_gray.cols || y >= truth_gray.rows) continue;
                sum += std::abs((int)chs[0].at<uint8_t>(y, x) -
                    (int)truth_gray.at<uint8_t>(y, x));
                ++cnt;
            }
        }
    }
    return cnt > 0 ? sum / cnt : 0.0;
}

void test_hr_sr_anchors() {
    // True 80x80 HR, 40x40 measured LR, biased solve, full-grid packed container;
    // the production solve (PnP-ADMM, 2 passes).
    const cv::Mat original = make_test_image(80, 80);
    cv::Mat measured, solved;
    cv::resize(original, measured, cv::Size(40, 40), 0, 0, cv::INTER_AREA);
    cv::add(measured, cv::Scalar(8, 8, 8), solved);
    const int m = measured.rows * measured.cols;
    cv::Mat packed(CS_HEADER_PIXELS + m, 1, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<int> rx, ry;
    rx.reserve(m); ry.reserve(m);
    for (int row = 0; row < measured.rows; ++row)
        for (int col = 0; col < measured.cols; ++col) {
            rx.push_back(row); ry.push_back(col);
            packed.at<cv::Vec3b>(CS_HEADER_PIXELS + row * measured.cols + col) =
                measured.at<cv::Vec3b>(row, col);
        }
    cv::Mat orig_gray;
    cv::cvtColor(original, orig_gray, cv::COLOR_BGR2GRAY);
    // True-HR anchors at spread positions, plus one out-of-tile anchor
    // (exercises the skip path).
    const int pts[8][2] = {{10,10},{30,12},{55,20},{70,33},{15,45},{40,60},{62,70},{25,25}};
    std::vector<CsHrSample> anchors;
    for (int k = 0; k < 8; ++k) {
        CsHrSample s;
        s.idx = (uint32_t)(pts[k][1] * 80 + pts[k][0]);
        s.y = orig_gray.at<uint8_t>(pts[k][1], pts[k][0]);
        anchors.push_back(s);
    }
    anchors.push_back({(uint32_t)(200 * 80 + 200), 128});
    CsHrSrCtx ctx;
    ctx.anchors = &anchors;
    ctx.stride = 80;
    ctx.ox = 0; ctx.oy = 0;

    cv::Mat plain, anchored;
    cs_sr_upscale_tile_2x(solved, packed, rx, ry, 0.018f,
        8, 0, 2.0f, 2, true, plain);
    cs_sr_upscale_tile_2x(solved, packed, rx, ry, 0.018f,
        8, 0, 2.0f, 2, true, anchored, ctx);
    check(anchored.size() == original.size() && anchored.type() == CV_8UC3,
        "hr-sr: anchored solve keeps 2x geometry and color type");
    // Empty anchors: bit-identical to the unanchored call.
    CsHrSrCtx empty;
    cv::Mat nohr;
    cs_sr_upscale_tile_2x(solved, packed, rx, ry, 0.018f,
        8, 0, 2.0f, 2, true, nohr, empty);
    check(cv::norm(nohr, plain, cv::NORM_INF) == 0,
        "hr-sr: empty anchors are bit-identical to plain");
    // Same tile-local mapping framed in a larger canvas (stride 160,
    // origin 40,40): bit-identical output proves origin/stride plumbing.
    std::vector<CsHrSample> reframed;
    for (const CsHrSample& s : anchors) {
        const int gx = (int)(s.idx % 80u), gy = (int)(s.idx / 80u);
        CsHrSample r;
        r.idx = (uint32_t)((gy + 40) * 160 + (gx + 40));
        r.y = s.y;
        reframed.push_back(r);
    }
    CsHrSrCtx ctx2;
    ctx2.anchors = &reframed;
    ctx2.stride = 160;
    ctx2.ox = 40; ctx2.oy = 40;
    cv::Mat refr;
    cs_sr_upscale_tile_2x(solved, packed, rx, ry, 0.018f,
        8, 0, 2.0f, 2, true, refr, ctx2);
    check(cv::norm(refr, anchored, cv::NORM_INF) == 0,
        "hr-sr: reframed anchors are bit-identical");
    // The actual claim: anchors AND their neighborhoods improve.
    if (!anchored.empty() && !plain.empty()) {
        const double plain_c = hr_anchor_mean_err(plain, orig_gray, anchors, 80, true);
        const double anchored_c = hr_anchor_mean_err(anchored, orig_gray, anchors, 80, true);
        const double plain_n = hr_anchor_mean_err(plain, orig_gray, anchors, 80, false);
        const double anchored_n = hr_anchor_mean_err(anchored, orig_gray, anchors, 80, false);
        std::printf("       hr-sr center err: %.3f -> %.3f, neighbor err: %.3f -> %.3f\n",
            plain_c, anchored_c, plain_n, anchored_n);
        check(anchored_c < plain_c, "hr-sr: anchor pixels improve in-solve");
        check(anchored_n < plain_n, "hr-sr: anchor neighborhoods improve in-solve");
    }
}
