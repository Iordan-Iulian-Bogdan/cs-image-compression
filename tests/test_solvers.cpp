// Backend tests: DCT transforms, threshold weights, auto-reg.
#include "test_common.hpp"

void test_dct_adjoint() {
    // The FISTA gradient assumes S^T = forward DCT for S = inverse DCT.
    // If cv::dct is not orthonormal, the solve fixed point drifts off the
    // data even at full sampling. Check S^T S = I and <S a, b> = <a, S^T b>.
    // Same operations as cs_dct_inverse and cs_dct_forward.
    std::mt19937 rng(42);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    cv::Mat a(16, 24, CV_32F), bvec(16, 24, CV_32F);
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 24; ++j) { a.at<float>(i, j) = gauss(rng); bvec.at<float>(i, j) = gauss(rng); }
    cv::Mat sa, sadj_b, rtv;
    cv::dct(a, sa, cv::DCT_INVERSE);
    cv::dct(bvec, sadj_b, 0);
    double lhs = 0.0, rhs = 0.0, rt = 0.0;
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 24; ++j) {
            lhs += (double)sa.at<float>(i, j) * bvec.at<float>(i, j);
            rhs += (double)a.at<float>(i, j) * sadj_b.at<float>(i, j);
        }
    cv::Mat s1, s2;
    cv::dct(a, s1, cv::DCT_INVERSE);
    cv::dct(s1, s2, 0);
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 24; ++j) rt = std::max(rt, (double)std::fabs(s2.at<float>(i, j) - a.at<float>(i, j)));
    std::printf("       dct adjoint: <Sa,b>=%.6f <a,S^Tb>=%.6f roundtrip max err=%.2e\n", lhs, rhs, rt);
    check(std::fabs(lhs - rhs) < 1e-3 * (1.0 + std::fabs(lhs)), "dct: forward DCT is adjoint of inverse");
    check(rt < 1e-3, "dct: adjoint of synthesis inverts synthesis (S^T S = I)");
}

void test_dct_weights() {
    // frequency ramp: DC x1, peak AC x wscale, uniform at wscale <= 1
    cs_dct_basis bu = cs_make_dct_basis(64, 48, 1.0f);
    bool ones = true;
    for (float v : bu.wscale) ones = ones && (v == 1.0f);
    check(ones, "dct-ramp: wscale=1 keeps uniform weights");
    cs_dct_basis bw = cs_make_dct_basis(64, 48, 2.0f);
    check(bw.lip == 2.0f, "dct-ramp: Lipschitz constant 2");
    check(bw.wscale[0] == 1.0f, "dct-ramp: DC protected (x1)");
    const float peak = bw.wscale[(size_t)63 * 48 + 47];
    check(std::fabs(peak - 2.0f) < 1e-5f, "dct-ramp: peak AC carries x wscale");
    bool mono = true;
    for (int i = 1; i < 64; ++i) mono = mono && (bw.wscale[(size_t)i * 48] >= bw.wscale[(size_t)(i - 1) * 48]);
    check(mono, "dct-ramp: weights rise monotonically with frequency");
}

void test_auto_reg_params() {
    // The luma coef law: 0.01125/r, times the tile factor (n/64)^0.15 in [0.5, 1.2] below ratio 0.5,
    // ratio clamped to [0.05, 1], coef to [0.01, 0.3].
    const auto near = [](float a, double b) { return std::fabs(a - b) < 1e-5; };
    check(near(cs_auto_reg_coefficient(0.3f, 64), 0.0375), "auto-reg: 0.3 t64 == 0.01125/r");
    check(near(cs_auto_reg_coefficient(0.75f, 64), 0.015), "auto-reg: 0.75 t64 == 0.01125/r");
    check(near(cs_auto_reg_coefficient(0.3f, 1), 0.0200958), "auto-reg: tile factor below 0.5");
    check(near(cs_auto_reg_coefficient(0.75f, 1), 0.015), "auto-reg: no tile factor at 0.5 and up");
    check(near(cs_auto_reg_coefficient(1.0f, 64), 0.01125), "auto-reg: full sampling");
    check(near(cs_auto_reg_coefficient(0.01f, 4096), 0.27), "auto-reg: ratio clamped to 0.05, tile factor to 1.2");
}
