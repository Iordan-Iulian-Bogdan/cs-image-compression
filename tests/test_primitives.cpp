// Backend tests: container header, shuffle, tiling, production roundtrips, sample bits.
#include "test_common.hpp"
#include "sample_coder.hpp"
#include <chrono>

// Regression floors for the roundtrips (measured 2026-11-09: 960x720 ratio 0.5 32.3 dB / SSIM 0.974).
static const double ROUNDTRIP_MIN_PSNR = 30.0, ROUNDTRIP_MIN_SSIM = 0.95;

// ---------------------------------------------------------------------------
// 1. container header (v4: plain metadata)
// ---------------------------------------------------------------------------
void test_header() {
    const std::string meta = cs_build_metadata(1234567, 2016, 1512, 3024, 4032);
    check(meta.size() == CS_METADATA_BYTES, "header: metadata is 32 bytes");
    check(cs_build_metadata(1234567, 2016, 1512, 3024, 4032) ==
        cs_build_metadata(1234567, 2016, 1512, 3024, 4032), "header: metadata build deterministic");

    std::vector<uint8_t> buf(CS_HEADER_BYTES + 900);
    check(cs_write_header(buf.data(), buf.size(), meta), "header: write");
    check(buf[CS_OFF_VERSION] == CS_VERSION, "header: version byte");
    CsHeaderInfo info;
    check(!cs_parse_header(buf.data(), buf.size(), info) && !info.legacy, "header: mode 0 is not a production format");
    buf[CS_OFF_PAD] = CS_MODE_YCC420_SEG;
    check(cs_parse_header(buf.data(), buf.size(), info) && info.sampling_mode == CS_MODE_YCC420_SEG &&
        info.m == 1234567 && info.rows == 2016 && info.cols == 1512 &&
        info.org_h == 3024 && info.org_w == 4032 && info.odd_parity == 0, "header: parse roundtrip");

    // odd-size parity: optional 6th field, absent for even images (text unchanged)
    check(cs_build_metadata(1234567, 2016, 1512, 3024, 4032, 0) == meta, "header: parity 0 keeps the 5-field metadata");
    for (int p = 1; p <= 3; ++p) {
        std::vector<uint8_t> pbuf(CS_HEADER_BYTES);
        cs_write_header(pbuf.data(), pbuf.size(), cs_build_metadata(1234567, 2016, 1512, 3024, 4032, p));
        pbuf[CS_OFF_PAD] = CS_MODE_YCC420_SEG;
        CsHeaderInfo pinfo;
        check(cs_parse_header(pbuf.data(), pbuf.size(), pinfo) && pinfo.odd_parity == p &&
            pinfo.org_h == 3024 && pinfo.org_w == 4032, "header: parity field roundtrips");
    }
    {
        std::vector<uint8_t> pbuf(CS_HEADER_BYTES);
        cs_write_header(pbuf.data(), pbuf.size(), cs_build_metadata(1234567, 2016, 1512, 3024, 4032, 4));
        pbuf[CS_OFF_PAD] = CS_MODE_YCC420_SEG;
        CsHeaderInfo pinfo;
        check(!cs_parse_header(pbuf.data(), pbuf.size(), pinfo), "header: parity > 3 rejected");
    }

    CsHeaderInfo bad;
    std::vector<uint8_t> legacy(96, (uint8_t)3);   // an older container's version byte
    check(!cs_parse_header(legacy.data(), legacy.size(), bad) && bad.legacy, "header: older container detected");

    bool threw = false;
    try { cs_build_metadata(999999999, 99999, 99999, 999999, 999999); }
    catch (const std::runtime_error&) { threw = true; }
    check(threw, "header: oversized metadata throws");

    // the container file is the raw bytes
    const std::string path = ".cs_test_container.bin";
    std::vector<uint8_t> back;
    check(cs_write_container_file(path, buf.data(), buf.size() - 1) && cs_read_container_file(path, back) &&
        back.size() == buf.size() - 1 && std::memcmp(back.data(), buf.data(), back.size()) == 0,
        "header: container file roundtrips byte-exact");
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// 2. shuffle determinism
// ---------------------------------------------------------------------------
void test_shuffle() {
    std::vector<int> orig(5000);
    std::iota(orig.begin(), orig.end(), 0);

    std::vector<int> a = orig, b = orig;
    shuffle(a, 12345u);
    shuffle(b, 12345u);
    check(a == b, "shuffle: same seed -> same permutation");
    check(a != orig, "shuffle: permutation actually applied");
    std::sort(a.begin(), a.end());
    check(a == orig, "shuffle: a permutation (every element kept)");
}

// ---------------------------------------------------------------------------
// 3. tile grid helpers
// ---------------------------------------------------------------------------
void test_tile_helpers() {
    // reconstructImage placement
    const int ts = 8, gap = 0;
    std::vector<std::vector<cv::Mat>> tiles(2, std::vector<cv::Mat>(2));
    std::vector<std::vector<TileCoord>> coords(2, std::vector<TileCoord>(2));
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            tiles[i][j] = cv::Mat(ts, ts, CV_8UC3, cv::Scalar(uchar(20 + 40 * j), uchar(30 + 40 * i), uchar(100 + i * j)));
            coords[i][j] = { j * (ts + gap), i * (ts + gap) };
        }
    }
    cv::Mat re = reconstructImage(tiles, coords);
    check(!re.empty() && re.cols == 2 * ts && re.rows == 2 * ts, "reconstructImage: output size");
    bool placed = true;
    for (int i = 0; i < 2 && placed; i++) {
        for (int j = 0; j < 2 && placed; j++) {
            const cv::Vec3b expect = tiles[i][j].at<cv::Vec3b>(0, 0);
            const cv::Vec3b got = re.at<cv::Vec3b>(coords[i][j].y, coords[i][j].x);
            placed = got == expect;
        }
    }
    check(placed, "reconstructImage: tiles placed at coordinates");

    // blend: alpha=1 -> tile wins
    cv::Mat target = cv::Mat(2 * ts, 2 * ts, CV_8UC3, cv::Scalar(10, 10, 10));
    cv::Mat blended = blendTilesWithImage(tiles, coords, target, 1.0f);
    check(blended.at<cv::Vec3b>(0, 0) == tiles[0][0].at<cv::Vec3b>(0, 0), "blend: alpha=1 takes tile value");

    // blend: two identical solid tiles stacked at same coordinate -> half of each channel
    std::vector<std::vector<cv::Mat>> t2(1, std::vector<cv::Mat>(1));
    std::vector<std::vector<TileCoord>> c2(1, std::vector<TileCoord>(1));
    t2[0][0] = cv::Mat(ts, ts, CV_8UC3, cv::Scalar(100, 100, 100));
    c2[0][0] = { 0, 0 };
    cv::Mat base = cv::Mat(ts, ts, CV_8UC3, cv::Scalar(20, 20, 20));
    cv::Mat out = blendTilesWithImage(t2, c2, base, 0.5f);
    const cv::Vec3b px = out.at<cv::Vec3b>(0, 0);
    check(std::abs((int)px[0] - 60) <= 1, "blend: alpha=0.5 averages overlapping tiles");

    // feather blend: a tile edge touching the image border has no neighbor,
    // so it carries full weight. A single solid tile + feather must come
    // back bit-exact; before the fix the outer pixel ring came out black
    // (cosine weight 0 with no contributor, 0/1e-6 after normalization).
    {
        const int W = 80, H = 64;
        cv::Mat solid(H, W, CV_8UC3, cv::Scalar(128, 64, 200));
        std::vector<std::vector<cv::Mat>> ft(1, std::vector<cv::Mat>(1, solid.clone()));
        std::vector<std::vector<TileCoord>> fc(1, std::vector<TileCoord>(1, { 0, 0 }));
        cv::Mat pasted = reconstructImage(ft, fc);
        cv::Mat fbl = blendTilesWithImage(ft, fc, pasted, 0.5f, 24);
        double mx = 0.0;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const cv::Vec3b a = solid.at<cv::Vec3b>(y, x), b = fbl.at<cv::Vec3b>(y, x);
                for (int c = 0; c < 3; ++c) mx = (std::max)(mx, (double)std::abs((int)a[c] - (int)b[c]));
            }
        check(mx == 0.0, "blend feather: single tile keeps border pixels (no black frame)");
    }
    // feather blend: 2x2 overlapping tiles from a solid image agree
    // everywhere (borders full weight, seams partition of unity).
    {
        const int W = 64, H = 64, ov = 16;
        cv::Mat solid(H, W, CV_8UC3, cv::Scalar(90, 160, 40));
        std::vector<std::vector<cv::Mat>> gt;
        std::vector<std::vector<TileCoord>> gc;
        splitImageIntoTiles(solid, gt, gc, 2, 2, ov);
        cv::Mat pasted = reconstructImage(gt, gc);
        cv::Mat fbl = blendTilesWithImage(gt, gc, pasted, 0.5f, ov);
        double mx = 0.0;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const cv::Vec3b a = solid.at<cv::Vec3b>(y, x), b = fbl.at<cv::Vec3b>(y, x);
                for (int c = 0; c < 3; ++c) mx = (std::max)(mx, (double)std::abs((int)a[c] - (int)b[c]));
            }
        check(mx <= 1.0, "blend feather: overlapping tiles agree on solid image");
    }
}

// ---------------------------------------------------------------------------
// 4. full roundtrip + PSNR + determinism
// ---------------------------------------------------------------------------
void test_roundtrip() {
    const cv::Mat original = make_test_image(960, 720);
    const std::string tmp_in = ".cs_test_input.png";
    const std::string tmp_enc = ".cs_test_input.cs";
    const std::string tmp_out = ".cs_test_output.png";

    check(prod_encode(".cs_test_missing.png", tmp_enc, 0.5f) != 0, "roundtrip: missing input file rejected");
    cv::imwrite(tmp_in, original);
    {
        const char* tag = "roundtrip";
        char name[96];
        std::snprintf(name, sizeof(name), "%s: encode", tag);
        check(prod_encode(tmp_in, tmp_enc, 0.5f) == 0, name);
        const cv::Mat enc = read_container(tmp_enc);
        std::snprintf(name, sizeof(name), "%s: header mode %d", tag, CS_MODE_MULTISCALE);
        check(!enc.empty() && enc.data[CS_OFF_PAD] == CS_MODE_MULTISCALE, name);

        // the encode is deterministic (fixed sample seed): a second encode writes the same bytes
        {
            const std::string tmp_enc2 = ".cs_test_input2.cs";
            const bool same = prod_encode(tmp_in, tmp_enc2, 0.5f) == 0 &&
                cv::norm(read_container(tmp_enc2), enc, cv::NORM_INF) == 0.0;
            std::remove(tmp_enc2.c_str());
            std::snprintf(name, sizeof(name), "%s: encode is deterministic", tag);
            check(same, name);
        }
        // a container of another version is rejected before the solve
        cv::Mat old = enc.clone();
        old.data[CS_OFF_VERSION] = 3;
        std::snprintf(name, sizeof(name), "%s: older container rejected", tag);
        check(prod_decode(old, tmp_out) == -2, name);

        std::snprintf(name, sizeof(name), "%s: decode succeeds", tag);
        check(prod_decode(enc, tmp_out) == 0, name);
        const cv::Mat dec = cv::imread(tmp_out, cv::IMREAD_COLOR);
        std::snprintf(name, sizeof(name), "%s: decodes at the native size", tag);
        check(dec.size() == original.size(), name);
        if (dec.size() == original.size()) {
        const double p = cs_quality::psnr(original, dec);
        const double s = cs_quality::ssim(original, dec);
        std::printf("       %s PSNR: %.2f dB, SSIM: %.4f\n", tag, p, s);
        // Regression guards, not quality benchmarks: a broken index pipeline (wrong sampling,
        // misaligned offsets) collapses to single-digit dB and SSIM well below 0.5.
        std::snprintf(name, sizeof(name), "%s: PSNR > %.0f dB", tag, ROUNDTRIP_MIN_PSNR);
        check(p > ROUNDTRIP_MIN_PSNR, name);
        std::snprintf(name, sizeof(name), "%s: SSIM > %.2f", tag, ROUNDTRIP_MIN_SSIM);
        if (s >= 0.0) check(s > ROUNDTRIP_MIN_SSIM, name);
        else check(false, "roundtrip: SSIM unavailable");
        }
    }
    std::remove(tmp_in.c_str());
    std::remove(tmp_enc.c_str());
    std::remove(tmp_out.c_str());
}

// ---------------------------------------------------------------------------
// 4a. odd-size roundtrip: the 2x pipeline keeps the native size and pixel grid
// ---------------------------------------------------------------------------
void test_odd_size_roundtrip() {
    const std::string tmp_in = ".cs_test_odd_in.png";
    const std::string tmp_enc = ".cs_test_odd_in.cs";
    const std::string tmp_out = ".cs_test_odd_out.png";
    // 241 / 321: a 1 px last tile row / column of the multi-scale plan (no LR pixel of any layer falls in it);
    // the last case also with --ms-coarse 1
    const cv::Mat full = make_test_image(321, 241);
    double p_even = 0.0;
    const int sizes[7][3] = { { 238, 318, 0 }, { 239, 318, 0 }, { 238, 319, 0 }, { 239, 319, 0 },
                              { 241, 320, 0 }, { 240, 321, 0 }, { 241, 321, 1 } };   // rows, cols, coarse
    for (const auto& hw : sizes) {
        const cv::Mat original = full(cv::Rect(0, 0, hw[1], hw[0])).clone();
        cv::imwrite(tmp_in, original);
        char tag[96];
        bool ok = prod_encode(tmp_in, tmp_enc, 0.5f, 2000, (float)hw[2]) == 0;
        std::snprintf(tag, sizeof(tag), "odd size: %dx%d encode%s", hw[0], hw[1], hw[2] ? " (--ms-coarse 1)" : "");
        check(ok, tag);
        if (!ok) continue;
        ok = prod_decode(read_container(tmp_enc), tmp_out) == 0;
        std::snprintf(tag, sizeof(tag), "odd size: %dx%d decode", hw[0], hw[1]);
        check(ok, tag);
        if (!ok) continue;
        const cv::Mat dec = cv::imread(tmp_out, cv::IMREAD_COLOR);
        std::snprintf(tag, sizeof(tag), "odd size: %dx%d decodes at the native size", hw[0], hw[1]);
        check(dec.size() == original.size(), tag);
        if (dec.size() != original.size()) continue;
        // pixel-aligned with the original (no resample): about as good as the even crop of the same image (the
        // production decode measures up to 1.3 dB lower on these small crops; a shifted grid costs several dB)
        const double p = cs_quality::psnr(original, dec);
        std::printf("       odd size %dx%d PSNR: %.2f dB\n", hw[0], hw[1], p);
        if (hw[0] % 2 == 0 && hw[1] % 2 == 0) p_even = p;
        std::snprintf(tag, sizeof(tag), "odd size: %dx%d PSNR within 2 dB of the even crop", hw[0], hw[1]);
        check(p > p_even - 2.0, tag);
    }
    std::remove(tmp_in.c_str());
    std::remove(tmp_enc.c_str());
    std::remove(tmp_out.c_str());
}

void test_sample_bits() {
    // helper unit checks: identity at 8, bounded error below
    check(cs_quantize_sample(0, 4) == 0 && cs_quantize_sample(255, 4) == 15,
        "sample-bits: quantize endpoints at 4 bits");
    check(cs_dequantize_sample(0, 4) == 0 && cs_dequantize_sample(15, 4) == 255,
        "sample-bits: dequantize endpoints at 4 bits");
    {
        bool ok = true;
        for (int v = 0; v < 256; ++v) {
            const uint8_t q = cs_quantize_sample((uint8_t)v, 5);
            const uint8_t r = cs_dequantize_sample(q, 5);
            if (std::abs((int)r - v) > 5) { ok = false; break; } // 5-bit step ~8.2
        }
        check(ok, "sample-bits: 5-bit roundtrip within half a step");
    }
    check(cs_packed_bytes(3, 8) == 3 && cs_packed_bytes(3, 4) == 2 &&
        cs_packed_bytes(8, 1) == 1 && cs_packed_bytes(0, 4) == 0,
        "sample-bits: packed size math");
    {
        // pack/unpack roundtrip: pack quantizes raw 8-bit values, unpack
        // inverts to the quantize->dequantize fixed points
        const uint8_t vals[8] = { 0, 17, 34, 128, 200, 255, 7, 99 };
        uint8_t packed[8], back[8];
        cs_pack_samples(vals, 8, 4, packed);
        cs_unpack_samples(packed, 8, 4, back);
        bool ok = true;
        for (int i = 0; i < 8; ++i) ok = ok && (back[i] == cs_dequantize_sample(cs_quantize_sample(vals[i], 4), 4));
        check(ok, "sample-bits: 4-bit pack/unpack roundtrip exact");
    }
}

// Sample coding: the plane coder is lossless on any mask, and a coded container carries exactly the samples of
// the bit-packed one written with the same seed.
void test_sample_coding() {
    std::mt19937 rng(7);
    struct Case { int rows, cols, bits; double dens; int kind; const char* tag; };   // kind 0 smooth, 1 noise, 2 flat
    const Case cases[] = {
        { 300, 400, 6, 0.3, 0, "6-bit smooth, 30%" }, { 240, 320, 8, 1.0, 0, "8-bit smooth, full" },
        { 150, 170, 6, 0.02, 0, "6-bit smooth, 2%" }, { 80, 96, 8, 0.5, 1, "8-bit noise, 50%" },
        { 80, 96, 6, 1.0, 2, "6-bit flat, full" }, { 33, 9, 1, 0.5, 1, "1-bit noise" },
        { 40, 40, 8, 0.0, 0, "empty mask" }, { 5, 3, 6, 1.0, 0, "5x3 plane" } };
    for (const Case& c : cases) {
        const size_t N = (size_t)c.rows * c.cols;
        std::vector<uint8_t> val(N), mask(N), out(N, 0xEE);
        const int maxv = (1 << c.bits) - 1;
        size_t n = 0;
        for (int y = 0; y < c.rows; ++y)
            for (int x = 0; x < c.cols; ++x) {
                const size_t j = (size_t)y * c.cols + x;
                const double smooth = 0.5 + 0.35 * std::sin(x * 0.11) * std::cos(y * 0.07) + 0.02 * ((int)(rng() % 5) - 2);
                val[j] = c.kind == 1 ? (uint8_t)(rng() & maxv) : c.kind == 2 ? (uint8_t)(maxv / 3)
                    : (uint8_t)std::clamp((int)std::lround(smooth * maxv), 0, maxv);
                mask[j] = (rng() % 10000) < c.dens * 10000;
                n += mask[j];
            }
        const std::vector<uint8_t> enc = cs_plane_encode(val.data(), mask.data(), c.rows, c.cols, c.bits);
        bool ok = cs_plane_decode(enc.data(), enc.size(), mask.data(), c.rows, c.cols, c.bits, out.data());
        for (size_t j = 0; ok && j < N; ++j) ok = out[j] == (mask[j] ? val[j] : 0xEE);
        char name[96];
        std::snprintf(name, sizeof(name), "sample-coding %s: roundtrip exact, unsampled untouched", c.tag);
        check(ok, name);
        const size_t packed = n ? (n * (size_t)c.bits + 7) / 8 : 0;
        std::snprintf(name, sizeof(name), "sample-coding %s: %zu bytes, bit-packed %zu", c.tag, enc.size(), packed);
        check(c.kind == 0 && n > 10000 ? enc.size() < packed * 3 / 4 : enc.size() <= packed + 1, name);
        if (n > 0 && enc[0] == 0) {
            std::vector<uint8_t> cut(enc.begin(), enc.begin() + 1);
            std::snprintf(name, sizeof(name), "sample-coding %s: truncated tables rejected", c.tag);
            check(!cs_plane_decode(cut.data(), cut.size(), mask.data(), c.rows, c.cols, c.bits, out.data()), name);
        }
    }

    // One seed, the same layer written bit-packed and coded: identical sampled planes and masks.
    const cv::Mat photo = make_test_image(480, 360);
    cv::Mat lr;
    cv::resize(photo, lr, cv::Size(photo.cols / 2, photo.rows / 2));
    cv::Mat ids(lr.rows, lr.cols, CV_8U, cv::Scalar(0));
    ids(cv::Rect(0, 0, lr.cols / 2, lr.rows)).setTo(1);
    const auto same_key = [&](const cv::Mat& img, const cv::Mat& labels, const std::vector<float>& rates, const char* tag) {
        using clk = std::chrono::steady_clock;
        const auto ms = [](clk::time_point t0, clk::time_point t1) { return std::chrono::duration<double, std::milli>(t1 - t0).count(); };
        encode_image e(img, true);
        const auto t0 = clk::now();
        e.encode_ycc420_segmask_ids(labels, rates, nullptr, false);
        const auto t1 = clk::now();
        const cv::Mat packed = e.get_mat().clone();
        const auto t2 = clk::now();
        e.encode_ycc420_segmask_ids(labels, rates, nullptr, true, true);
        const auto t3 = clk::now();
        const cv::Mat coded = e.get_mat().clone();
        cv::Mat a[6], b[6];
        bool same = false;
        double da_ms = 0, db_ms = 0;
        try {
            decode_image da(packed), db(coded);
            const auto u0 = clk::now();
            da.get_sampled_ycc420(a[0], a[1], a[2], a[3], a[4], a[5]);
            const auto u1 = clk::now();
            db.get_sampled_ycc420(b[0], b[1], b[2], b[3], b[4], b[5]);
            const auto u2 = clk::now();
            da_ms = ms(u0, u1); db_ms = ms(u1, u2);
            same = cv::countNonZero(a[1]) > 0;
            for (int i = 0; i < 6; ++i) same = same && a[i].size() == b[i].size() && cv::norm(a[i], b[i], cv::NORM_INF) == 0;
        }
        catch (const std::exception& ex) { std::printf("       %s\n", ex.what()); }
        char name[160];
        std::snprintf(name, sizeof(name), "sample-coding %s: coded container yields the bit-packed container's samples", tag);
        check(same, name);
        std::printf("       sample-coding %s: %d luma samples, packed %zu bytes, coded %zu bytes (%.3f); layer encode %.0f / %.0f ms, "
            "sample read %.0f / %.0f ms\n", tag, cv::countNonZero(a[1]), packed.total() * 3, coded.total() * 3,
            (double)coded.total() / packed.total(), ms(t0, t1), ms(t2, t3), da_ms, db_ms);
        std::snprintf(name, sizeof(name), "sample-coding %s: coded container is smaller", tag);
        check(coded.total() < packed.total(), name);
    };
    same_key(lr, ids, { 0.7f, 0.25f }, "layer");
}
