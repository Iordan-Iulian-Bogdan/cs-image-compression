// Backend tests: container header, shuffle, tiling, production roundtrips, sample bits.
#include "test_common.hpp"
#include "sample_coder.hpp"
#include <chrono>

// Regression floors for the production roundtrips (measured 2026-11-09 on the GPU build: 960x720 ratio 0.5
// 32.3 dB / SSIM 0.974 single- and multi-scale; 480x360 sample-bits 27.1..29.4 dB).
static const double ROUNDTRIP_MIN_PSNR = 30.0, ROUNDTRIP_MIN_SSIM = 0.95, SAMPLE_BITS_MIN_PSNR = 25.0;

// ---------------------------------------------------------------------------
// 1. container header (v4: plain metadata, no password)
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
    check(cs_parse_header(buf.data(), buf.size(), info) && info.version == CS_VERSION &&
        info.m == 1234567 && info.rows == 2016 && info.cols == 1512 &&
        info.org_h == 3024 && info.org_w == 4032 && info.odd_parity == 0, "header: parse roundtrip");

    // odd-size parity: optional 6th field, absent for even images (text unchanged)
    check(cs_build_metadata(1234567, 2016, 1512, 3024, 4032, 0) == meta, "header: parity 0 keeps the 5-field metadata");
    for (int p = 1; p <= 3; ++p) {
        std::vector<uint8_t> pbuf(CS_HEADER_BYTES);
        cs_write_header(pbuf.data(), pbuf.size(), cs_build_metadata(1234567, 2016, 1512, 3024, 4032, p));
        CsHeaderInfo pinfo;
        check(cs_parse_header(pbuf.data(), pbuf.size(), pinfo) && pinfo.odd_parity == p &&
            pinfo.org_h == 3024 && pinfo.org_w == 4032, "header: parity field roundtrips");
    }
    {
        std::vector<uint8_t> pbuf(CS_HEADER_BYTES);
        cs_write_header(pbuf.data(), pbuf.size(), cs_build_metadata(1234567, 2016, 1512, 3024, 4032, 4));
        CsHeaderInfo pinfo;
        check(!cs_parse_header(pbuf.data(), pbuf.size(), pinfo), "header: parity > 3 rejected");
    }

    CsHeaderInfo bad;
    std::vector<uint8_t> legacy(96, (uint8_t)3);   // a v3 (encrypted) container's version byte
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
// 2. shuffle determinism / reversal
// ---------------------------------------------------------------------------
void test_shuffle() {
    std::vector<int> orig(5000);
    std::iota(orig.begin(), orig.end(), 0);

    std::vector<int> a = orig, b = orig;
    shuffle(a, 12345u);
    shuffle(b, 12345u);
    check(a == b, "shuffle: same seed -> same permutation");
    check(a != orig, "shuffle: permutation actually applied");

    reverseShuffle(a, 12345u);
    check(a == orig, "shuffle: reverseShuffle restores original order");
}

// ---------------------------------------------------------------------------
// 3. tile grid helpers
// ---------------------------------------------------------------------------
void test_tile_helpers() {
    const std::vector<std::string> order = spiralOrder(4);
    check(order.size() == 16, "spiralOrder: 16 entries for 4x4 grid");
    std::set<std::string> uniq(order.begin(), order.end());
    check(uniq.size() == 16, "spiralOrder: all entries unique");

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
        splitImageIntoTiles(solid, gt, gc, 2, ov);
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
// 4. full production roundtrip (single- and multi-scale) + PSNR + determinism
// ---------------------------------------------------------------------------
void test_roundtrip() {
    const cv::Mat original = make_test_image(960, 720);
    const std::string tmp_in = ".cs_test_input.png";
    const std::string tmp_enc = ".cs_test_input.cs";
    const std::string tmp_out = ".cs_test_output.png";

    check(prod_encrypt(".cs_test_missing.png", tmp_enc, 0.5f) != 0, "roundtrip: missing input file rejected");
    cv::imwrite(tmp_in, original);
    for (const bool ms : { false, true }) {
        const char* tag = ms ? "roundtrip multi-scale" : "roundtrip single-scale";
        char name[96];
        std::snprintf(name, sizeof(name), "%s: encrypt", tag);
        check(prod_encrypt(tmp_in, tmp_enc, 0.5f, ms) == 0, name);
        const cv::Mat enc = read_container(tmp_enc);
        const int want_mode = ms ? CS_MODE_MULTISCALE : CS_MODE_YCC420_SEG_HRDETAIL;
        std::snprintf(name, sizeof(name), "%s: header mode %d", tag, want_mode);
        check(!enc.empty() && enc.data[CS_OFF_PAD] == want_mode, name);

        // the encode is deterministic (fixed sample seed): a second encode writes the same bytes
        {
            const std::string tmp_enc2 = ".cs_test_input2.cs";
            const bool same = prod_encrypt(tmp_in, tmp_enc2, 0.5f, ms) == 0 &&
                cv::norm(read_container(tmp_enc2), enc, cv::NORM_INF) == 0.0;
            std::remove(tmp_enc2.c_str());
            std::snprintf(name, sizeof(name), "%s: encode is deterministic", tag);
            check(same, name);
        }
        // a container of another version is rejected before the solve (-2 single-scale; a multi-scale one with
        // another version byte is no longer recognised as multi-scale and fails the same way)
        cv::Mat old = enc.clone();
        old.data[CS_OFF_VERSION] = 3;
        std::snprintf(name, sizeof(name), "%s: older container rejected", tag);
        check(prod_decrypt(old, tmp_out) == -2, name);

        std::snprintf(name, sizeof(name), "%s: decrypt succeeds", tag);
        check(prod_decrypt(enc, tmp_out) == 0, name);
        const cv::Mat dec = cv::imread(tmp_out, cv::IMREAD_COLOR);
        std::snprintf(name, sizeof(name), "%s: decodes at the native size", tag);
        check(dec.size() == original.size(), name);
        if (dec.size() != original.size()) continue;
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
    const cv::Mat full = make_test_image(321, 241);
    double p_even = 0.0;
    const int sizes[4][2] = { { 240, 320 }, { 241, 320 }, { 240, 321 }, { 241, 321 } };   // rows, cols
    for (const auto& hw : sizes) {
        const cv::Mat original = full(cv::Rect(0, 0, hw[1], hw[0])).clone();
        cv::imwrite(tmp_in, original);
        char tag[96];
        bool ok = prod_encrypt(tmp_in, tmp_enc, 0.5f) == 0;
        std::snprintf(tag, sizeof(tag), "odd size: %dx%d encrypt", hw[0], hw[1]);
        check(ok, tag);
        if (!ok) continue;
        ok = prod_decrypt(read_container(tmp_enc), tmp_out) == 0;
        std::snprintf(tag, sizeof(tag), "odd size: %dx%d decrypt", hw[0], hw[1]);
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
    check(cs_packed_bgr_bytes(4, 6, 4) == 7,
        "sample-bits: BGR split-depth size math");
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
    {
        const uint8_t vals[12] = { 2, 70, 245, 22, 110, 230, 42, 150, 210, 62, 190, 180 };
        uint8_t packed[16] = {}, back[12] = {};
        cs_pack_samples_bgr(vals, 4, 6, 4, packed);
        cs_unpack_samples_bgr(packed, 4, 6, 4, back);
        bool ok = true;
        for (int i = 0; i < 12; ++i) {
            const int bits = (i % 3 == 1) ? 6 : 4;
            ok = ok && back[i] == cs_dequantize_sample(cs_quantize_sample(vals[i], bits), bits);
        }
        check(ok, "sample-bits: interleaved BGR split-depth roundtrip");
    }

    const std::string tmp_in = ".cs_test_sb_in.png";
    const std::string tmp_enc = ".cs_test_sb_in.cs";
    const std::string tmp_out = ".cs_test_sb_out.png";
    const cv::Mat original = make_test_image(480, 360);
    cv::imwrite(tmp_in, original);

    // encode at the given depths (0 = the default 6-bit luma / 8-bit chroma), check the header, decode, score
    const auto run = [&](int bits, int chroma_bits, const char* tag, int want_y, int want_c, cv::Mat& enc) {
        char name[96];
        std::snprintf(name, sizeof(name), "sample-bits %s: encrypt", tag);
        const bool ok = prod_encrypt(tmp_in, tmp_enc, 0.5f, false, 2000, bits, chroma_bits) == 0;
        check(ok, name);
        if (!ok) return -1.0;
        enc = read_container(tmp_enc);
        {
            decrypt_image probe(enc);
            std::snprintf(name, sizeof(name), "sample-bits %s: header reads back %d/%d", tag, want_y, want_c);
            check(probe.get_sample_bits() == want_y && probe.get_sample_bits_chroma() == want_c, name);
        }
        std::snprintf(name, sizeof(name), "sample-bits %s: decrypt", tag);
        check(prod_decrypt(enc, tmp_out) == 0, name);
        const cv::Mat dec = cv::imread(tmp_out, cv::IMREAD_COLOR);
        return dec.size() == original.size() ? cs_quality::psnr(original, dec) : -1.0;
    };
    cv::Mat enc8, enc4, enc_def, enc64;
    const double p8 = run(8, 0, "8-bit", 8, 8, enc8);
    const double p4 = run(4, 0, "4-bit", 4, 4, enc4);
    const double pd = run(0, 0, "default", 6, 8, enc_def);
    const double p64 = run(6, 4, "split 6/4", 6, 4, enc64);
    std::printf("       sample-bits PSNR 8-bit: %.2f dB, 4-bit: %.2f dB, default 6/8: %.2f dB, 6/4: %.2f dB\n",
        p8, p4, pd, p64);
    check(enc4.total() < enc8.total(), "sample-bits: 4-bit container is smaller");
    // Regression guards, not quality benchmarks: plumbing breakage (wrong offsets, misaligned sections)
    // collapses to single-digit dB.
    check(std::isfinite(p8) && p8 > SAMPLE_BITS_MIN_PSNR, "sample-bits: 8-bit PSNR sane");
    check(std::isfinite(p4) && p4 > p8 - 10.0, "sample-bits: 4-bit PSNR within its quantization budget");
    check(std::isfinite(pd) && pd > SAMPLE_BITS_MIN_PSNR, "sample-bits: default 6-bit PSNR sane");
    check(std::isfinite(p64) && p64 > SAMPLE_BITS_MIN_PSNR, "sample-bits: split-depth PSNR sane");

    // Out-of-range depths are rejected, not silently ignored.
    check(prod_encrypt(tmp_in, tmp_enc, 0.5f, false, 2000, 9) == -1, "sample-bits: out-of-range depth rejected");

    std::remove(tmp_in.c_str());
    std::remove(tmp_enc.c_str());
    std::remove(tmp_out.c_str());
}

// --sample-coding: the plane coder is lossless on any mask, and a coded container carries exactly the samples of
// the bit-packed one written with the same key.
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

    // One key, the same layer written bit-packed and coded: identical sampled planes and masks.
    const cv::Mat photo = make_test_image(480, 360);
    cv::Mat lr;
    cv::resize(photo, lr, cv::Size(photo.cols / 2, photo.rows / 2));
    cv::Mat ids(lr.rows, lr.cols, CV_8U, cv::Scalar(0));
    ids(cv::Rect(0, 0, lr.cols / 2, lr.rows)).setTo(1);
    const auto same_key = [&](const cv::Mat& img, const cv::Mat& labels, const std::vector<float>& rates, const char* tag) {
        using clk = std::chrono::steady_clock;
        const auto ms = [](clk::time_point t0, clk::time_point t1) { return std::chrono::duration<double, std::milli>(t1 - t0).count(); };
        encrypt_image e(img, true);
        const auto t0 = clk::now();
        e.encrypt_ycc420_segmask_ids(1.0f, labels, rates, 6, 8, nullptr, false, false, false);
        const auto t1 = clk::now();
        const cv::Mat packed = e.get_mat().clone();
        const auto t2 = clk::now();
        e.encrypt_ycc420_segmask_ids(1.0f, labels, rates, 6, 8, nullptr, false, false, true);
        const auto t3 = clk::now();
        const cv::Mat coded = e.get_mat().clone();
        cv::Mat a[6], b[6];
        bool same = false;
        double da_ms = 0, db_ms = 0;
        try {
            decrypt_image da(packed), db(coded);
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
    // CS_TEST_PHOTO=<image>: the same check at photo scale with the production segmentation, sparse and full.
    if (const char* path = std::getenv("CS_TEST_PHOTO")) {
        const cv::Mat big = cv::imread(path, cv::IMREAD_COLOR);
        check(!big.empty(), "sample-coding: CS_TEST_PHOTO loads");
        if (!big.empty()) {
            cv::Mat half;
            cv::resize(big, half, cv::Size(big.cols / 2, big.rows / 2));
            for (const float r : { 0.2f, 0.6f, 1.0f }) {
                cv::Mat labels;
                cs_seg_auto_ids(half, 8, half.rows, half.cols, labels);
                std::vector<float> rates = cs_seg_auto_rates(half, labels, 8);
                rates.resize(8, 0.0f);
                cs_seg_tile_refine(half, labels, rates, 8, r);
                char tag[48];
                std::snprintf(tag, sizeof(tag), "photo, sample ratio %.1f", r);
                same_key(half, labels, rates, tag);
            }
        }
    }

    // Production roundtrips (anchors included), single- and multi-scale.
    const std::string tmp_in = ".cs_test_sc_in.png", tmp_enc = ".cs_test_sc_in.cs", tmp_out = ".cs_test_sc_out.png";
    cv::imwrite(tmp_in, photo);
    for (int ms = 0; ms < 2; ++ms) {
        char name[96];
        std::snprintf(name, sizeof(name), "sample-coding %s: encrypt", ms ? "multi-scale" : "single-scale");
        const bool ok = prod_encrypt(tmp_in, tmp_enc, 0.5f, ms != 0, 2000, 0, 0, true) == 0;
        check(ok, name);
        if (!ok) continue;
        const cv::Mat enc = read_container(tmp_enc);
        std::snprintf(name, sizeof(name), "sample-coding %s: decrypt", ms ? "multi-scale" : "single-scale");
        check(prod_decrypt(enc, tmp_out) == 0, name);
        const cv::Mat dec = cv::imread(tmp_out, cv::IMREAD_COLOR);
        const double p = dec.size() == photo.size() ? cs_quality::psnr(photo, dec) : -1.0;
        std::printf("       sample-coding %s PSNR: %.2f dB\n", ms ? "multi-scale" : "single-scale", p);
        std::snprintf(name, sizeof(name), "sample-coding %s: PSNR sane", ms ? "multi-scale" : "single-scale");
        check(std::isfinite(p) && p > SAMPLE_BITS_MIN_PSNR, name);
    }
    std::remove(tmp_in.c_str()); std::remove(tmp_enc.c_str()); std::remove(tmp_out.c_str());
}
