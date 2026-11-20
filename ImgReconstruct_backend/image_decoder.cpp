#include "image_decoder.hpp"
#include "cs_runtime.hpp"
#include "helper_functions.hpp"
#include "sample_coder.hpp"
#include "image_encoder.hpp"
#include "image_preview.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cctype>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <exception>
#include <iostream>
#include <map>
#include <tuple>
#include <set>
#include <utility>
#include <algorithm>
#include <thread>
#include <mutex>
#include <vector>
#include <omp.h>

// A num_tiles x num_tiles grid of per-tile values.
template <class T> using CsGrid = std::vector<std::vector<T>>;
template <class T> static CsGrid<T> cs_grid(int num_tiles) {
    return CsGrid<T>((size_t)num_tiles, std::vector<T>((size_t)num_tiles));
}

decode_image::decode_image(cv::Mat input) {
    parse_container_header(input);
}

void decode_image::parse_container_header(const cv::Mat& input) {
    if (input.empty()) {
        throw std::runtime_error("failed to load encoded image");
    }

    input.copyTo(encoded_img);
    encoded_img = encoded_img.reshape(0, encoded_img.total());

    CsHeaderInfo info;
    if (!cs_parse_header(encoded_img.data, encoded_img.total() * 3, info)) {
        if (info.legacy) {
            throw std::runtime_error("not a v4 container: re-encode the photo with this version");
        }
        throw std::runtime_error("failed to parse encoded image header");
    }

    m = info.m;
    rows = info.rows;
    cols = info.cols;
    org_size.height = info.org_h;
    org_size.width = info.org_w;
    odd_parity = info.odd_parity;

    // the sampling mode travels in the header (no decode-side flags needed)
    sampling_mode = info.sampling_mode;
    sample_bits = info.sample_bits;
    sample_bits_chroma = info.sample_bits_chroma;
    hr_detail = info.hr_detail;
    hr_count = info.hr_count;
    hr_samples.clear();
}

// Restores the raw measurement layout from a bit-packed container (the
// encoder's in-memory anchor pass): each section unpacks independently into a fresh raw-size Mat
// that replaces the packed container, so every downstream reader runs
// unchanged. Sections are (raw byte offset, value count) in ascending
// order. No-op at 8 bits.
void decode_image::unpack_container_measurements(
    const std::vector<cs_body_section>& sections) {
    if ((sample_bits == 8 && sample_bits_chroma == 8) || sections.empty()) {
        return;
    }
    const size_t packed_bytes = encoded_img.total() * encoded_img.elemSize();
    const size_t raw_len = cs_unpacked_body_bytes(sections);
    const size_t raw_pixels = (raw_len + 2) / 3;
    cv::Mat raw(1, (int)raw_pixels, CV_8UC3, cv::Scalar(0, 0, 0));
    cs_unpack_container_body(encoded_img.data, packed_bytes, sections, raw.data);
    encoded_img = raw.reshape(0, (int)raw_pixels);
}

// A coded body is one blob at y0: [u32 Y length][u32 Cr length][u32 Cb length][streams].
constexpr size_t kCodedHeadBytes = 12;
static void cs_coded_stream_lengths(const uint8_t* src, size_t total_bytes, size_t y0, size_t len[3]) {
    if (y0 + kCodedHeadBytes > total_bytes) throw std::runtime_error("coded container: body too short");
    for (int i = 0; i < 3; ++i) len[i] = cs_get_le32(src + y0 + 4 * i);
}

void decode_image::decode_coded_measurements(size_t y0, int mY, int mC, int crows, int ccols) {
    const uint8_t* src = encoded_img.data;
    const size_t total_bytes = encoded_img.total() * encoded_img.elemSize();
    const size_t anchor_len = hr_detail ? (size_t)hr_count : 0;
    size_t len[3];
    cs_coded_stream_lengths(src, total_bytes, y0, len);
    const size_t blob_len = kCodedHeadBytes + len[0] + len[1] + len[2];
    if (y0 + blob_len + anchor_len > total_bytes) throw std::runtime_error("coded container: body too short");
    const uint8_t* blob = src + y0;

    const size_t raw_len = y0 + (size_t)mY + (size_t)2 * mC + anchor_len;
    const size_t raw_pixels = (raw_len + 2) / 3;
    cv::Mat raw(1, (int)raw_pixels, CV_8UC3, cv::Scalar(0, 0, 0));
    std::memcpy(raw.data, src, y0);
    struct Plane { const std::vector<int>* r; const std::vector<int>* c; int n, rows, cols, bits; size_t off; };
    const Plane pl[3] = {
        { &ri_x, &ri_y, mY, rows, cols, sample_bits, y0 },
        { &ri_cx1, &ri_cy1, mC, crows, ccols, sample_bits_chroma, y0 + (size_t)mY },
        { &ri_cx2, &ri_cy2, mC, crows, ccols, sample_bits_chroma, y0 + (size_t)mY + (size_t)mC } };
    const size_t soff[3] = { kCodedHeadBytes, kCodedHeadBytes + len[0], kCodedHeadBytes + len[0] + len[1] };
    bool ok[3] = { false, false, false };
    const auto plane = [&](int i) {
        const Plane& p = pl[i];
        std::vector<uint8_t> val((size_t)p.rows * p.cols, 0), mask(val.size(), 0);
        for (int k = 0; k < p.n; ++k) mask[(size_t)(*p.r)[k] * p.cols + (*p.c)[k]] = 1;
        ok[i] = cs_plane_decode(blob + soff[i], len[i], mask.data(), p.rows, p.cols, p.bits, val.data());
        uint8_t* dst = raw.data + p.off;
        if (ok[i])
            for (int k = 0; k < p.n; ++k)
                dst[k] = cs_dequantize_sample(val[(size_t)(*p.r)[k] * p.cols + (*p.c)[k]], p.bits);
    };
    {   // the chroma planes (a quarter of the luma's positions each) decode beside the luma
        CsJoinedThread chroma;
        chroma.t = std::thread([&] { plane(1); plane(2); });
        plane(0);
    }
    if (!ok[0] || !ok[1] || !ok[2]) throw std::runtime_error("coded container: corrupt sample stream");
    std::memcpy(raw.data + y0 + (size_t)mY + (size_t)2 * mC, src + y0 + blob_len, anchor_len);
    encoded_img = raw.reshape(0, (int)raw_pixels);
}

// Opens the seg-mask body prefix [0][2][0][nclasses][rates][seglen][label PNG],
// caches the rates/labels and regenerates the sample indices into ri_x/ri_y.
// after_labels runs once the labels and rates are cached, before the luma draw. draws (the encoder that wrote this
// container, CsSampler::draw_hint): its luma draw is copied instead of regenerated.
CsSegPrefix decode_image::open_segmask_prefix(const std::function<void()>& after_labels, const CsSampler* draws) {
    const size_t total_bytes = encoded_img.total() * encoded_img.elemSize();
    const uint32_t pre_bytes = cs_get_le32(encoded_img.data + CS_OFF_PAD + CS_OFF_SEGPRE_LEN);
    if (pre_bytes == 0 || CS_HEADER_BYTES + (size_t)pre_bytes > total_bytes) {
        throw std::runtime_error("seg-mask container: corrupt prefix length");
    }
    const uint8_t* prefix = encoded_img.data + CS_HEADER_BYTES;
    CsSegPrefix pre;
    if (!cs_segmask_parse_prefix(prefix, pre_bytes, pre) || pre.total != pre_bytes) {
        throw std::runtime_error("seg-mask container: corrupt prefix");
    }
    const cv::Mat png(1, (int)pre.seglen, CV_8U, (void*)(prefix + (pre.total - pre.seglen)));
    cv::Mat labels = cv::imdecode(png, cv::IMREAD_GRAYSCALE);
    if (labels.empty()) {
        throw std::runtime_error("seg-mask container: label decode failed");
    }
    if (labels.rows != rows || labels.cols != cols) {
        cv::resize(labels, labels, cv::Size(cols, rows), 0, 0, cv::INTER_NEAREST);
    }
    seg_rates_q = pre.rates_q;
    seg_ids = labels;
    chroma_draw_v2 = pre.chroma_draw_v2;
    if (after_labels) after_labels();
    if (draws) copy_luma_draw(*draws);
    else returnSegMaskIndices(ri_x, ri_y, rows, cols, labels, pre.rates_q, cs_sample_seed);
    if ((int)ri_x.size() != m) {
        throw std::runtime_error("seg-mask container: count mismatch");
    }
    return pre;
}

void decode_image::get_sampled_ycc420(cv::Mat& y, cv::Mat& y_mask,
    cv::Mat& cr, cv::Mat& cr_mask, cv::Mat& cb, cv::Mat& cb_mask) {
    if (sampling_mode != CS_MODE_YCC420_SEG && sampling_mode != CS_MODE_YCC420_SEG_HRDETAIL) {
        throw std::runtime_error("not a production container (seg-luma 4:2:0, mode 11 or 19)");
    }

    // The encoder's anchor pass (this thread, this container): its draws are copied, none is regenerated.
    const CsSampler* const draws = draw_hint();
    CsSideTask chroma_draw;
    // Trailing mode-19 anchor payload (compact flag map + luma, hr_count = byte length) after the luma / chroma
    // sections: read into native-luma anchors. Its offset in the container as stored is known before the
    // luma draw (the prefix length sits in the header pad; the stored section lengths follow from m and the
    // sample bits, or head the coded blob), so it is read on a side thread meanwhile. The thread keeps its own
    // reference to the container: unpacking replaces encoded_img below.
    CsSideTask hr_read;
    const bool coded = encoded_img.data[CS_OFF_PAD + CS_OFF_SAMPLE_CODING] == 1;
    // Bit-packed bodies: the anchor offset needs the chroma count, which chroma draw v2 raises from the support
    // (a count floor), so that read starts once the chroma draw is done (below).
    const auto start_hr_read = [&](int mC) {
        hr_read.start([this, coded, mC, src = encoded_img] {
            const size_t total_bytes = src.total() * src.elemSize();
            const size_t y0 = (size_t)CS_HEADER_BYTES + cs_get_le32(src.data + CS_OFF_PAD + CS_OFF_SEGPRE_LEN);
            size_t pay_off = y0;
            if (coded) {
                size_t len[3];
                cs_coded_stream_lengths(src.data, total_bytes, y0, len);
                pay_off += kCodedHeadBytes + len[0] + len[1] + len[2];
            }
            else {
                pay_off += cs_packed_bytes((size_t)m, sample_bits) + 2 * cs_packed_bytes((size_t)mC, sample_bits_chroma);
            }
            const size_t pay_len = (size_t)hr_count;
            if (hr_count == 0 || pay_off + pay_len > total_bytes ||
                !cs_read_anchor_payload(src.data + pay_off, pay_len, cs_sample_seed, hr_samples)) {
                throw std::runtime_error("hr-detail container: corrupt payload");
            }
        });
    };
    if (hr_detail && coded) start_hr_read(0);
    // The body prefix [0][v][0][nclasses][rates][seglen][label PNG] regenerates the luma draw. The chroma draw
    // (inside the label support) needs only the labels and rates, so it starts once they are read and runs beside
    // that draw.
    const CsSegPrefix pre = open_segmask_prefix([&] {
        if (draws) { copy_chroma_draw(*draws); return; }
        chroma_draw.start([&] {
            chroma_support = cs_seg_chroma_support(seg_ids, seg_rates_q);
            returnYcc420ChromaIndices(rows, cols, m, cs_sample_seed);
        });
    }, draws);
    chroma_draw.wait();
    if (hr_detail && !coded) start_hr_read(m_chroma);
    const size_t y0 = (size_t)CS_HEADER_BYTES + pre.total;   // the samples: [Y x mY][Cr x mC][Cb x mC]
    const int mY = m, mC = m_chroma;

    int crows, ccols;
    cs_chroma_dims(rows, cols, crows, ccols);

    // Coded and bit-packed bodies are first restored to one byte per sample (the anchor payload
    // rides after Cb as it is). The length check below then runs against that raw layout.
    if (coded) decode_coded_measurements(y0, mY, mC, crows, ccols);
    else
        unpack_container_measurements(cs_ycc_body_sections(y0, (size_t)mY, (size_t)mC, sample_bits, sample_bits_chroma,
            hr_detail ? (size_t)hr_count : 0));

    if (encoded_img.total() * 3 < y0 + (size_t)mY + (size_t)2 * mC) {
        throw std::runtime_error("ycc container: body too short for luma/chroma payload");
    }
    const uint8_t* body = encoded_img.data + y0;

    hr_read.wait();

    y = cv::Mat::zeros(rows, cols, CV_8U);
    y_mask = cv::Mat::zeros(rows, cols, CV_8U);
    cr = cv::Mat::zeros(crows, ccols, CV_8U);
    cr_mask = cv::Mat::zeros(crows, ccols, CV_8U);
    cb = cv::Mat::zeros(crows, ccols, CV_8U);
    cb_mask = cv::Mat::zeros(crows, ccols, CV_8U);

    // The seg-luma draws and the chroma draws pick distinct positions (partial Fisher-Yates over disjoint pools), so
    // those scatters run in parallel.
    #pragma omp parallel for
    for (int k = 0; k < mY; k++) {
        y.at<uint8_t>(ri_x[k], ri_y[k]) = body[k];
        y_mask.at<uint8_t>(ri_x[k], ri_y[k]) = 1;
    }
    #pragma omp parallel for
    for (int k = 0; k < mC; k++) {
        cr.at<uint8_t>(ri_cx1[k], ri_cy1[k]) = body[mY + k];
        cr_mask.at<uint8_t>(ri_cx1[k], ri_cy1[k]) = 1;
        cb.at<uint8_t>(ri_cx2[k], ri_cy2[k]) = body[mY + mC + k];
        cb_mask.at<uint8_t>(ri_cx2[k], ri_cy2[k]) = 1;
    }
}

float decode_image::get_compression_ratio() {
    return float(m) / (rows * cols);
}
cv::Size decode_image::get_org_size() {
    return org_size;
}

// Initial solver solution of a want-sized tile: the generic image-like ref for Y (resized to the tile, since
// tile_size is only the last processed tile and can disagree with edge/clamped tiles) and a neutral flat-chroma
// solution on the 4:2:0 grid for Cr and Cb, with the /10 convention createRefSolutions uses. Planes with samples
// replace it with a fill of their own samples before the solve.
static void build_generic_warm_start(cv::Mat refs[3], const std::vector<cv::Mat>& generic_refs, cv::Size want)
{
    const cv::Mat& m = generic_refs[0];
    if (m.empty()) {
        refs[0] = cv::Mat(want, CV_32F, cv::Scalar(0));
        cv::dct(refs[0], refs[0], 0);
        refs[0] /= 10.0f;
    } else if (m.size() != want) {
        cv::resize(m, refs[0], want, 0, 0, cv::INTER_AREA);
    } else {
        refs[0] = m.clone();
    }
    const cv::Size chroma_size((want.width + 1) / 2, (want.height + 1) / 2);
    refs[1] = cv::Mat(chroma_size, CV_32F, cv::Scalar(0.5f));
    cs_dct_forward(refs[1]);
    refs[1] /= 10.0f;
    refs[2] = refs[1].clone();
}

// YCC 4:2:0 tile solver: all tiles in one parallel loop (they are independent), each tile solves
// luma at full resolution and each chroma plane natively on its coarse grid
// with per-channel FISTA (DCT, 2 reweights, wscale 2), from a smooth fill of the plane's own
// samples (build_generic_warm_start for a plane without samples). Solved
// planes merge YCrCb -> BGR per tile, and every tile goes through the cs-sr
// 2x stage (cs_sr_upscale_tile_2x) into hr_out.
// tile_coef (per-segment decode) replaces the luma coef per tile.
static void decode_tiles_ycc420(int num_threads,
    const CsGrid<cv::Mat>& y_meas,
    const CsGrid<indices>& y_idx,
    const CsGrid<cv::Mat>& cr_meas,
    const CsGrid<indices>& cr_idx,
    const CsGrid<cv::Mat>& cb_meas,
    const CsGrid<indices>& cb_idx,
    CsGrid<cv::Mat>& mats_out,
    const CsGrid<TileCoord>& coordinates,
    int num_tiles, int iterations, cv::Size tile_size, float coef, int fista_iters,
    CsGrid<cv::Mat>& hr_out, std::mutex& hr_grid_mutex,
    const CsHrSrCtx& hr_sr, int admm_iters, bool prox_off,
    const CsGrid<float>* tile_coef,
    std::shared_ptr<void>* free_later) {   // receives host state the caller can free off the critical path

    constexpr int reweights = 2;
    constexpr float wscale = 2.0f;
    const std::vector<cv::Mat> ref = createRefSolutions(tile_size.width, tile_size.height);
    const int iters_c = iterations - iterations / 2;
    const float coef_c = coef * 0.5f;

    // Luma coef of a tile: its class's (per-segment decode), else the global one.
    auto coef_y_of = [&](int i, int j) { return tile_coef ? (*tile_coef)[i][j] : coef; };
    // coarse (chroma) tile from the same halving the prep used: exact cover
    auto coarse_size = [&](int i, int j) {
        const int Th = mats_out[i][j].rows, Tw = mats_out[i][j].cols;
        const int xo = coordinates[i][j].x, yo = coordinates[i][j].y;
        return cv::Size((xo + Tw + 1) / 2 - xo / 2, (yo + Th + 1) / 2 - yo / 2);
    };

    auto solved = cs_grid<char>(num_tiles);
    // Their 2x tiles (flat as well, the bilinear 2x of the fill) share one buffer per size, read-only.
    std::map<std::pair<int, int>, cv::Mat> ms_flat_hr;
    auto ms_flat = cs_grid<char>(num_tiles);
    // A layer's tiles without any sample (luma or chroma) lie outside its scale's tiles and their margin,
    // where the composite gives the layer no weight: a flat fill stands in for their solve (counted as solved).
    for (int i = 0; i < num_tiles; ++i)
        for (int j = 0; j < num_tiles; ++j)
            if (!y_meas[i][j].empty() && !mats_out[i][j].empty() && y_idx[i][j].ri_x_g.empty() &&
                cr_idx[i][j].ri_x_g.empty() && cb_idx[i][j].ri_x_g.empty()) {
                mats_out[i][j] = cv::Mat(mats_out[i][j].rows, mats_out[i][j].cols, CV_8UC3, cv::Scalar::all(128));
                solved[i][j] = 1;
                cv::Mat& f = ms_flat_hr[{ mats_out[i][j].cols, mats_out[i][j].rows }];
                if (f.empty()) f = cv::Mat(2 * mats_out[i][j].rows, 2 * mats_out[i][j].cols, CV_8UC3, cv::Scalar::all(128));
                ms_flat[i][j] = 1;
            }
    (void)free_later;

    // No tile reads another's result (the old anti-diagonal wavefront fed each tile's warm start from its solved
    // west / north neighbours; the sample fill below replaced it on almost every plane), so one flat loop.
    {
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
        for (int t = 0; t < num_tiles * num_tiles; ++t) {
            try {
                const int i = t / num_tiles;
                const int j = t % num_tiles;
                if (y_meas[i][j].empty()) continue;

                const float tile_coef_y = coef_y_of(i, j);
                const struct indices& y_tidx = y_idx[i][j];
                if (!solved[i][j]) {
                    cv::Mat x0[3];
                    build_generic_warm_start(x0, ref, mats_out[i][j].size());

                    const int Th = mats_out[i][j].rows;
                    const int Tw = mats_out[i][j].cols;
                    const cv::Size coarse_want = coarse_size(i, j);
                    const int cTw = coarse_want.width, cTh = coarse_want.height;
                    // the generic chroma refs are the halved tile; parity on
                    // edge tiles can leave them 1px off the solved grid — refit
                    // (warm start only, the solve re-converges regardless)
                    for (int c = 1; c < 3; ++c) {
                        if (x0[c].empty() || x0[c].size() != coarse_want) {
                            if (x0[c].empty()) {
                                x0[c] = cv::Mat(coarse_want, CV_32F, cv::Scalar(0.5f));
                                cs_dct_forward(x0[c]);
                                x0[c] /= 10.0f;
                            } else {
                                cv::resize(x0[c], x0[c], coarse_want, 0, 0, cv::INTER_LINEAR);
                            }
                        }
                    }

                    cv::Mat dummy;
                    // Luma at full tile resolution, chroma natively on the half-resolution grid, where proximal
                    // updates are exact at the near-flat warm starts.
                    auto solve_plane = [&](const cv::Mat& meas, float c, int R, int C,
                        const std::vector<int>& rx, const std::vector<int>& ry,
                        int it, cv::Mat& r, bool nxt, cv::Mat& nr) {
                        reconstruct_color_channel_fista(meas, 0, c, R, C, rx, ry, it, r, nxt, nr, reweights, fista_iters, wscale);
                    };
                    // Planes start from a smooth fill of their own samples. The generic start is pixels/10 in a solver
                    // whose tail is IDCT x 255, a 10x-dark start that small coefs do not converge
                    // from in the auto iteration count (IMG_3690 at coef x0.2: Cr 23.8 vs 50.8 dB).
                    auto fill_start = [&](const cv::Mat& meas, int R, int C,
                        const std::vector<int>& rx, const std::vector<int>& ry, cv::Mat& r) {
                        if (rx.empty()) return;
                        std::vector<float> b, f;
                        cs_point_sample_fill(meas, 0, R, C, rx, ry, b, f);
                        cv::Mat X(R, C, CV_32F, f.data());
                        cs_dct_forward(X);
                        r = X.clone();
                    };
                    fill_start(y_meas[i][j], Th, Tw, y_tidx.ri_x_g, y_tidx.ri_y_g, x0[0]);
                    fill_start(cr_meas[i][j], cTh, cTw, cr_idx[i][j].ri_x_g, cr_idx[i][j].ri_y_g, x0[1]);
                    fill_start(cb_meas[i][j], cTh, cTw, cb_idx[i][j].ri_x_g, cb_idx[i][j].ri_y_g, x0[2]);
                    solve_plane(y_meas[i][j], tile_coef_y, Th, Tw,
                        y_tidx.ri_x_g, y_tidx.ri_y_g, iterations, x0[0],
                        false, dummy);
                    solve_plane(cr_meas[i][j], coef_c, cTh, cTw,
                        cr_idx[i][j].ri_x_g, cr_idx[i][j].ri_y_g, iters_c, x0[1],
                        cb_idx[i][j].ri_x_g.empty(), x0[2]);   // Cb keeps its own fill when it has samples
                    solve_plane(cb_meas[i][j], coef_c, cTh, cTw,
                        cb_idx[i][j].ri_x_g, cb_idx[i][j].ri_y_g, iters_c, x0[2],
                        false, dummy);

                    cv::Mat cr_full, cb_full;
                    cv::resize(x0[1], cr_full, cv::Size(Tw, Th), 0, 0, cv::INTER_LINEAR);
                    cv::resize(x0[2], cb_full, cv::Size(Tw, Th), 0, 0, cv::INTER_LINEAR);
                    cv::Mat chs[3] = { x0[0], cr_full, cb_full }; // (Y, Cr, Cb)
                    cv::Mat ycc_tile;
                    cv::merge(chs, 3, ycc_tile);
                    ycc_tile.convertTo(ycc_tile, CV_8UC3);
                    cv::cvtColor(ycc_tile, mats_out[i][j], cv::COLOR_YCrCb2BGR);
                }
                if (mats_out[i][j].empty()) continue;
                // Fused 2x: the cs-sr stage refines luma-only against the tile's own LR luma
                // samples: y_meas channel 0 carries luma, and the BGR2Y weights collapse to 1.0
                // on the gray-as-BGR container (chroma comes from the AVIR upscale inside).
                cv::Mat hr;
                if (y_idx[i][j].ri_x_g.empty()) {
                    // A layer's tile outside its scale's tiles and their margin: no samples, and the
                    // composite gives it no weight, so a bilinear 2x stands in for the cs-sr solve.
                    if (ms_flat[i][j]) hr = ms_flat_hr.at({ mats_out[i][j].cols, mats_out[i][j].rows });
                    else cv::resize(mats_out[i][j], hr, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);
                }
                else {
                    CsHrSrCtx tile_hr = hr_sr;
                    tile_hr.ox = coordinates[i][j].x * 2;
                    tile_hr.oy = coordinates[i][j].y * 2;
                    cs_sr_upscale_tile_2x(mats_out[i][j], y_meas[i][j],
                        y_idx[i][j].ri_x_g, y_idx[i][j].ri_y_g,
                        tile_coef_y, iterations, fista_iters, wscale, admm_iters, prox_off, hr, tile_hr);
                }
                if (hr.empty())
                    cs_upscale_2x_avir(mats_out[i][j], hr);
                // Serialize the cell write under the grid mutex (the preview reads the grid). hr is
                // this tile's own buffer (fresh, or the batched pass's), so the cell takes it without a copy.
                if (!hr.empty()) {
                    std::lock_guard<std::mutex> lk(hr_grid_mutex);
                    hr_out[i][j] = hr;
                }
            }
            catch (const std::exception& e) {
                #pragma omp critical
                {
                    std::cerr << "Tile decoding failed: " << e.what() << std::endl;
                }
            }
        }
    }
}

cv::Mat decode_image::read_container(const std::string& path) {
    std::vector<uint8_t> bytes;
    if (!cs_read_container_file(path, bytes) || bytes.empty()) return cv::Mat();
    bytes.resize((bytes.size() + 2) / 3 * 3, 0);
    return cv::Mat(1, (int)(bytes.size() / 3), CV_8UC3, bytes.data()).clone();
}

int decode_image::decode_image_tiled(
    const std::string& input_path,
    const std::string& output_path,
    bool show_preview,
    int fista_iters,
    const CsPhotoUpscalerOptions& photo_up
) {
    const cv::Mat container = read_container(input_path);
    if (container.empty()) {
        std::cerr << "Error: cannot read container '" << input_path << "'" << std::endl;
        return -2;
    }
    return decode_image_tiled(container, output_path, show_preview, fista_iters, photo_up);
}

// Cheap single-image boundary calm for the single-stage solve: edge-aware
// bilateral correction near low-density segment frontiers, no re-solves.
// Range weights follow actual image edges rather than the approximate label
// contour, so ringing is damped without baking a misregistered contour into
// a new discontinuity. Strength fades to zero one overlap width inside the
// class; dense segments pass through untouched.
static void cs_seg_calm_boundaries(cv::Mat& img, const cv::Mat& seg_ids,
    const cv::Mat& sample_mask, const std::vector<int>& classes, int overlap) {
    if (overlap <= 0 || img.empty() || seg_ids.empty() || sample_mask.empty()) return;
    bool low_known = false, lowf[256] = { false };   // low-density classes, from the label-grid pass below
    // Low-density test of every class in one pass (area and sampled count per class) before any
    // full-frame work: most decodes have no low-density class and return here. On the label grid
    // when ids and mask share it and the image is 1x or 2x of it (nearest 2x replicates each label
    // 2x2, so both counts scale by 4 and every ratio is unchanged).
    if (seg_ids.type() == CV_8U && sample_mask.depth() == CV_8U) {
        const bool native = seg_ids.size() == sample_mask.size() &&
            (img.size() == seg_ids.size() || img.size() == cv::Size(2 * seg_ids.cols, 2 * seg_ids.rows));
        cv::Mat ids = seg_ids, chan = sample_mask;
        if (!native) {
            if (ids.size() != img.size()) cv::resize(ids, ids, img.size(), 0, 0, cv::INTER_NEAREST);
            if (chan.size() != img.size()) cv::resize(chan, chan, img.size(), 0, 0, cv::INTER_NEAREST);
        }
        if (chan.channels() > 1) cv::extractChannel(chan, chan, 0);
        long long area[256] = { 0 }, sampled[256] = { 0 };
        #pragma omp parallel
        {
            long long a[256] = { 0 }, s[256] = { 0 };
            #pragma omp for nowait
            for (int y = 0; y < ids.rows; ++y) {
                const uint8_t* ir = ids.ptr<uint8_t>(y);
                const uint8_t* mr = chan.ptr<uint8_t>(y);
                for (int x = 0; x < ids.cols; ++x) { ++a[ir[x]]; s[ir[x]] += mr[x] != 0; }
            }
            #pragma omp critical
            for (int c = 0; c < 256; ++c) { area[c] += a[c]; sampled[c] += s[c]; }
        }
        bool any_low = false;
        for (int c : classes)
            if (c >= 0 && c < 256 && area[c] > 0 && (double)sampled[c] / (double)area[c] < 0.4) any_low = lowf[c] = true;
        if (!any_low) return;
        low_known = true;
    }
    cv::Mat ids = seg_ids, mask = sample_mask;
    if (ids.size() != img.size())
        cv::resize(ids, ids, img.size(), 0, 0, cv::INTER_NEAREST);
    if (mask.size() != img.size())
        cv::resize(mask, mask, img.size(), 0, 0, cv::INTER_NEAREST);
    cv::Mat chan;
    if (mask.channels() == 1) chan = mask;
    else cv::extractChannel(mask, chan, 0);
    // Restrict the expensive bilateral to the union of low-density boundary
    // bands (dilated by the filter radius): bands are a small fraction of
    // the frame, so this is ~50x cheaper than full-frame filtering.
    cv::Mat band_union = cv::Mat::zeros(ids.size(), CV_8U);
    std::vector<cv::Mat> low_masks;
    for (int c : classes) {
        if (low_known && (c < 0 || c >= 256 || !lowf[c])) continue;   // dense: the label-grid pass already decided
        cv::Mat in_c = (ids == c);
        const int area = cv::countNonZero(in_c);
        if (area == 0) continue;
        cv::Mat m;
        cv::bitwise_and(chan, in_c, m);
        if ((double)cv::countNonZero(m) / (double)area >= 0.4) continue;
        low_masks.push_back(in_c);
        cv::Mat dist;
        cv::distanceTransform(in_c, dist, cv::DIST_L2, 3);
        cv::bitwise_or(band_union, dist <= (float)overlap, band_union);
    }
    if (low_masks.empty()) return;
    cv::Mat work_zone;
    cv::dilate(band_union, work_zone, cv::getStructuringElement(cv::MORPH_ELLIPSE, {33, 33}));
    cv::Rect crop = cv::boundingRect(work_zone);
    crop &= cv::Rect(0, 0, img.cols, img.rows);
    if (crop.empty()) return;
    cv::Mat f;
    img.convertTo(f, CV_32FC3);
    cv::Mat smooth_crop;
    cv::bilateralFilter(f(crop), smooth_crop, 13, 18.0, 4.0, cv::BORDER_REPLICATE);
    int touched = 0;
    for (const cv::Mat& in_full : low_masks) {
        cv::Mat in_c = in_full(crop);
        cv::Mat owner, distance, gate, gate3, correction;
        in_c.convertTo(owner, CV_32F, 1.0 / 255.0);
        cv::distanceTransform(in_c, distance, cv::DIST_L2, 3);
        gate = 1.0f - distance / (float)overlap;
        cv::max(gate, 0, gate);
        cv::min(gate, 1, gate);
        cv::multiply(gate, owner, gate);
        cv::merge(std::vector<cv::Mat>{gate, gate, gate}, gate3);
        cv::Mat f_crop = f(crop);
        cv::multiply(smooth_crop - f_crop, gate3, correction);
        f_crop += correction;
        touched++;
    }
    if (touched > 0) {
        f.convertTo(img, CV_8UC3);
        std::cout << "seg-solve: boundary calm on " << touched << " low-density classes" << std::endl;
    }
}

// Tile paste + cosine-feathered blend of the solved tiles over the plain tile paste.
static cv::Mat cs_composite_tiles(const CsGrid<cv::Mat>& tiles,
    const CsGrid<TileCoord>& coords, int feather)
{
    cv::Mat base = reconstructImage(tiles, coords);
    return blendTilesWithImage(tiles, coords, base, 0.5f, feather);
}

// Sorted class ids present in a label map.
static std::vector<int> cs_seg_present_classes(const cv::Mat& ids) {
    bool seen[256] = { false };
    for (int y = 0; y < ids.rows; ++y) {
        const uint8_t* row = ids.ptr<uint8_t>(y);
        for (int x = 0; x < ids.cols; ++x) seen[row[x]] = true;
    }
    std::vector<int> classes;
    for (int c = 0; c < 256; ++c) if (seen[c]) classes.push_back(c);
    return classes;
}

// Per-tile luma coef of the automatic per-segment decode (a YCC-seg container carries its label map). Pure tiles
// take their own class. Mixed tiles take the highest-rate intersecting class: the dense side keeps its detail, and
// over-serving the sparse side is harmless (verified: full background at bird iterations loses nothing); ties fall
// back to majority, then the lower id. The coef is the global one scaled by the class's own sampled rate vs the
// container's achieved ratio (sqrt density rationale of a per-tile law, evaluated per segment), clamped to
// [0.5x, 2x]: sparser classes (fewer measurements) get a stronger prior, denser ones weaker. Verified: +0.2..1.8 dB /
// +0.004..0.028 SSIM and lower CIEDE2000-p95 across 4 images x 3 ratios. winners receives each tile's class (-1 for
// empty tiles). One scan per tile — negligible next to a FISTA solve.
static void cs_seg_resolve_tile_coef(const cv::Mat& ids, const std::vector<uint8_t>& rates_q, float coef,
    float base_ratio, const CsGrid<TileCoord>& coords,
    const CsGrid<cv::Mat>& tile_mats, int num_tiles,
    CsGrid<float>& out, std::vector<int>& winners) {
    out.assign(num_tiles, std::vector<float>(num_tiles, coef));
    winners.assign(num_tiles * num_tiles, -1);
    if (ids.empty()) return;
    std::vector<int> counts(256, 0);
    const auto rate_of = [&](int c) {
        return (c >= 0 && c < (int)rates_q.size()) ? rates_q[c] : 0;
    };
    for (int i = 0; i < num_tiles; ++i) {
        for (int j = 0; j < num_tiles; ++j) {
            if (tile_mats[i][j].empty()) continue;
            const int x0 = coords[i][j].x, y0 = coords[i][j].y;
            const int xe = (std::min)(x0 + tile_mats[i][j].cols, ids.cols);
            const int ye = (std::min)(y0 + tile_mats[i][j].rows, ids.rows);
            std::fill(counts.begin(), counts.end(), 0);
            for (int y = y0; y < ye; ++y) {
                const uint8_t* row = ids.ptr<uint8_t>(y);
                for (int x = x0; x < xe; ++x) counts[row[x]]++;
            }
            int distinct = 0, sole = 0;
            for (int c = 0; c < 256; ++c) {
                if (counts[c] > 0) { distinct++; sole = c; }
            }
            int best = sole, best_n = counts[sole];
            if (distinct > 1) {
                best = -1;
                best_n = -1;
                int best_q = -1;
                for (int c = 0; c < 256; ++c) {
                    if (counts[c] == 0) continue;
                    const int qc = rate_of(c);
                    if (qc > best_q || (qc == best_q && (best < 0 || counts[c] > best_n ||
                        (counts[c] == best_n && c < best)))) {
                        best = c;
                        best_n = counts[c];
                        best_q = qc;
                    }
                }
            }
            winners[i * num_tiles + j] = best;
            float c = coef;
            if (best >= 0 && c > 0.0f && base_ratio > 0.0f) {
                const double rc = (double)rate_of(best) / 255.0;
                if (rc > 0.0) {
                    double f = (double)base_ratio / rc;
                    if (f < 0.5) f = 0.5;
                    if (f > 2.0) f = 2.0;
                    c = (float)((double)c * f);
                }
            }
            out[i][j] = c;
        }
    }
}

// A multi-scale container (mode 20, container_format.hpp) as opened: the photo's size, the tile size, the layers'
// containers and the scale of every tile.
struct CsMsContainer {
    int W = 0, H = 0, T = 0;
    std::vector<int> scale;        // per layer: 2..6
    std::vector<cv::Mat> layers;   // per layer: its seg container
    int layer_of[7] = { -1, -1, -1, -1, -1, -1, -1 };   // scale -> layer
    cv::Mat map;                   // ceil(H / T) x ceil(W / T): the scale of each T px tile
};

static bool cs_is_multiscale(const cv::Mat& container) {
    return container.type() == CV_8UC3 && container.isContinuous() &&
        container.total() * 3 >= (size_t)CS_HEADER_BYTES && container.data[CS_OFF_VERSION] == CS_VERSION &&
        container.data[CS_OFF_PAD] == CS_MODE_MULTISCALE;
}

// Opens the container's descriptor and splits off the layers.
static CsMsContainer cs_ms_open(const cv::Mat& container) {
    const uint8_t* buf = container.data;
    const size_t bytes = container.total() * 3;
    const size_t dlen = cs_get_le32(buf + CS_OFF_PAD + CS_OFF_SEGPRE_LEN);
    std::vector<uint8_t> d;
    if (dlen >= 12 && CS_HEADER_BYTES + dlen <= bytes) d.assign(buf + CS_HEADER_BYTES, buf + CS_HEADER_BYTES + dlen);
    size_t p = 0;
    const auto get = [&](int n) {
        if (p + n > d.size()) throw std::runtime_error("multi-scale: corrupt descriptor");
        uint32_t v = 0;
        for (int b = 0; b < n; ++b) v |= (uint32_t)d[p++] << (8 * b);
        return v;
    };
    CsMsContainer c;
    if (get(1) != 1) throw std::runtime_error("multi-scale: unknown descriptor version");
    c.W = (int)get(2);
    c.H = (int)get(2);
    c.T = (int)get(2);
    const int n = (int)get(1);
    if (c.W < 2 || c.H < 2 || c.T < 8 || n < 1 || n > 5) throw std::runtime_error("multi-scale: corrupt descriptor");
    c.scale.resize(n);
    std::vector<size_t> len(n);
    for (int i = 0; i < n; ++i) { c.scale[i] = (int)get(1); len[i] = get(4); }
    const size_t mlen = get(4);
    if (p + mlen > d.size()) throw std::runtime_error("multi-scale: corrupt descriptor");
    c.map = cv::imdecode(std::vector<uint8_t>(d.begin() + (ptrdiff_t)p, d.begin() + (ptrdiff_t)(p + mlen)), cv::IMREAD_GRAYSCALE);
    if (c.map.rows != (c.H + c.T - 1) / c.T || c.map.cols != (c.W + c.T - 1) / c.T)
        throw std::runtime_error("multi-scale: corrupt scale map");
    c.layers.resize(n);
    size_t off = CS_HEADER_BYTES + dlen;
    for (int i = 0; i < n; ++i) {
        if (len[i] == 0 || len[i] % 3 != 0 || off + len[i] > bytes) throw std::runtime_error("multi-scale: corrupt layer table");
        if (c.scale[i] < 2 || c.scale[i] > 6 || c.layer_of[c.scale[i]] >= 0) throw std::runtime_error("multi-scale: corrupt layer table");
        c.layer_of[c.scale[i]] = i;
        c.layers[i] = cv::Mat(1, (int)(len[i] / 3), CV_8UC3, const_cast<uint8_t*>(buf + off)).clone();
        off += len[i];
    }
    for (int y = 0; y < c.map.rows; ++y)   // every tile's scale has a layer
        for (int x = 0; x < c.map.cols; ++x)
            if (c.map.at<uint8_t>(y, x) > 6 || c.layer_of[c.map.at<uint8_t>(y, x)] < 0) throw std::runtime_error("multi-scale: corrupt scale map");
    return c;
}

// The layers' decodes blended into the photo, tile by tile: a tile whose 3x3 neighbourhood holds one scale is that
// layer resampled; the others blend their neighbours' layers with the weights of a 17x17 box over the tile-constant
// scale map (separable on the tile grid: per row / column at most two tiles, by their share of the window).
static cv::Mat cs_ms_composite(const CsMsContainer& c, const std::vector<cv::Mat>& decs) {
    const int W = c.W, H = c.H, T = c.T, th = c.map.rows, tw = c.map.cols;
    const cv::Mat& map = c.map;
    constexpr int R = 8;   // the 17x17 box (0, 4 and 16 measured worse with the encoder's 8 px margin)
    struct Share { int t0, t1; float f0; };   // window share of tile t0 (f0) and t1 (1 - f0)
    const auto shares = [&](int len_px) {
        std::vector<Share> v((size_t)len_px);
        for (int p = 0; p < len_px; ++p) {
            int t0 = -1, t1 = -1, c0 = 0;
            for (int k = -R; k <= R; ++k) {
                const int t = (std::min)(len_px - 1, (std::max)(0, p + k)) / T;   // BORDER_REPLICATE
                if (t0 < 0 || t == t0) { t0 = t; ++c0; } else t1 = t;
            }
            v[(size_t)p] = { t0, t1 < 0 ? t0 : t1, c0 / (2.0f * R + 1.0f) };
        }
        return v;
    };
    const std::vector<Share> sy = shares(H), sx = shares(W);
    cv::Mat out(H, W, CV_8UC3);
    #pragma omp parallel for schedule(dynamic, 4)
    for (int t = 0; t < th * tw; ++t) {
        const int ty = t / tw, tx = t % tw;
        const cv::Rect r(tx * T, ty * T, (std::min)(T, W - tx * T), (std::min)(T, H - ty * T));
        bool used[7] = {};
        int nused = 0;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                const int yy = ty + dy, xx = tx + dx;
                if (yy < 0 || yy >= th || xx < 0 || xx >= tw) continue;
                const int s = map.at<uint8_t>(yy, xx);
                nused += !used[s];
                used[s] = true;
            }
        // Each layer in use here resampled over the tile (cubic, the pixel-centre mapping of cv::resize).
        cv::Mat res[7];
        for (int s = 2; s <= 6; ++s) {
            if (!used[s]) continue;
            const cv::Mat& d = decs[(size_t)c.layer_of[s]];
            if (d.size() == cv::Size(W, H)) { res[s] = d(r); continue; }
            const float fx = (float)d.cols / W, fy = (float)d.rows / H;
            cv::Mat mx(r.height, r.width, CV_32F), my(r.height, r.width, CV_32F);
            for (int y = 0; y < r.height; ++y)
                for (int x = 0; x < r.width; ++x) {
                    mx.at<float>(y, x) = (r.x + x + 0.5f) * fx - 0.5f;
                    my.at<float>(y, x) = (r.y + y + 0.5f) * fy - 0.5f;
                }
            cv::remap(d, res[s], mx, my, cv::INTER_CUBIC, cv::BORDER_REPLICATE);
        }
        if (nused == 1) { res[map.at<uint8_t>(ty, tx)].copyTo(out(r)); continue; }
        for (int y = 0; y < r.height; ++y) {
            const Share& a = sy[(size_t)(r.y + y)];
            uint8_t* o = out.ptr<uint8_t>(r.y + y) + 3 * r.x;
            for (int x = 0; x < r.width; ++x) {
                const Share& b = sx[(size_t)(r.x + x)];
                const int st[4] = { map.at<uint8_t>(a.t0, b.t0), map.at<uint8_t>(a.t0, b.t1),
                                    map.at<uint8_t>(a.t1, b.t0), map.at<uint8_t>(a.t1, b.t1) };
                const float wt[4] = { a.f0 * b.f0, a.f0 * (1 - b.f0), (1 - a.f0) * b.f0, (1 - a.f0) * (1 - b.f0) };
                float acc[3] = { 0, 0, 0 };
                for (int k = 0; k < 4; ++k) {
                    if (wt[k] == 0.0f) continue;
                    const uint8_t* p = res[st[k]].ptr<uint8_t>(y) + 3 * x;
                    for (int ch = 0; ch < 3; ++ch) acc[ch] += wt[k] * p[ch];
                }
                for (int ch = 0; ch < 3; ++ch) o[3 * x + ch] = cv::saturate_cast<uint8_t>(acc[ch]);
            }
        }
    }
    return out;
}

// Multi-scale container: decodes every layer with the caller's options, then composites them.
static int cs_decode_multiscale(const cv::Mat& container, const std::string& output_path,
    int fista_iters, const CsPhotoUpscalerOptions& photo_up, cv::Mat* decoded_out) {
    try {
        const auto t0 = std::chrono::high_resolution_clock::now();
        const auto ms = [](auto a, auto b) { return (long long)std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count(); };
        const CsMsContainer c = cs_ms_open(container);
        const int n = (int)c.layers.size();
        // One thread per layer: each unpacks, preps and solves on its own.
        std::vector<int> rc(n, -1);
        std::vector<cv::Mat> decs(n);
        {
            std::vector<std::thread> threads;
            for (int i = 0; i < n; ++i)
                threads.emplace_back([&, i] {
                    const auto tl = std::chrono::high_resolution_clock::now();
                    try {
                        cs_layer_scale_tl = c.scale[i];   // the layer's chroma draw (v2 stratifies coarse layers)
                        rc[i] = decode_image::decode_image_tiled(c.layers[i], "", false, fista_iters, photo_up, &decs[i]);
                    } catch (...) { rc[i] = -1; }
                    std::printf("multi-scale: %dx layer done %lld ms after the start\n", c.scale[i], ms(tl, std::chrono::high_resolution_clock::now()));
                });
            for (std::thread& t : threads) t.join();
        }
        for (int i = 0; i < n; ++i)
            if (rc[i] != 0 || decs[i].empty() || decs[i].type() != CV_8UC3)
                throw std::runtime_error("multi-scale: layer " + std::to_string(c.scale[i]) + "x failed to decode");
        const auto tc = std::chrono::high_resolution_clock::now();
        const cv::Mat out = cs_ms_composite(c, decs);
        std::cout << "multi-scale: composite " << ms(tc, std::chrono::high_resolution_clock::now()) << " ms" << std::endl;
        if (!output_path.empty() && !cv::imwrite(output_path, out))
            throw std::runtime_error("failed to write '" + output_path + "'");
        if (decoded_out) *decoded_out = out;
        std::cout << "multi-scale: " << n << " layers composited, " << ms(t0, std::chrono::high_resolution_clock::now()) << " ms" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}

int decode_image::decode_image_tiled(
    const cv::Mat& encoded_img_in,
    const std::string& output_path,
    bool show_preview,
    int fista_iters,
    const CsPhotoUpscalerOptions& photo_up,
    cv::Mat* decoded_out
) {
    if (cs_is_multiscale(encoded_img_in))
        return cs_decode_multiscale(encoded_img_in, output_path, fista_iters, photo_up, decoded_out);
    if (fista_iters < 0 || fista_iters > 500) {
        std::cerr << "Error: fista_iters must be in [0, 500] (0 = auto-map from the iterations)" << std::endl;
        return -1;
    }

    // header parse: a foreign or corrupt container is rejected before any solve
    decode_image dimgs;
    try {
        dimgs = decode_image(encoded_img_in);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -2;
    }
    const int sm = dimgs.get_sampling_mode();
    if (sm != CS_MODE_YCC420_SEG && sm != CS_MODE_YCC420_SEG_HRDETAIL) {
        std::cerr << "Error: container mode " << sm << " is not a production format" << std::endl;
        return -2;
    }
    const cv::Size org_size = dimgs.get_org_size();
    const float ratio = dimgs.get_compression_ratio();

    // iterations: the solver saturates by ~5 steps across compression
    // ratios 0.25..1.0 (the PSNR plateau starts at 5 everywhere, time scales
    // linearly beyond it). Keep a knee-proportional allocation with a cap of 8.
    const int iterations = (std::min)(8, (int)std::ceil(5.0f / ratio));

    // Tile grid of ~32-px tiles (about 64x64 on a 2000 px image), the
    // regime cs_auto_reg_coefficient was fitted for. One whole-image tile
    // measured up to ~4 dB worse on some images (flat on others) and was
    // several times slower. org_size is already the solve-grid size here
    // (the output is 2x org_size), so no halving.
    const int overlap = photo_up.auto_overlap;
    // The square count: the grid size the regularization law sees.
    const int reg_tiles = (std::min)(96, (std::max)(1, ((std::max)(org_size.width, org_size.height) + 16) / 32));
    // Per-axis counts: at least the square count, and enough that no tile
    // exceeds 48 x 64 solve px (rows x cols; the size the grid was
    // fitted with). Images that already fit keep
    // the square grid as is.
    constexpr int kMaxRows = 48, kMaxCols = 64;
    const auto tiles_for = [&](int len, int max_len) {
        const int step = max_len - overlap;   // stride between tile origins
        return (std::max)(reg_tiles, step > 0 ? (len - overlap + step - 1) / step : reg_tiles);
    };
    dimgs.grid_rows = tiles_for(org_size.height, kMaxRows);
    dimgs.grid_cols = tiles_for(org_size.width, kMaxCols);
    if (dimgs.grid_rows != reg_tiles || dimgs.grid_cols != reg_tiles)
        std::cout << "tile grid: " << dimgs.grid_rows << " x " << dimgs.grid_cols
                  << " (tiles kept within the SR kernel's 96 x 128 px)" << std::endl;
    const int num_tiles = (std::max)(dimgs.grid_rows, dimgs.grid_cols);

    // Automatic regularization: the coef from the fitted law. The identity-prox decoder (prox_off): the law
    // over-regularizes it; x0.2 is the 77-photo sweep's optimum at every ratio (x0.15-0.2 plateau, +0.20 dB).
    float coef = cs_auto_reg_coefficient(ratio, reg_tiles);
    if (photo_up.prox_off) coef *= 0.2f;

    // Luma at full resolution, chroma natively on the 4:2:0 grid.
    return decode_image_tiled_ycc420(dimgs, output_path, num_tiles, overlap,
        iterations, omp_get_max_threads(), coef, show_preview, fista_iters, photo_up, decoded_out);
}

// Tile-local (row, column) lists of a tile mask's sampled pixels (1 == sampled), in raster order.
static indices cs_mask_positions(const cv::Mat& mask, size_t reserve) {
    std::vector<int> r, c;
    r.reserve(reserve); c.reserve(reserve);
    for (int y = 0; y < mask.rows; y++) {
        const uint8_t* mp = mask.ptr<uint8_t>(y);
        for (int x = 0; x < mask.cols; x++) {
            if (mp[x]) { r.push_back(y); c.push_back(x); }
        }
    }
    return { r, c };
}

// A tile plane's ephemeral v1 container for the solvers: the gray plane as BGR, channel 0 carries the samples
// (the convention reconstruct reads).
static cv::Mat cs_tile_container(const cv::Mat& plane, const indices& at) {
    cv::Mat bgr;
    cv::cvtColor(plane, bgr, cv::COLOR_GRAY2BGR);
    encode_image img(bgr, false);
    img.encode(at.ri_x_g, at.ri_y_g);
    return img.get_mat();
}

int decode_image::decode_image_tiled_ycc420(decode_image& dimgs,
    const std::string& output_path, int num_tiles, int overlap, int iterations,
    int num_threads, float coef, bool show_preview, int fista_iters,
    const CsPhotoUpscalerOptions& photo_up, cv::Mat* decoded_out) {
    try {
        const cv::Size org_size = dimgs.get_org_size();
        const std::string windowName = output_path.empty() ? "decoded tiles" : output_path;

        // per-plane sampled images + 0/1 masks (luma full-res, chroma coarse)
        cv::Mat Y, Ym, Cr, Crm, Cb, Cbm;
        dimgs.get_sampled_ycc420(Y, Ym, Cr, Crm, Cb, Cbm);
        // The grid solve's per-tile containers (tens of thousands of small buffers) are freed on
        // this thread while the calm pass and the write run; joined when the decode returns.
        CsJoinedThread tile_frees;
        // Automatic per-segment decode: the container carries its label map, so each tile takes a luma
        // coef scaled by its own class's sampled rate (chroma keeps the global coef at 0.5x).
        const bool autoseg = !dimgs.get_seg_ids().empty();
        if (autoseg)
            std::cout << "seg-solve: per-class auto params ("
                      << (int)dimgs.get_seg_rates_q().size() << " classes)" << std::endl;

        cv::Mat reconstructed;
        {
        const int rows = Y.rows, cols = Y.cols;

        // luma tiling reuses the standard splitter; chroma tiles cover the matching 4:2:0 rects
        CsGrid<cv::Mat> y_tiles, y_mask_tiles;
        CsGrid<TileCoord> coordinates, mask_coords;
        const int grid_r = dimgs.grid_rows > 0 ? dimgs.grid_rows : num_tiles;
        const int grid_c = dimgs.grid_cols > 0 ? dimgs.grid_cols : num_tiles;
        splitImageIntoTiles(Y, y_tiles, coordinates, grid_r, grid_c, overlap);
        splitImageIntoTiles(Ym, y_mask_tiles, mask_coords, grid_r, grid_c, overlap);
        CsGrid<float> seg_coef;
        if (autoseg) {
            std::vector<int> seg_winners;
            cs_seg_resolve_tile_coef(dimgs.get_seg_ids(), dimgs.get_seg_rates_q(), coef, dimgs.get_compression_ratio(),
                coordinates, y_tiles, num_tiles, seg_coef, seg_winners);
            std::vector<int> per_class(256, 0);
            for (int v : seg_winners) if (v >= 0) per_class[v]++;
            std::cout << "seg-solve: single stage, per-class tile counts:";
            for (int c = 0; c < 256; ++c)
                if (per_class[c] > 0) std::cout << " " << c << "=" << per_class[c];
            std::cout << std::endl;
        }

        auto cr_tiles = cs_grid<cv::Mat>(num_tiles), cr_mask_tiles = cr_tiles, cb_tiles = cr_tiles, cb_mask_tiles = cr_tiles;
        auto y_idx = cs_grid<indices>(num_tiles), cr_idx = y_idx, cb_idx = y_idx;
        auto y_eph = cs_grid<cv::Mat>(num_tiles), cr_eph = y_eph, cb_eph = y_eph;   // the tiles' solver containers
        auto decoded = cs_grid<cv::Mat>(num_tiles);

        cv::Size tile_size(0, 0);
        // Kept tiles first, then the per-tile prep in parallel; tile_size is the last kept
        // tile's, as in the serial order.
        std::vector<char> kept_tile((size_t)num_tiles * num_tiles, 0);
        for (int i = 0; i < num_tiles; i++)
            for (int j = 0; j < num_tiles; j++) {
                if (y_tiles[i][j].empty() || y_mask_tiles[i][j].empty()) continue;
                kept_tile[(size_t)i * num_tiles + j] = 1;
                tile_size = y_tiles[i][j].size();
            }
        std::exception_ptr prep_error;
        #pragma omp parallel for schedule(dynamic, 8)
        for (int t = 0; t < num_tiles * num_tiles; t++) {
            if (!kept_tile[(size_t)t]) continue;
            const int i = t / num_tiles, j = t % num_tiles;
            try {
                const int x = coordinates[i][j].x, y0 = coordinates[i][j].y;
                const int w = y_tiles[i][j].cols, h = y_tiles[i][j].rows;
                // the tile's 4:2:0 rect (the solver uses the same formula)
                const cv::Rect crect(x / 2, y0 / 2, (x + w + 1) / 2 - x / 2, (y0 + h + 1) / 2 - y0 / 2);
                cr_tiles[i][j] = Cr(crect).clone();
                cr_mask_tiles[i][j] = Crm(crect).clone();
                cb_tiles[i][j] = Cb(crect).clone();
                cb_mask_tiles[i][j] = Cbm(crect).clone();

                y_idx[i][j] = cs_mask_positions(y_mask_tiles[i][j], 1024);
                cr_idx[i][j] = cs_mask_positions(cr_mask_tiles[i][j], 256);
                cb_idx[i][j] = cs_mask_positions(cb_mask_tiles[i][j], 256);
                y_eph[i][j] = cs_tile_container(y_tiles[i][j], y_idx[i][j]);
                cr_eph[i][j] = cs_tile_container(cr_tiles[i][j], cr_idx[i][j]);
                cb_eph[i][j] = cs_tile_container(cb_tiles[i][j], cb_idx[i][j]);

                decoded[i][j] = cv::Mat::zeros(h, w, CV_8UC3);
            }
            catch (...) {
                #pragma omp critical
                if (!prep_error) prep_error = std::current_exception();
            }
        }
        if (prep_error) std::rethrow_exception(prep_error);

        reconstructed = cv::Mat::zeros(rows, cols, CV_8UC3);

        // Every tile is upscaled 2x (cs-sr) during the solve; the preview follows the HR grid
        // (a tile appears only once upscaled).
        auto ycc_hr_tiles = cs_grid<cv::Mat>(num_tiles);
        ImagePreview preview;
        std::mutex ycc_hr_grid_mutex;
        CsGrid<TileCoord> ycc_hr_coordinates;
        if (show_preview) {
            ycc_hr_coordinates = coordinates;
            for (int i = 0; i < num_tiles; ++i)
                for (int j = 0; j < num_tiles; ++j) {
                    ycc_hr_coordinates[i][j].x = coordinates[i][j].x * 2;
                    ycc_hr_coordinates[i][j].y = coordinates[i][j].y * 2;
                }
            preview.Start(windowName, reconstructed, ycc_hr_coordinates,
                          ycc_hr_tiles, &ycc_hr_grid_mutex);
        }

        // HR anchors for the fused cs-sr solve (stride is the luma output width).
        CsHrSrCtx ycc_hr_sr_ctx;
        if (!dimgs.hr_samples.empty()) {
            ycc_hr_sr_ctx.anchors = &dimgs.hr_samples;
            ycc_hr_sr_ctx.stride = cols * 2;
        }
        std::shared_ptr<void> sr_free_later;   // freed on tile_frees with the tile containers
        decode_tiles_ycc420(num_threads, y_eph, y_idx, cr_eph, cr_idx, cb_eph, cb_idx,
            decoded, coordinates, num_tiles, iterations, tile_size, coef, fista_iters,
            ycc_hr_tiles, ycc_hr_grid_mutex, ycc_hr_sr_ctx, photo_up.sr_pnp_iters, photo_up.prox_off,
            autoseg ? &seg_coef : nullptr, &sr_free_later);

        if (show_preview) {
            preview.Stop();
        }

        // Fused tiles are adopted (AVIR fallback for misses); blending happens in 2x solve geometry.
        for (int i = 0; i < num_tiles; ++i)
            for (int j = 0; j < num_tiles; ++j) {
                if (ycc_hr_tiles[i][j].empty() && !decoded[i][j].empty())
                    cs_upscale_2x_avir(decoded[i][j], ycc_hr_tiles[i][j]);
                coordinates[i][j].x *= 2;
                coordinates[i][j].y *= 2;
            }
        decoded = std::move(ycc_hr_tiles);

        // Feathered compositing in 2x geometry over a plain tile paste (zeros would halve every
        // pixel in the zero-overlap alpha blend); then resize to 2x
        // the recorded size
        reconstructed = cs_composite_tiles(decoded, coordinates, overlap * 2);

        const cv::Size final_size(org_size.width * 2, org_size.height * 2);
        if (final_size.width > 0 && final_size.height > 0 && reconstructed.size() != final_size)
            cv::resize(reconstructed, reconstructed, final_size);

        tile_frees.t = std::thread([a = std::move(y_tiles), b = std::move(y_mask_tiles), c = std::move(cr_tiles),
            d = std::move(cr_mask_tiles), e = std::move(cb_tiles), f = std::move(cb_mask_tiles),
            g = std::move(y_idx), h = std::move(cr_idx), i = std::move(cb_idx), j = std::move(y_eph),
            k = std::move(cr_eph), l = std::move(cb_eph), m = std::move(decoded),
            o = std::move(sr_free_later)] {});
        }

        // Frontier calm (luma mask Ym): the per-class decode otherwise leaves the low-density
        // boundary bands uncalmed (halo where the class coef steps).
        if (autoseg) {
            const cv::Mat& sid = dimgs.get_seg_ids();
            cs_seg_calm_boundaries(reconstructed, sid, Ym, cs_seg_present_classes(sid), overlap);
        }

        // HR anchors: exact native-luma replacements on the final composite.
        if (!dimgs.hr_samples.empty() && !reconstructed.empty()) {
            const size_t applied = cs_apply_hr_detail(reconstructed, dimgs.hr_samples);
            std::cout << "HR-detail: applied " << applied << "/"
                      << dimgs.hr_samples.size() << " anchors" << std::endl;
        }

        if (decoded_out) *decoded_out = reconstructed;   // in-memory caller (encoder anchor pass): padded geometry
        else {
        dimgs.crop_odd_parity(reconstructed);
        cv::imwrite(output_path, reconstructed);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: ycc decoding pipeline failed: " << e.what() << std::endl;
        return -3;
    }

    return 0;
}

