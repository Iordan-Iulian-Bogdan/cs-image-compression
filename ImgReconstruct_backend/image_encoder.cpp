#include "image_encoder.hpp"
#include "cs_runtime.hpp"
#include "sample_coder.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cstring>
#include <thread>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <iostream>

// Measurement stamping helpers. cv::Mat::at<T>(r,c) recomputes
// data + step*r + c*elemSize() on every element and the compiler cannot hoist
// step out of the loop because the source and destination Mats alias through
// their members. Hoisting the row stride and copying fixed-size elements with
// memcpy lets the compiler emit a bare load/store per sample instead of an
// integer multiply plus a member reload. Output is bit-identical.
static inline void cs_gather_bgr(const uint8_t* base, size_t step,
                                 const int* rx, const int* ry, int m, uint8_t* dst) {
    for (int k = 0; k < m; ++k) {
        std::memcpy(dst + (size_t)k * 3,
                    base + (size_t)rx[k] * step + (size_t)ry[k] * 3, 3);
    }
}

static inline void cs_gather_u8(const uint8_t* base, size_t step,
                                const int* rx, const int* ry, int m, uint8_t* dst) {
    for (int k = 0; k < m; ++k) {
        dst[k] = base[(size_t)rx[k] * step + (size_t)ry[k]];
    }
}
encode_image::encode_image(const cv::Mat& input, bool global_image) {
    input.copyTo(input_img);
    rows = input_img.rows;
    cols = input_img.cols;
    org_size.width = cols;
    org_size.height = rows;
    // No mult-8 stretch of global images: every sampler draws indices on the
    // rows x cols grid and gathers pixels from input_img, so a stretched
    // input_img shifted content by up to 4 output px on sizes not divisible
    // by 8 (cropping, not undoing, the stretch). Tilers handle partial tiles.
    (void)global_image;
}

// ---- container build steps shared by every encode_* mode ----

// Allocates the 1 x total container and writes its header.
// Metadata carries the budget m; returns the container byte buffer.
uint8_t* encode_image::alloc_container(int total) {
    encoded_img = cv::Mat(1, total, CV_8UC3, cv::Scalar(0, 0, 0));

    const std::string text = cs_build_metadata(m, rows, cols, org_size.height, org_size.width, odd_parity);
    uint8_t* buf = encoded_img.data;
    if (!cs_write_header(buf, total * 3, text)) {
        throw std::runtime_error("failed to write the container header");
    }
    return buf;
}

// Records the sample bit depths in the pad.
static void stamp_sample_bits(uint8_t* buf, int sample_bits, int chroma_bits) {
    buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS] = (uint8_t)sample_bits;
    buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_CHROMA] = (uint8_t)chroma_bits;
    buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_MARKER] = CS_SAMPLE_BITS_MARKER;
}

// Sample coding: replaces the raw container (8-bit values in draw order at y0: [Y x mY][Cr x mC][Cb x mC], then
// the anchor payload) by the coded one (CS_OFF_SAMPLE_CODING). Each plane's values are quantized to its
// sample bits, laid on the plane by their positions and entropy-coded.
void encode_image::code_container_body(size_t y0, int mY, int mC, int crows, int ccols, size_t anchor_len,
    int sample_bits, int chroma_bits) {
    const uint8_t* raw = encoded_img.data;
    struct Plane { const std::vector<int>* r; const std::vector<int>* c; int n, rows, cols, bits; size_t off; };
    const Plane pl[3] = {
        { &ri_x, &ri_y, mY, rows, cols, sample_bits, y0 },
        { &ri_cx1, &ri_cy1, mC, crows, ccols, chroma_bits, y0 + (size_t)mY },
        { &ri_cx2, &ri_cy2, mC, crows, ccols, chroma_bits, y0 + (size_t)mY + (size_t)mC } };
    std::vector<uint8_t> blob(12);
    for (int i = 0; i < 3; ++i) {
        const Plane& p = pl[i];
        std::vector<uint8_t> val((size_t)p.rows * p.cols, 0), mask(val.size(), 0);
        for (int k = 0; k < p.n; ++k) {
            const size_t j = (size_t)(*p.r)[k] * p.cols + (*p.c)[k];
            if (mask[j]) throw std::runtime_error("sample coding: a position was drawn twice");
            mask[j] = 1;
            val[j] = cs_quantize_sample(raw[p.off + k], p.bits);
        }
        const std::vector<uint8_t> s = cs_plane_encode(val.data(), mask.data(), p.rows, p.cols, p.bits);
        cs_put_le32(blob.data() + 4 * i, (uint32_t)s.size());
        blob.insert(blob.end(), s.begin(), s.end());
    }
    const size_t len = y0 + blob.size() + anchor_len;
    cv::Mat out(1, (int)((len + 2) / 3), CV_8UC3, cv::Scalar(0, 0, 0));
    std::memcpy(out.data, raw, y0);   // header + seg prefix
    out.data[CS_OFF_PAD + CS_OFF_SAMPLE_CODING] = 1;
    std::memcpy(out.data + y0, blob.data(), blob.size());
    std::memcpy(out.data + y0 + blob.size(), raw + y0 + (size_t)mY + (size_t)2 * mC, anchor_len);
    encoded_img = out;
}

cv::Mat encode_image::decode_own_container(const std::function<cv::Mat(const cv::Mat&)>& decode) {
    struct Scope {
        explicit Scope(const CsSampler* e) { draw_hint() = e; }
        ~Scope() { draw_hint() = nullptr; }
    } scope(this);
    return decode(encoded_img);
}

void encode_image::get_mat(cv::Mat& dest) {
    encoded_img.copyTo(dest);
}

cv::Mat encode_image::get_mat() {
    return encoded_img;// .clone();
}

void encode_image::sample_positions(std::vector<int>& rows, std::vector<int>& cols) const {
    rows = ri_x; cols = ri_y;
}

// source | binary mask | cyan overlay of the sample positions (rx = row, ry = col) on img.
static bool cs_sampling_mask_view(const cv::Mat& input_img, const std::vector<int>& ri_x, const std::vector<int>& ri_y,
    cv::Mat& out) {
    if (input_img.empty() || ri_x.empty() || ri_x.size() != ri_y.size()) {
        return false;
    }
    const int r = input_img.rows, c = input_img.cols;
    cv::Mat mask(r, c, CV_8UC1, cv::Scalar(0));
    for (size_t k = 0; k < ri_x.size(); ++k) {
        if (ri_x[k] >= 0 && ri_x[k] < r && ri_y[k] >= 0 && ri_y[k] < c) {
            mask.at<uchar>(ri_x[k], ri_y[k]) = 255;
        }
    }

    cv::Mat gray;
    if (input_img.channels() == 3) {
        cv::cvtColor(input_img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = input_img;
    }
    cv::Mat gray3;
    cv::cvtColor(gray, gray3, cv::COLOR_GRAY2BGR);

    cv::Mat mask3;
    cv::cvtColor(mask, mask3, cv::COLOR_GRAY2BGR);

    cv::Mat overlay = gray3.clone();
    overlay.setTo(cv::Scalar(60, 60, 60));
    cv::Mat tint(r, c, CV_8UC3, cv::Scalar(0, 0, 0));
    tint.setTo(cv::Scalar(255, 180, 0), mask);
    cv::addWeighted(overlay, 0.55, gray3, 0.45, 0, overlay);
    (cv::max)(overlay, tint, overlay);

    auto label = [](cv::Mat& panel, const std::string& t) {
        cv::putText(panel, t, cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.6,
            cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(panel, t, cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.6,
            cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    };
    char info[96];
    std::snprintf(info, sizeof(info), "mask m=%zu ratio~%.2f", ri_x.size(),
        (double)ri_x.size() / (double)((long long)r * c));
    label(gray3, "source");
    label(mask3, info);
    label(overlay, "cyan=sampled");

    cv::Mat row;
    cv::hconcat(std::vector<cv::Mat>{gray3, mask3, overlay}, row);
    out = row;
    return true;
}

// --show-mask output: <output>.mask.png, then (window) a window until a key.
static void cs_present_mask_view(const cv::Mat& view, const std::string& output_path, bool window) {

    std::string mask_path;
    if (!output_path.empty()) {
        mask_path = output_path;
        const size_t dot = mask_path.find_last_of('.');
        if (dot != std::string::npos && dot > mask_path.find_last_of("/\\")) {
            mask_path.insert(dot, ".mask");
        } else {
            mask_path += ".mask";
        }
        if (mask_path.size() < 4 || mask_path.substr(mask_path.size() - 4) != ".png") {
            mask_path += ".png";
        }
        if (!cv::imwrite(mask_path, view)) {
            std::cerr << "Warning: --show-mask: failed to write " << mask_path << "\n";
            mask_path.clear();
        } else {
            std::cout << "Sampling mask written to " << mask_path << std::endl;
        }
    }
    if (!window) return;

    try {
        const char* win = "sampling mask";
        cv::namedWindow(win, cv::WINDOW_NORMAL);
        // fit roughly on screen without depending on desktop metrics
        const int max_w = 1600, max_h = 900;
        double s = 1.0;
        if (view.cols > max_w) s = (double)max_w / view.cols;
        if (view.rows * s > max_h) s = (double)max_h / view.rows;
        cv::Mat show;
        if (s < 1.0) {
            cv::resize(view, show, cv::Size(), s, s, cv::INTER_AREA);
        } else {
            show = view;
        }
        cv::imshow(win, show);
        std::cout << "Press any key in the mask window to continue..." << std::endl;
        cv::waitKey(0);
        cv::destroyWindow(win);
    } catch (const cv::Exception&) {
        // headless / no GUI backend: file output above is enough
        if (mask_path.empty()) {
            std::cerr << "Warning: --show-mask: GUI unavailable and no mask path set\n";
        }
    }
}

void encode_image::encode_ycc420_segmask_ids(const cv::Mat& ids, const std::vector<float>& rates,
    const CsAnchorPlan* anchors, bool coded, bool same_as_last) {
    constexpr int sample_bits = 6, chroma_bits = 8;
    const int out_mode = anchors ? CS_MODE_YCC420_SEG_HRDETAIL : CS_MODE_YCC420_SEG;
    if (rates.empty() || rates.size() > 255) {
        throw std::runtime_error("seg-mask: need 1..255 per-class rates");
    }
    if (ids.empty() || ids.type() != CV_8U || ids.rows != rows || ids.cols != cols) {
        throw std::runtime_error("seg-mask: label map must be 8-bit at pipeline geometry");
    }

    std::vector<uint8_t> q;
    q.reserve(rates.size());
    for (float r : rates) q.push_back(cs_segmask_quantize_rate(std::clamp(r, 0.0f, 1.0f)));


    int crows, ccols;
    cs_chroma_dims(rows, cols, crows, ccols);
    // same_as_last (the anchored re-encode: this object's previous call had the same labels, rates and flags):
    // the draws, the sampled planes and the label PNG are that call's, kept in the object.
    if (!same_as_last) {
        // The Y / Cr / Cb planes, the label PNG and the chroma support do not need the luma draw: on a
        // side thread beside it.
        CsSideTask side;
        side.start([&] {
            cv::Mat ycc;
            cv::cvtColor(input_img, ycc, cv::COLOR_BGR2YCrCb);
            std::vector<cv::Mat> planes;
            cv::split(ycc, planes); // 0:Y 1:Cr 2:Cb to match COLOR_YCrCb2BGR order
            seg_y = planes[0];
            cv::resize(planes[1], seg_cr, cv::Size(ccols, crows), 0, 0, cv::INTER_AREA);
            cv::resize(planes[2], seg_cb, cv::Size(ccols, crows), 0, 0, cv::INTER_AREA);
            seg_png.clear();
            if (!cv::imencode(".png", ids, seg_png, { cv::IMWRITE_PNG_COMPRESSION, 6 }) || seg_png.empty() ||
                seg_png.size() > (static_cast<size_t>(1) << 24)) {
                throw std::runtime_error("seg-mask: label PNG encode failed");
            }
            chroma_support = cs_seg_chroma_support(ids, q);
        });
        // luma follows the segment budget; chroma is drawn on the 4:2:0 grid
        returnSegMaskIndices(ri_x, ri_y, rows, cols, ids, q, cs_sample_seed);
        side.wait();
        if (ri_x.empty()) {
            throw std::runtime_error("seg-mask: rates select zero samples (raise a rate)");
        }
        m = (int)ri_x.size();
        returnYcc420ChromaIndices(rows, cols, m, cs_sample_seed);
    }
    const int mY_written = (int)ri_x.size();
    const int mC = m_chroma;
    if (mY_written == 0 || seg_png.empty() || seg_y.empty())
        throw std::runtime_error("seg-mask: same_as_last without a previous call");
    std::vector<uint8_t> prefix;
    prefix.reserve(3 + 1 + q.size() + 4);
    prefix.insert(prefix.end(), {0, 2, 0});   // seg prefix version 2: chroma inside the labels
    prefix.push_back((uint8_t)q.size());
    prefix.insert(prefix.end(), q.begin(), q.end());
    prefix.resize(prefix.size() + 4);
    cs_put_le32(prefix.data() + prefix.size() - 4, (uint32_t)seg_png.size());
    const size_t pre_bytes = prefix.size() + seg_png.size();

    const size_t meas_bytes = (size_t)mY_written + (size_t)2 * mC;
    const int meas_pixels = (int)((meas_bytes + 2) / 3);
    const int pre_pixels = (int)((pre_bytes + 2) / 3);
    // Compact anchor payload (positions drawn from the sample seed).
    std::vector<uint8_t> anchor_payload;
    if (anchors) {
        anchor_payload = cs_anchor_payload(*anchors, cs_sample_seed);
        std::cout << "HR anchors: " << cs_get_le32(anchor_payload.data()) << " (" << anchor_payload.size() << " payload bytes)" << std::endl;
    }
    const int hr_pixels = (int)((anchor_payload.size() + 2) / 3);
    const int total = meas_pixels + CS_HEADER_PIXELS + pre_pixels + hr_pixels;
    uint8_t* buf = alloc_container(total);

    buf[CS_OFF_PAD] = (uint8_t)out_mode;
    std::memcpy(buf + CS_HEADER_BYTES, prefix.data(), prefix.size());
    std::memcpy(buf + CS_HEADER_BYTES + prefix.size(), seg_png.data(), seg_png.size());
    // the prefix byte length rides in the pad
    cs_put_le32(buf + CS_OFF_PAD + CS_OFF_SEGPRE_LEN, (uint32_t)pre_bytes);

    uint8_t* body = buf + CS_HEADER_BYTES + pre_bytes;
    cs_gather_u8(seg_y.data, seg_y.step, ri_x.data(), ri_y.data(), mY_written, body);
    cs_gather_u8(seg_cr.data, seg_cr.step, ri_cx1.data(), ri_cy1.data(), mC, body + mY_written);
    cs_gather_u8(seg_cb.data, seg_cb.step, ri_cx2.data(), ri_cy2.data(), mC, body + mY_written + mC);
    stamp_sample_bits(buf, sample_bits, chroma_bits);

    // Compact anchors ride as a trailing section after Cb; the
    // payload byte length sits in the pad after the mode byte (the seg
    // prefix length already holds CS_OFF_HR_COUNT's bytes).
    const size_t anchor_len = anchors ? anchor_payload.size() : 0;
    if (anchors) {
        cs_put_le32(buf + CS_OFF_PAD + 1, (uint32_t)anchor_payload.size());
        std::memcpy(body + mY_written + 2 * mC, anchor_payload.data(), anchor_payload.size());
    }

    const size_t y0 = (size_t)CS_HEADER_BYTES + pre_bytes;
    if (coded) {
        code_container_body(y0, mY_written, mC, crows, ccols, anchor_len, sample_bits, chroma_bits);
    }
    else {
        // Bit-packed (the anchor pass's in-memory container): prefix + label PNG stay raw; the three planes pack
        // independently; the anchor payload follows them as it is.
        encoded_img = cs_pack_container_body(encoded_img,
            cs_ycc_body_sections(y0, (size_t)mY_written, (size_t)mC, sample_bits, chroma_bits, anchor_len));
    }
    sampling_mode = out_mode;
}

void encode_image::encode(const std::vector<int>& ri_x_g, const std::vector<int>& ri_y_g) {
    m = (int)ri_x_g.size();
    uint8_t* buf = alloc_container(m + CS_HEADER_PIXELS);

    // write all m measurements (indices CS_HEADER_PIXELS .. CS_HEADER_PIXELS+m-1)
    cs_gather_bgr(input_img.data, input_img.step, ri_x_g.data(), ri_y_g.data(), m,
                  buf + (size_t)CS_HEADER_PIXELS * 3);
}

// The closed loop's anchor plan: `count` anchors in the 16x16 blocks where the decode `dec` is furthest from `ref`,
// their exact luma taken from `native` (all three at decode output geometry).
static CsAnchorPlan cs_closed_loop_plan(const cv::Mat& ref, const cv::Mat& dec, const cv::Mat& native, int count) {
    CsAnchorPlan plan;
    plan.count = count;
    plan.flags = cs_error_anchor_flags(ref, dec, count);
    cv::Mat ycc;
    cv::cvtColor(native, ycc, cv::COLOR_BGR2YCrCb);
    cv::extractChannel(ycc, plan.luma, 0);
    return plan;
}

// The s-x layer (s = 2..6) of a W x H photo is the photo area-shrunk by s / 2, sides even; the pipeline samples its
// bilinear half (the layer's LR).
static cv::Size ms_layer_size(int W, int H, int s) {
    const double f = s / 2.0;
    return cv::Size((int)(W / f) / 2 * 2, (int)(H / f) / 2 * 2);
}
static void ms_make_layer(const cv::Mat& src, int s, cv::Mat& layer, cv::Mat& lr) {
    const cv::Size size = ms_layer_size(src.cols, src.rows, s);
    if (s == 2) layer = src(cv::Rect(0, 0, size.width, size.height));
    else cv::resize(src, layer, size, 0, 0, cv::INTER_AREA);
    cv::resize(layer, lr, size / 2);
}
// The T px tile (of n along this axis) that holds the centre of pixel p of a grid `len` pixels long spanning the
// photo's `full` pixels.
static inline int ms_tile_of(int p, int len, int full, int T, int n) {
    return (std::min)(n - 1, (int)((p + 0.5) * full / len) / T);
}

// Multi-scale planner features, per 120 px tile and scale s = 2..6 (the layer as cs_encode_multiscale builds it:
// the photo area-shrunk by s/2, its LR = bilinear half): D = RGB MSE of the layer cubic-upscaled back to full size
// (the scale's floor; 0 at 2x), G = luma MSE between the layer and its LR bilinear-upscaled (detail cs-sr must
// recover), V = mean squared Laplacian of the LR luma (how hard the CS solve is). Tiles by pixel centre.
struct MsTileFeat { float D, G, V; };
static std::vector<MsTileFeat> ms_tile_features(const cv::Mat& src, int s, int T) {
    const int W = src.cols, H = src.rows, th = (H + T - 1) / T, tw = (W + T - 1) / T;
    cv::Mat layer, lr;
    ms_make_layer(src, s, layer, lr);
    std::vector<double> sd((size_t)th * tw, 0.0), sg((size_t)th * tw, 0.0), sv((size_t)th * tw, 0.0);
    std::vector<double> nd((size_t)th * tw, 0.0), ng((size_t)th * tw, 0.0), nv((size_t)th * tw, 0.0);
    // Per-pixel squared error, summed per tile; rows grouped by tile row so threads never share a tile.
    const auto per_tile = [&](const cv::Mat& e, std::vector<double>& sum, std::vector<double>& cnt) {
        const int eh = e.rows, ew = e.cols;
        std::vector<int> tx((size_t)ew);
        for (int x = 0; x < ew; ++x) tx[(size_t)x] = ms_tile_of(x, ew, W, T, tw);
        #pragma omp parallel for schedule(dynamic, 1)
        for (int r = 0; r < th; ++r) {
            for (int y = 0; y < eh; ++y) {
                if (ms_tile_of(y, eh, H, T, th) != r) continue;
                const float* p = e.ptr<float>(y);
                for (int x = 0; x < ew; ++x) { sum[(size_t)r * tw + tx[(size_t)x]] += p[x]; cnt[(size_t)r * tw + tx[(size_t)x]] += 1.0; }
            }
        }
    };
    if (s > 2) {
        cv::Mat up, d;
        cv::resize(layer, up, cv::Size(W, H), 0, 0, cv::INTER_CUBIC);
        cv::absdiff(up, src, d);
        d.convertTo(d, CV_32FC3);
        d = d.mul(d);
        cv::Mat ch[3];
        cv::split(d, ch);
        cv::Mat e = (ch[0] + ch[1] + ch[2]) / 3.0f;
        per_tile(e, sd, nd);
    }
    cv::Mat ly, ry;
    cv::cvtColor(layer, ly, cv::COLOR_BGR2GRAY);
    cv::cvtColor(lr, ry, cv::COLOR_BGR2GRAY);
    ly.convertTo(ly, CV_32F);
    ry.convertTo(ry, CV_32F);
    cv::Mat rup, eg, lap;
    cv::resize(ry, rup, ly.size(), 0, 0, cv::INTER_LINEAR);
    eg = ly - rup;
    eg = eg.mul(eg);
    per_tile(eg, sg, ng);
    cv::Laplacian(ry, lap, CV_32F, 1, 1.0, 0.0, cv::BORDER_REPLICATE);
    lap = lap.mul(lap);
    per_tile(lap, sv, nv);
    std::vector<MsTileFeat> out((size_t)th * tw);
    for (size_t t = 0; t < out.size(); ++t)
        out[t] = { (float)(nd[t] > 0 ? sd[t] / nd[t] : 0.0), (float)(ng[t] > 0 ? sg[t] / ng[t] : 0.0), (float)(nv[t] > 0 ? sv[t] / nv[t] : 0.0) };
    return out;
}

// A multi-scale layer's luma rate per LR pixel (lrw x lrh), x 255: its tile's where the tile is scale s, else the rate
// of a scale-s tile within kMsMargin px (full res; 8 neighbours), else 0. The margin lets the decoder's 17 px feather
// blend two sampled layers (IMG_0620 r0.3: -0.86 -> -0.20 dB vs production at equal bytes; 16 px fixes it fully but
// costs the other photos 0.15-0.4 dB).
constexpr int kMsMargin = 8;
static cv::Mat ms_layer_rates(const cv::Mat& map, const cv::Mat& qmap, int s, int W, int H, int T, int lrw, int lrh) {
    const int th = map.rows, tw = map.cols;
    cv::Mat q8(lrh, lrw, CV_8U);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < lrh; ++y) {
        const int ty = ms_tile_of(y, lrh, H, T, th);
        uint8_t* qr = q8.ptr<uint8_t>(y);
        for (int x = 0; x < lrw; ++x) {
            const int tx = ms_tile_of(x, lrw, W, T, tw);
            int q = map.at<uint8_t>(ty, tx) == s ? qmap.at<uint8_t>(ty, tx) : 0;
            if (q == 0) {
                const double X = (x + 0.5) * W / lrw, Y = (y + 0.5) * H / lrh;
                for (int k = 0; k < 8 && q == 0; ++k) {
                    static const int ox[8] = { -1, 1, 0, 0, -1, 1, -1, 1 }, oy[8] = { 0, 0, -1, 1, -1, -1, 1, 1 };
                    const int nx = (int)(X + ox[k] * kMsMargin), ny = (int)(Y + oy[k] * kMsMargin);
                    if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
                    if (map.at<uint8_t>(ny / T, nx / T) == s) q = qmap.at<uint8_t>(ny / T, nx / T);
                }
            }
            qr[x] = (uint8_t)q;
        }
    }
    return q8;
}

// Per-tile error model of the planner, per scale 2x..6x: MSE = D + exp(a . [1, lD, lG, lV, lG^2, lV^2, lG lV]
// + lq min(b . [1, lD, lG, lV], -0.05)), lX = log(1 + X) of ms_tile_features clamped to the training maximum, lq =
// log(rate): the scale's floor D plus a reconstruction error that falls with the rate. Fitted on 16 dataset photos
// (none of the evaluation set), each scale as a whole-photo layer at random rates held over 6x6-tile blocks (as a
// plan's regions; per-tile random rates let starved tiles borrow from their neighbours), production decode; held-out
// log error RMSE 0.56 (2x) / 0.40-0.43 (3x-6x).
struct MsErrModel { double a[7], b[4], lmax[3]; };
static const MsErrModel kMsModel[5] = {
    { { -1.04656, 0, 0.732277, 0.505001, -0.0323643, -0.0571419, 0.0693871 }, { 0.252447, 0, 0.316446, -0.277803 }, { 0, 7.05747, 10.1007 } },
    { { -0.685989, 0.156419, 0.989368, 0.11643, 0.110375, 0.0369118, -0.142217 }, { 0.427264, 0.00145417, 0.494178, -0.417393 }, { 6.18446, 7.0095, 9.75632 } },
    { { -0.565965, 0.173049, 0.975387, 0.0586176, 0.0448857, 0.0146763, -0.0620851 }, { 0.502667, -0.0198731, 0.560642, -0.471172 }, { 6.75201, 7.30921, 9.88583 } },
    { { -0.915025, 0.360086, 0.466349, 0.337929, 0.0130645, -0.0166716, 0.00610654 }, { 0.331764, 0.102952, 0.354456, -0.37899 }, { 7.21561, 6.98675, 9.93173 } },
    { { -1.10041, 0.325245, 0.567029, 0.312329, 0.0504057, -0.000978317, -0.0446536 }, { 0.159896, 0.113276, 0.305696, -0.337293 }, { 7.46319, 6.9594, 10.1227 } }
};

// The plan: per tile the (scale, rate) of least predicted error for the budget of a production container
// at --ratio (6-bit luma: 1.2 x ratio of the LR pixels, capped at 1, plus the anchors), by a Lagrangian over all
// tiles. Costs in luma samples: one carries 0.75 byte + its chroma share 0.5, an anchor ~1 byte, so an anchor = 0.8.
// The tiles' own samples leave out the margins; the plan is redone with the budget scaled by planned / counted
// samples (twice), which lands the container within a few % of production's bytes.
// A tile no LR pixel of the 2x layer falls in (a 1 px last row or column, e.g. a photo 241 px high) takes the plan
// of the tile it borders, whose layer covers it; a (tile, scale) with no LR pixel of that layer is never chosen
// (it would draw no samples, so a layer of only such tiles would be empty).
// coarse (--ms-coarse, 0..1): moves the plan's coarseness (the area mean of log(s / 2) / log 3: 0 = all 2x, 1 = all
// 6x) from the plain plan's c0 to c0 + coarse (1 - c0), at the same budget. A second multiplier mu (bisected) prices
// coarseness: each tile minimizes E - mu * area * log(s / 2) / log 3 + lambda * C, so the tiles that lose the least
// predicted error per step of coarseness go coarser first. 0 = the plain plan.
static void ms_auto_plan(const cv::Mat& src, int T, float ratio, int hr_detail, cv::Mat& map, cv::Mat& qmap, float coarse) {
    const int W = src.cols, H = src.rows, th = (H + T - 1) / T, tw = (W + T - 1) / T, nt = th * tw;
    constexpr int NQ = 20;
    double qs[NQ];
    for (int i = 0; i < NQ; ++i) qs[i] = std::exp(std::log(0.02) * (1.0 - i / (NQ - 1.0)));
    constexpr double kAnchor = 0.8;
    const double anchor_cost = hr_detail > 0 ? kAnchor * hr_detail / nt : 0.0;   // per 2x tile
    std::vector<double> E((size_t)nt * 5 * NQ), C((size_t)nt * 5 * NQ), P((size_t)nt * 5 * NQ);   // error, cost, coarseness
    std::vector<int> nb((size_t)nt, -1);   // the tile a tile without 2x LR pixels copies
    // The five scales' features are independent: 3x..6x on a thread each beside 2x.
    std::vector<MsTileFeat> feats[5];
    {
        CsSideTask side[5];
        for (int si = 1; si < 5; ++si) side[si].start([&, si] { feats[si] = ms_tile_features(src, si + 2, T); });
        feats[0] = ms_tile_features(src, 2, T);
        for (CsSideTask& t : side) t.wait();
    }
    for (int si = 0; si < 5; ++si) {
        const int s = si + 2;
        const std::vector<MsTileFeat>& ft = feats[si];
        const double ls = std::log(s / 2.0) / std::log(3.0);
        const cv::Size lr = ms_layer_size(W, H, s) / 2;
        std::vector<double> rows_in((size_t)th, 0.0), cols_in((size_t)tw, 0.0);   // LR pixels per tile row / column
        for (int y = 0; y < lr.height; ++y) rows_in[(size_t)ms_tile_of(y, lr.height, H, T, th)] += 1.0;
        for (int x = 0; x < lr.width; ++x) cols_in[(size_t)ms_tile_of(x, lr.width, W, T, tw)] += 1.0;
        if (s == 2)
            for (int t = 0; t < nt; ++t) {
                const int ty = t / tw, tx = t % tw, dy = rows_in[(size_t)ty] == 0.0, dx = cols_in[(size_t)tx] == 0.0;
                if ((dy && ty > 0) || (dx && tx > 0)) nb[(size_t)t] = (ty - (dy && ty > 0)) * tw + tx - (dx && tx > 0);
            }
        for (int t = 0; t < nt; ++t) {
            const int ty = t / tw, tx = t % tw;
            const double npx = rows_in[(size_t)ty] * cols_in[(size_t)tx];
            const double full = (double)(std::min)(T, H - ty * T) * (std::min)(T, W - tx * T);
            const MsErrModel& m = kMsModel[si];
            const double lD = (std::min)(m.lmax[0], std::log1p((double)ft[(size_t)t].D)), lG = (std::min)(m.lmax[1], std::log1p((double)ft[(size_t)t].G)),
                         lV = (std::min)(m.lmax[2], std::log1p((double)ft[(size_t)t].V));
            const double base = m.a[0] + m.a[1] * lD + m.a[2] * lG + m.a[3] * lV + m.a[4] * lG * lG + m.a[5] * lV * lV + m.a[6] * lG * lV;
            const double slope = (std::min)(-0.05, m.b[0] + m.b[1] * lD + m.b[2] * lG + m.b[3] * lV);
            for (int i = 0; i < NQ; ++i) {
                const size_t o = ((size_t)t * 5 + si) * NQ + i;
                E[o] = npx > 0.0 ? full * (ft[(size_t)t].D + std::exp(base + slope * std::log(qs[i])))
                                 : 1e300;   // never below the 1e300 start of choose (no infinity under /fp:fast)
                P[o] = full * ls;
                C[o] = qs[i] * npx + (s == 2 ? anchor_cost : 0.0);
            }
        }
    }
    const double budget = (std::min)(1.0, 1.2 * ratio) * (double)(W / 2) * (H / 2) + kAnchor * hr_detail;
    std::vector<int> pick((size_t)nt);
    double mu = 0.0;   // price of coarseness (--ms-coarse)
    const auto choose = [&](double lam) {
        double cost = 0.0;
        for (int t = 0; t < nt; ++t) {
            double best = 1e300; int bo = 0;
            if (nb[(size_t)t] >= 0) bo = pick[(size_t)nb[(size_t)t]];   // an earlier tile
            else
                for (int o = 0; o < 5 * NQ; ++o) {
                    const size_t k = (size_t)t * 5 * NQ + o;
                    const double v = E[k] - mu * P[k] + lam * C[k];
                    if (v < best) { best = v; bo = o; }
                }
            pick[(size_t)t] = bo;
            cost += C[(size_t)t * 5 * NQ + bo];
        }
        return cost;
    };
    const auto fit_budget = [&](double target) {   // lambda for the budget
        double lo = 1e-6, hi = 1e6;   // cost falls as lambda grows
        for (int it = 0; it < 60; ++it) {
            const double mid = std::sqrt(lo * hi);
            if (choose(mid) > target) lo = mid; else hi = mid;
        }
        return choose(hi);
    };
    const auto coarseness = [&]() {
        double c = 0.0, a = 0.0;
        for (int t = 0; t < nt; ++t) {
            c += P[(size_t)t * 5 * NQ + pick[(size_t)t]];
            a += P[(size_t)t * 5 * NQ + 4 * NQ];   // P at 6x = the tile's area
        }
        return c / a;
    };
    double c0 = -1.0, c1 = -1.0;   // the plain plan's coarseness, the target
    map.create(th, tw, CV_8U);
    qmap.create(th, tw, CV_8U);
    int count[7] = {};
    double target = budget, cost = 0.0, counted = 0.0;
    for (int pass = 0; pass < 3; ++pass) {
        mu = 0.0;
        cost = fit_budget(target);
        if (coarse > 0.0f) {
            if (c0 < 0.0) { c0 = coarseness(); c1 = c0 + coarse * (1.0 - c0); }
            double lo = 1e-4, hi = 1e5;   // coarseness grows with mu
            for (int it = 0; it < 40; ++it) {
                mu = std::sqrt(lo * hi);
                fit_budget(target);
                if (coarseness() < c1) lo = mu; else hi = mu;
            }
            mu = hi;
            cost = fit_budget(target);
        }
        std::fill(count, count + 7, 0);
        for (int t = 0; t < nt; ++t) {
            const int si = pick[(size_t)t] / NQ, i = pick[(size_t)t] % NQ;
            map.at<uint8_t>(t / tw, t % tw) = (uint8_t)(si + 2);
            qmap.at<uint8_t>(t / tw, t % tw) = (uint8_t)(std::min)(254.0, (std::max)(1.0, std::round(qs[i] * 255.0)));
            ++count[si + 2];
        }
        counted = kAnchor * hr_detail * count[2] / nt;   // the samples the layers will draw, margins included
        for (int s = 2; s <= 6; ++s) {
            if (!count[s]) continue;
            const cv::Size lr = ms_layer_size(W, H, s) / 2;
            const cv::Mat q8 = ms_layer_rates(map, qmap, s, W, H, T, lr.width, lr.height);
            counted += cv::sum(q8)[0] / 255.0;
        }
        if (std::abs(counted / budget - 1.0) < 0.01) break;
        target *= budget / counted;
    }
    if (coarse > 0.0f)
        std::printf("multi-scale plan: --ms-coarse %.2f: coarseness %.3f -> %.3f (target %.3f)\n", coarse, c0, coarseness(), c1);
    std::printf("multi-scale plan: budget %.0f luma samples, planned %.0f, with margins %.0f; tiles 2x %d, 3x %d, 4x %d, 5x %d, 6x %d\n",
        budget, cost, counted, count[2], count[3], count[4], count[5], count[6]);
}

// A layer's seg labels from its rate map (ms_layer_rates): one label per distinct rate, label 0 = rate 0 (the
// pixels outside the layer's tiles and their margin). rates receives each label's rate.
static cv::Mat ms_layer_labels(const cv::Mat& q8, std::vector<float>& rates) {
    cv::Mat ids(q8.rows, q8.cols, CV_8U);
    int label_of[256];
    std::fill(label_of, label_of + 256, -1);
    rates = { 0.0f };
    label_of[0] = 0;
    for (int y = 0; y < q8.rows; ++y) {
        const uint8_t* qr = q8.ptr<uint8_t>(y);
        uint8_t* id = ids.ptr<uint8_t>(y);
        for (int x = 0; x < q8.cols; ++x) {
            const int q = qr[x];
            if (label_of[q] < 0) { label_of[q] = (int)rates.size(); rates.push_back(q / 255.0f); }
            id[x] = (uint8_t)label_of[q];
        }
    }
    return ids;
}

// The outer container (container_format.hpp): the header, the descriptor [version][W][H][T][layer count]
// [scale, byte length per layer][map PNG length][map PNG], then the layers' containers.
static cv::Mat ms_assemble(const cv::Mat& map, int W, int H, int T, const std::vector<uint8_t>& scales,
    const std::vector<cv::Mat>& layers) {
    std::vector<uint8_t> png;
    cv::imencode(".png", map, png, { cv::IMWRITE_PNG_COMPRESSION, 9 });
    std::vector<uint8_t> desc;
    const auto put = [&](uint32_t v, int n) { for (int b = 0; b < n; ++b) desc.push_back((uint8_t)(v >> (8 * b))); };
    put(1, 1); put(W, 2); put(H, 2); put(T, 2); put((uint32_t)layers.size(), 1);
    for (size_t i = 0; i < layers.size(); ++i) { put(scales[i], 1); put((uint32_t)(layers[i].total() * 3), 4); }
    put((uint32_t)png.size(), 4);
    desc.insert(desc.end(), png.begin(), png.end());
    size_t body = desc.size();
    for (const cv::Mat& c : layers) body += c.total() * 3;
    const int total = CS_HEADER_PIXELS + (int)((body + 2) / 3);
    cv::Mat out(1, total, CV_8UC3, cv::Scalar(0, 0, 0));
    uint8_t* buf = out.data;
    if (!cs_write_header(buf, (size_t)total * 3, cs_build_metadata((int)layers.size(), H, W, H, W)))
        throw std::runtime_error("multi-scale: header write failed");
    buf[CS_OFF_PAD] = CS_MODE_MULTISCALE;
    const size_t desc_len = desc.size();
    cs_put_le32(buf + CS_OFF_PAD + CS_OFF_SEGPRE_LEN, (uint32_t)desc_len);
    std::memcpy(buf + CS_HEADER_BYTES, desc.data(), desc_len);
    size_t off = CS_HEADER_BYTES + desc_len;
    for (const cv::Mat& c : layers) { std::memcpy(buf + off, c.data, c.total() * 3); off += c.total() * 3; }
    return out;
}

int cs_encode_multiscale(const std::string& input_path, const std::string& output_path,
    int hr_detail, const std::function<cv::Mat(const cv::Mat&)>& anchor_decode, float ratio, bool show_mask,
    bool mask_window, float coarse) {
    constexpr int T = 120;
    try {
        if (!(ratio >= 0.001f && ratio <= 1.0f)) throw std::runtime_error("--ratio must be in [0.001, 1]");
        if (!(coarse >= 0.0f && coarse <= 1.0f)) throw std::runtime_error("--ms-coarse must be in [0, 1]");
        if (hr_detail < 1 || hr_detail > CS_HRDETAIL_MAX_SAMPLES || !anchor_decode)
            throw std::runtime_error("bad anchor count or no anchor decoder (internal error)");
        const cv::Mat src = cv::imread(input_path, cv::IMREAD_COLOR);
        if (src.empty()) throw std::runtime_error("cannot read '" + input_path + "'");
        const int W = src.cols, H = src.rows, th = (H + T - 1) / T, tw = (W + T - 1) / T;
        if (W > 65535 || H > 65535) throw std::runtime_error("multi-scale: image wider or taller than 65535");
        cv::Mat map(th, tw, CV_8U), qmap(th, tw, CV_8U);
        ms_auto_plan(src, T, ratio, hr_detail, map, qmap, coarse);
        // One seg-mask layer per scale in use: the photo pre-shrunk by s/2 (area), then the pipeline's 2x
        // (bilinear); labels = the tile's quantized rate where the tile is this
        // scale's, rate 0 elsewhere; chroma drawn inside that support (seg prefix version 2).
        std::vector<uint8_t> scales;
        std::vector<cv::Mat> layers;
        const int gw = W / 2, gh = H / 2;   // --show-mask: the 2x layer's sampling grid
        std::vector<int> mask_r, mask_c;
        for (int s = 2; s <= 6; ++s) {
            if (cv::countNonZero(map == s) == 0) continue;
            cv::Mat layer, lr;
            ms_make_layer(src, s, layer, lr);
            const int lw = layer.cols, lh = layer.rows;
            std::vector<float> rates;
            const cv::Mat ids = ms_layer_labels(ms_layer_rates(map, qmap, s, W, H, T, lr.cols, lr.rows), rates);
            encode_image e(lr, true);
            // The 2x layer: production's closed loop (decode, flag the worst 16x16 blocks, re-encode with the same
            // samples plus anchors there), the flags limited to its own tiles (the others decode flat).
            const int n2 = s == 2 ? cv::countNonZero(map == 2) : 0;
            const bool anchored = n2 > 0;
            e.encode_ycc420_segmask_ids(ids, rates, nullptr, !anchored);   // the anchor pass's container is decoded here and replaced below
            if (anchored) {
                const cv::Mat dec = e.decode_own_container(anchor_decode);
                if (dec.empty() || dec.type() != CV_8UC3 || dec.cols != lw || dec.rows != lh)
                    throw std::runtime_error("multi-scale: anchor pass decode failed or has the wrong geometry");
                cv::Mat ref = layer.clone();
                for (int y = 0; y < lh; ++y)
                    for (int x = 0; x < lw; ++x)
                        if (map.at<uint8_t>(y / T, x / T) != 2) ref.at<cv::Vec3b>(y, x) = dec.at<cv::Vec3b>(y, x);
                const CsAnchorPlan ap = cs_closed_loop_plan(ref, dec, layer,
                    (int)std::lround((double)hr_detail * n2 / ((double)th * tw)));
                std::cout << "multi-scale: 2x layer anchors " << ap.count << ", " << cv::countNonZero(ap.flags)
                          << " blocks flagged" << std::endl;
                e.encode_ycc420_segmask_ids(ids, rates, &ap, true, true);
            }
            if (show_mask) {   // this layer's LR pixel (r, c) covers s x s photo pixels: its centre on the grid
                std::vector<int> rr, cc;
                e.sample_positions(rr, cc);
                for (size_t k = 0; k < rr.size(); ++k) {
                    mask_r.push_back((std::min)(gh - 1, (int)((rr[k] + 0.5) * s / 2.0)));
                    mask_c.push_back((std::min)(gw - 1, (int)((cc[k] + 0.5) * s / 2.0)));
                }
            }
            cv::Mat c = e.get_mat();
            if (!c.isContinuous()) c = c.clone();
            std::cout << "multi-scale: " << s << "x layer " << lw << "x" << lh << ", " << cv::countNonZero(map == s)
                      << " tiles, " << rates.size() << " labels, " << c.total() * 3 << " bytes" << std::endl;
            scales.push_back((uint8_t)s);
            layers.push_back(c);
        }
        const cv::Mat out = ms_assemble(map, W, H, T, scales, layers);
        if (!cs_write_container_file(output_path, out.data, out.total() * 3)) throw std::runtime_error("failed to write '" + output_path + "'");
        std::cout << "multi-scale: " << layers.size() << " layers, container " << out.total() * 3 << " bytes" << std::endl;
        if (show_mask) {   // on the photo shrunk as the 2x layer is (crop to even, then the pipeline's bilinear 2x)
            cv::Mat grid, view;
            cv::resize(src(cv::Rect(0, 0, gw * 2, gh * 2)), grid, cv::Size(gw, gh));
            if (cs_sampling_mask_view(grid, mask_r, mask_c, view)) cs_present_mask_view(view, output_path, mask_window);
            else std::cerr << "Warning: --show-mask: no sampling indices available\n";
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
