#pragma once
// Shared includes, check() and procedural test images for the backend test
// suite. Each tests/test_*.cpp file covers one topic; test_main.cpp runs them.
#include "container_format.hpp"
#include "image_encoder.hpp"
#include "image_decoder.hpp"
#include "quality_utils.hpp"

#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>
extern int g_failures;

inline void check(bool ok, const char* name) {
    if (ok) {
        std::printf("[PASS] %s\n", name);
    }
    else {
        std::printf("[FAIL] %s\n", name);
        g_failures++;
    }
    std::fflush(stdout); // crash diagnostics: never lose the tail to buffering
}

// procedural test image: smooth gradient + shapes (structure, no noise)
inline cv::Mat make_test_image(int w, int h) {
    cv::Mat img(h, w, CV_8UC3);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            img.at<cv::Vec3b>(y, x) = cv::Vec3b(
                (uchar)(128 + 60 * std::sin(x * 0.05)),
                (uchar)(80 + x * 80 / w),
                (uchar)(60 + y * 100 / h));
        }
    }
    cv::rectangle(img, cv::Rect(w / 4, h / 4, w / 2, h / 2), cv::Scalar(40, 30, 220), cv::FILLED);
    cv::circle(img, cv::Point(w / 2, h / 2), (w < h ? w : h) / 6, cv::Scalar(200, 220, 60), cv::FILLED);
    return img;
}

// Encode / decode as the CLI runs them: the multi-scale container (mode 20) with hr_detail anchors placed by the
// encoder's in-memory anchor pass, and the default decode.
inline int prod_encode(const std::string& in, const std::string& out, float ratio, int hr_detail = 2000,
    float coarse = 0.0f) {
    const std::function<cv::Mat(const cv::Mat&)> anchor_decode = [](const cv::Mat& container) {
        CsPhotoUpscalerOptions up;
        up.auto_overlap = 8;
        up.sr_pnp_iters = 8;
        up.prox_off = false;
        cv::Mat dec;
        if (decode_image::decode_image_tiled(container, "", false, 12, up, &dec) != 0) dec.release();
        return dec;
    };
    return cs_encode_multiscale(in, out, hr_detail, anchor_decode, ratio, false, false, coarse);
}

inline int prod_decode(const cv::Mat& container, const std::string& out) {
    return decode_image::decode_image_tiled(container, out, false, 0, CsPhotoUpscalerOptions());
}

// The container as the decoder reads it from disk.
inline cv::Mat read_container(const std::string& path) { return decode_image::read_container(path); }

// Test entry points, called in order from main() in test_main.cpp.
void test_header();
void test_shuffle();
void test_tile_helpers();
void test_roundtrip();
void test_odd_size_roundtrip();
void test_dct_adjoint();
void test_dct_weights();
void test_hr_detail();
void test_hr_sr_anchors();
void test_sample_bits();
void test_sample_coding();
void test_encoder_geometry();
void test_chroma_draw_v2();
void test_auto_reg_params();
