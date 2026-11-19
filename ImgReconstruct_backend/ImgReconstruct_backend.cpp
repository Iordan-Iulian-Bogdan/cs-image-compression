#include "image_decoder.hpp"
#include "image_encoder.hpp"
#include "photo_upscaler.hpp"
#include "quality_utils.hpp"
#include "cs_runtime.hpp"
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <string>
#include <vector>
#include <iostream>
#include <thread>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace {

void print_usage(const char* exe) {
    std::cout <<
        "Compressive-sensing image codec (CPU)\n"
        "\n"
        "Usage:\n"
        "  " << exe << " encode <photo> <out.cs> [--ratio r] [--show-mask] [--no-preview]\n"
        "  " << exe << " decode <in.cs> <out.png> [--no-preview]\n"
        "  " << exe << " encode|decode @<pairs.tsv> [options]   (batch: one 'input<TAB>output' per line, one process)\n"
        "  " << exe << " compare <original.png> <decoded.png> <out_prefix> [--amp <f>]\n"
        "\n"
        "  --ratio <r>     byte budget as a sampling ratio in [0.001, 1] (default 1)\n"
        "  --show-mask     write <out.cs>.mask.png (source | sample mask | overlay) and show it\n"
        "  --no-preview    no live tile preview window while decoding (no mask window while encoding)\n"
        "\n"
        "  compare scores decoding quality (PSNR/SSIM/MAE/max) and writes\n"
        "    <out_prefix>_sidebyside.png  original | decoded | JET error map\n"
        "    <out_prefix>_error.png       amplified absolute-error map\n"
        "\n"
        "Examples:\n"
        "  " << exe << " encode photo.png photo.cs --ratio 0.3\n"
        "  " << exe << " decode photo.cs photo.dec.png\n"
        "  " << exe << " compare photo.png photo.dec.png report --amp 4\n";
}

int run_compare(const std::string& original_path, const std::string& decoded_path,
                const std::string& out_prefix, double amplification)
{
    // full-size photos can exhaust OpenCL buffers in UMat paths
    cv::ocl::setUseOpenCL(false);
    cv::Mat original = cv::imread(original_path, cv::IMREAD_COLOR);
    if (original.empty()) {
        std::cerr << "Error: cannot load original '" << original_path << "'" << std::endl;
        return 1;
    }
    cv::Mat decoded = cv::imread(decoded_path, cv::IMREAD_COLOR);
    if (decoded.empty()) {
        std::cerr << "Error: cannot load decoded '" << decoded_path << "'" << std::endl;
        return 1;
    }
    if (decoded.size() != original.size()) {
        cv::resize(decoded, decoded, original.size());
    }

    const double p = cs_quality::psnr(original, decoded);
    const double s = cs_quality::ssim(original, decoded);
    const double mae = cs_quality::mean_abs_error(original, decoded);
    const double mx = cs_quality::max_abs_error(original, decoded);
    double edge_p = -1.0, edge_mae = -1.0, edge_share = 0.0;
    cs_quality::edge_scores(original, decoded, edge_p, edge_mae, edge_share);

    std::printf("PSNR: %.2f dB\n", p);
    if (s >= 0.0) std::printf("SSIM: %.4f\n", s);
    else          std::printf("SSIM: n/a (OpenCV quality module not available)\n");
    std::printf("MAE:  %.3f\n", mae);
    std::printf("MAX:  %.1f\n", mx);
    if (edge_p >= 0.0)
        std::printf("Edge-PSNR (top-10%% gradient, share %.3f): %.2f dB\n", edge_share, edge_p);
    if (edge_mae >= 0.0)
        std::printf("Edge-MAE: %.3f\n", edge_mae);

    const cv::Mat err = cs_quality::error_map(original, decoded, amplification);
    const cv::Mat strip = cs_quality::side_by_side(original, decoded, amplification);
    if (err.empty() || strip.empty()) {
        std::cerr << "Error: failed to build error visualization" << std::endl;
        return 1;
    }
    const std::string err_path = out_prefix + "_error.png";
    const std::string strip_path = out_prefix + "_sidebyside.png";
    if (!cv::imwrite(err_path, err)) {
        std::cerr << "Error: cannot write '" << err_path << "'" << std::endl;
        return 1;
    }
    if (!cv::imwrite(strip_path, strip)) {
        std::cerr << "Error: cannot write '" << strip_path << "'" << std::endl;
        return 1;
    }
    std::printf("Wrote %s\n", err_path.c_str());
    std::printf("Wrote %s\n", strip_path.c_str());
    return 0;
}

bool parse_float(const char* s, float& out) {
    try { size_t pos; float v = std::stof(s, &pos); if (pos != std::strlen(s)) return false; out = v; return true; }
    catch (...) { return false; }
}

} // namespace


int main(int argc, char* argv[])
{
    // OpenMP workers sleep between parallel regions instead of spinning: spinning workers took the cores the decode's
    // other threads (side preps, PNG write) need: -0.4 s per decode. An
    // OMP_WAIT_POLICY already set is kept. libomp has its own CRT and reads the OS environment block on first use, so
    // the variable goes there (_putenv_s would only change this CRT's copy).
    if (GetEnvironmentVariableA("OMP_WAIT_POLICY", nullptr, 0) == 0) SetEnvironmentVariableA("OMP_WAIT_POLICY", "PASSIVE");
    if (argc < 2) {
        print_usage(argv[0]);
        return 64; // usage error
    }

    const std::string mode = argv[1];
    if (mode != "encode" && mode != "decode" && mode != "compare") {
        std::cerr << "Error: unknown mode '" << mode << "'" << std::endl;
        print_usage(argv[0]);
        return 64;
    }

    if (mode == "compare") {
        if (argc < 5) {
            std::cerr << "Error: compare needs <original.png> <decoded.png> <out_prefix>" << std::endl;
            print_usage(argv[0]);
            return 64;
        }
        double amp = 0.0;
        for (int i = 5; i < argc; i++) {
            const std::string a = argv[i];
            if (a == "--amp" && i + 1 < argc) {
                amp = std::atof(argv[++i]);
            }
            else {
                std::cerr << "Error: unknown compare option '" << a << "'" << std::endl;
                print_usage(argv[0]);
                return 64;
            }
        }
        return run_compare(argv[2], argv[3], argv[4], amp);
    }

    // encode / decode @<pairs.tsv>: a batch of jobs in this process (no output path argument)
    const bool batch = argc >= 3 && argv[2][0] == '@';
    if (argc < (batch ? 3 : 4)) {
        std::cerr << "Error: missing input/output paths" << std::endl;
        print_usage(argv[0]);
        return 64;
    }
    const bool enc = mode == "encode";

    const std::string input = argv[2];
    const std::string output = batch ? std::string() : argv[3];

    bool show_preview = true, show_mask = false;
    float ratio = 1.0f;
    for (int i = batch ? 3 : 4; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--no-preview") show_preview = false;
        else if (enc && a == "--show-mask") show_mask = true;
        else if (enc && a == "--ratio") {
            if (i + 1 >= argc || !parse_float(argv[++i], ratio)) {
                std::cerr << "Error: --ratio requires a number" << std::endl;
                return 64;
            }
        }
        else {
            std::cerr << "Error: unknown " << (enc ? "encode" : "decode") << " option '" << a << "'" << std::endl;
            print_usage(argv[0]);
            return 64;
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    int rc = 0;

    try {
        // Batch (encode / decode @<pairs.tsv>): one process for the whole list. A failed job is reported and the batch goes on; rc = 1 if any failed.
        std::vector<std::pair<std::string, std::string>> jobs;
        if (batch) {
            std::ifstream lf(input.substr(1));
            if (!lf) { std::cerr << "Error: cannot read batch list '" << input.substr(1) << "'" << std::endl; return 64; }
            for (std::string line; std::getline(lf, line);) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.empty()) continue;
                const size_t tab = line.find('\t');
                if (tab == std::string::npos) { std::cerr << "Error: batch line without a tab: " << line << std::endl; return 64; }
                jobs.emplace_back(line.substr(0, tab), line.substr(tab + 1));
            }
        }
        const auto batch_line = [&](size_t j, int r, std::chrono::high_resolution_clock::time_point tj) {
            if (r != 0) std::cerr << "Error: job failed: " << jobs[j].first << std::endl;
            std::cout << "[batch] " << (j + 1) << "/" << jobs.size() << " " << jobs[j].second << " " <<
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - tj).count() << " ms" << std::endl;
        };
        if (enc) {
            // The anchor count: 450000 x0.5 up to ratio 0.5, x2 from 0.7, log-linear between (settings oracle,
            // 88 photos).
            const int hr_detail = (int)std::lround(450000.0 * (double)(0.5f * std::pow(4.0f, std::clamp((ratio - 0.5f) / 0.2f, 0.0f, 1.0f))));
            // The encoder decodes its own anchor-free container (cs-sr + PnP-ADMM with the NLM prox, 8 ADMM passes,
            // tile overlap 8 and 12 FISTA steps per pass: it only ranks 16x16 blocks by error) in memory, then
            // anchors the worst blocks. Flag map and anchor luma ride in the container.
            const std::function<cv::Mat(const cv::Mat&)> anchor_decode = [](const cv::Mat& container) {
                CsPhotoUpscalerOptions pass_up;
                pass_up.auto_overlap = 8;
                pass_up.sr_pnp_iters = 8;
                pass_up.prox_off = false;
                cv::Mat dec;
                if (decode_image::decode_image_tiled(container, "", false, 12, pass_up, &dec) != 0)
                    dec.release();
                return dec;
            };
            if (!batch) jobs.emplace_back(input, output);
            int failed = 0;
            for (size_t j = 0; j < jobs.size(); ++j) {
                const auto tj = std::chrono::high_resolution_clock::now();
                const int r = cs_encode_multiscale(jobs[j].first, jobs[j].second, hr_detail, anchor_decode, ratio,
                    show_mask && !batch, show_preview);
                if (!batch) { rc = r; break; }
                failed += r != 0;
                batch_line(j, r, tj);
            }
            if (batch) rc = failed ? 1 : 0;
        }
        else if (batch) {
            // A side thread reads the next container while the current one decodes.
            cv::Mat next;
            if (!jobs.empty()) next = decode_image::read_container(jobs[0].first);
            int failed = 0;
            for (size_t j = 0; j < jobs.size(); ++j) {
                const auto tj = std::chrono::high_resolution_clock::now();
                cv::Mat cur = std::move(next);
                std::thread reader;
                if (j + 1 < jobs.size()) reader = std::thread([&next, &jobs, j] { next = decode_image::read_container(jobs[j + 1].first); });
                const int r = cur.empty() ? -1 : decode_image::decode_image_tiled(cur, jobs[j].second,
                    false, 0, CsPhotoUpscalerOptions());
                if (reader.joinable()) reader.join();
                failed += r != 0;
                batch_line(j, r, tj);
            }
            rc = failed ? 1 : 0;
        }
        else {
            rc = decode_image::decode_image_tiled(input, output, show_preview, 0, CsPhotoUpscalerOptions());
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Unexpected error: " << e.what() << std::endl;
        return 3;
    }

    if (rc != 0) {
        std::cerr << (enc ? "Encoding" : "Decoding") << " failed (code " << rc << ")" << std::endl;
        return 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Execution time: " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " milliseconds" << std::endl;

    return 0;
}
