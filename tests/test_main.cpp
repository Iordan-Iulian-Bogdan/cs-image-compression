// Test suite for the CS image encoding backend.
// Covers: container header format,
// index shuffle determinism, tiling helpers, and a full encode->decode
// roundtrip scored by PSNR + SSIM.
#include "test_common.hpp"

int g_failures = 0;

int main() {
    // Large photos trip CL_MEM_OBJECT_ALLOCATION_FAILURE on this host's
    // OpenCL runtime during heavy GEMM; CPU path is deterministic for tests.
    cv::ocl::setUseOpenCL(false);
    test_header();
    test_shuffle();
    test_tile_helpers();
    test_roundtrip();
    test_odd_size_roundtrip();
    test_dct_adjoint();
    test_dct_weights();
    test_hr_detail();
    test_hr_sr_anchors();
    test_sample_bits();
    test_sample_coding();
    test_encoder_geometry();
    test_auto_reg_params();

    if (g_failures == 0) {
        std::printf("ALL TESTS PASSED\n");
        return 0;
    }
    std::printf("%d TEST(S) FAILED\n", g_failures);
    return 1;
}

