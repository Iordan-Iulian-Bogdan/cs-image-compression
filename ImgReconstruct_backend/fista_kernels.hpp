#pragma once
// Internal FISTA/DCT kernels shared by the solver (helper_functions.cpp) and the
// super-resolution stage (superres.cpp). Not part of the public backend API.
#include "helper_functions.hpp"

int cs_fista_inner_iters(int iterations, int fista_iters);
void cs_extract_channel_measurements(const cv::Mat& pixel_measurements, int channel, int m, std::vector<float>& b);
void cs_fista_core_single(float* x, const float* b, const int* rix, const int* riy, int m, int rows, int cols, const float* w, float lambda, int iters, const cs_dct_basis& bx, std::vector<float>& y, std::vector<float>& x_prev, std::vector<float>& grad, std::vector<float>& pix, std::vector<float>& sc);
