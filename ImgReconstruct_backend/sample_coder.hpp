#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

// Lossless entropy coder for one sampled plane (--sample-coding; docs/sample-coding.md).
// The sample positions are known to both sides (they regenerate from the key), so only the values are coded:
// the samples are split into 11 passes by the 8x8 Bayer rank of their position, each sample is predicted from
// the nearest samples of the earlier passes (inverse-square-distance weighted mean, integer arithmetic), and
// the wrapped residual is rANS-coded with a static histogram per context (nearest distance x local activity).
//   val  : rows x cols, row-major; values < 2^bits at the sampled positions (others ignored / left untouched)
//   mask : rows x cols, nonzero = sampled
// A plane that codes larger than its bit-packed size ships bit-packed instead (one type byte ahead of either form).
// The stream depends on the sample values only (no key).
std::vector<uint8_t> cs_plane_encode(const uint8_t* val, const uint8_t* mask, int rows, int cols, int bits);
// False on a malformed stream. val is written only at the sampled positions.
bool cs_plane_decode(const uint8_t* data, size_t len, const uint8_t* mask, int rows, int cols, int bits, uint8_t* val);
