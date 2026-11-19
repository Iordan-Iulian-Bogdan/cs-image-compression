#ifndef IMGRECONSTRUCT_BACKEND_PHOTO_UPSCALER_HPP_
#define IMGRECONSTRUCT_BACKEND_PHOTO_UPSCALER_HPP_

#include <string>

// Decode options of the 2x stage (cs-sr with PnP-ADMM, cs_sr_upscale_tile_2x in helper_functions.hpp).
struct CsPhotoUpscalerOptions {
  // Tile overlap of the automatic decrypt grid (the encoder's anchor pass: 8).
  int auto_overlap = 24;
  // ADMM outer iterations (--sr-pnp-iters, clamped [1,50]).
  int sr_pnp_iters = 8;
  // HR-anchor data-term weight for the cs-sr solve (0 = off: post-hoc
  // replacement only). Anchors are exact native luma, so 1.0 pulls each
  // anchor pixel at full weight beside the LR samples.
  float sr_hr_weight = 1.0f;
};

#endif  // IMGRECONSTRUCT_BACKEND_PHOTO_UPSCALER_HPP_
