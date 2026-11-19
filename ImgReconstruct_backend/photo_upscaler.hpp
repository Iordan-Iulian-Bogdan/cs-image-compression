#ifndef IMGRECONSTRUCT_BACKEND_PHOTO_UPSCALER_HPP_
#define IMGRECONSTRUCT_BACKEND_PHOTO_UPSCALER_HPP_

// Decode options of the tile grid and the 2x stage (cs-sr with PnP-ADMM, cs_sr_upscale_tile_2x in
// helper_functions.hpp). The defaults are the decoder's; the encoder's anchor pass sets its own.
struct CsPhotoUpscalerOptions {
  // Tile overlap of the automatic decode grid (the anchor pass: 8).
  int auto_overlap = 24;
  // ADMM outer iterations (the anchor pass: 8).
  int sr_pnp_iters = 2;
  // Identity ADMM prox instead of fastNlMeans; the luma coef is then scaled x0.2 (the anchor pass: off).
  bool prox_off = true;
};

#endif  // IMGRECONSTRUCT_BACKEND_PHOTO_UPSCALER_HPP_
