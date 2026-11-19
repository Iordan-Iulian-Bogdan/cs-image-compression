# Entropy coding of the stored samples (`--sample-coding`)

Status: the encode default since 2026-11-18 (`--no-sample-coding` writes the bit-packed body instead); opt-in
`--sample-coding` from 2026-11-13 until then. Made the default when the container format dropped its encryption
and PNG wrapper (v4): the wrapper's deflate had shrunk bit-packed containers by ~10%, so
raw bit-packed files grew by that much, while coded ones were already incompressible. Decode needs no flag. Lossless: a coded container
carries exactly the samples of the bit-packed container written with the same key, so the decode is the same.

## Why

The production containers store every sample value raw (luma 6 bits, Cr / Cb 8 bits) in the order of the key's
draw. The PNG wrapper's deflate then sees values in random order and removes only 5-12%. But the decoder
regenerates the sample positions from the key before it reads any value, so both sides know where every sample
sits, and neighbouring samples of a photo are strongly correlated. Coding each value against a prediction from
the already decoded samples around it costs 2-3 bits for luma and 1.5-2.5 for chroma.

## How it works

`ImgReconstruct_backend/sample_coder.{hpp,cpp}`: `cs_plane_encode` / `cs_plane_decode`, one call per plane
(the luma, Cr and Cb of a seg layer). Inputs are the plane of quantized values, the 0/1 sample mask and the bit
depth.

1. **Passes.** A sample's pass is the 8x8 Bayer rank of its position (0..63), in 11 groups: ranks below
   1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64. The first pass holds 1/64 of the positions; at full density the
   passes below 16 are the regular 8-, 4- and 2-pixel lattices. The rank is public: it depends on the position
   only.
2. **Prediction.** A sample of pass p is predicted from the sampled positions of the passes before p: the up to
   6 nearest (candidate offsets precomputed per pass and position phase, nearest first, at most 128; the scan
   stops early once 3 are found and the next is more than 3x as far as the nearest), weighted by 1 / distance^2.
   All integer arithmetic (weights 2^20 / d^2, 64-bit sums, prediction in 1/16 levels), so every build decodes
   the same values. The first pass is DPCM along the scan.
3. **Context.** One of 74: the distance class of the nearest neighbour (d^2 <= 1, 2, 5, 16, more; fewer than
   3 neighbours is its own class) x 12 bins of the weighted mean absolute deviation of the neighbours around
   the prediction, plus one context for the first pass and one for "no neighbour".
4. **Residual.** value - prediction, wrapped to the bit depth and zigzag-mapped, so the alphabet is 2^bits.
5. **Entropy coder.** Static rANS (32-bit state, byte renormalisation, 12-bit frequencies), one stream per
   plane. The encoder computes all residuals and contexts (parallel within a pass), builds a histogram per
   context and writes the tables ahead of the stream: per context a u16 count, then 12-bit frequencies.
   The decoder interleaves: per pass, predictions and contexts in parallel, then the serial rANS read.
6. **Fallback.** A plane that codes larger than its bit-packed size (very few samples, where the tables
   dominate, or noise) is stored bit-packed in scan order. One type byte heads either form, so a stream is
   never more than a byte above the packed size.

## Container

- `CS_OFF_SAMPLE_CODING` (pad offset 5, unused by the seg modes 11 / 19) = 1 marks a coded body.
- Body after the seg prefix: `[u32 luma bytes][u32 Cr bytes][u32 Cb bytes][luma][Cr][Cb]`, then the anchor payload
  exactly as before. (Until the v4 format of 2026-11-18 the blob was AES-256-CTR sealed with the domain tag
  `CS_CODED_TAG`; v4 has no encryption.)
- The streams hold the values quantized to the stamped sample bits (6 / 8 by default). The decoder
  (`decrypt_image::decode_coded_measurements`) decodes the three planes and rebuilds the raw layout the
  bit-packed path produces (dequantized values in draw order, then the anchor payload), so everything after
  `get_sampled_ycc420` is untouched.
- Multi-scale (mode 20): each layer is its own seg container and is coded independently; the outer descriptor
  already carries each layer's byte length.
- The anchor pass (encode, decode, flag blocks, re-encode with the same key) writes its first container
  bit-packed; only the final one is coded.
- Encoder: `encrypt_image::code_container_body`, reached with `coded = true` in `encrypt_ycc420_segmask_ids`.
  CLI: `--sample-coding` on `encrypt` (single- and multi-scale, batch mode included).

Not coded: the anchors (8-bit native luma in high-error blocks). They are the hardest pixels of the photo and
would need the 2x2 means around them as side information.
