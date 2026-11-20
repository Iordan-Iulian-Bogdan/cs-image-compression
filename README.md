# Compressive-sensing image compression (CPU)

A photo is stored as a sparse set of pixel samples (luma at full resolution, Cr / Cb on the 4:2:0 grid) plus a few
exact native-luma "anchor" pixels placed on the blocks the decoder gets most wrong. The decoder reconstructs the
photo tile by tile with reweighted-L1 FISTA in the DCT basis, then upscales each tile 2x with a PnP-ADMM stage
(`cs-sr`) that also honours the anchors. CPU only, no learned (neural-network) stage.

## Format

The encoder writes a multi-scale container (mode 20): each 120 px tile is coded at its own scale (2x..6x), planned
for the byte budget of `--ratio`: each tile's (scale, rate) options are priced in estimated coded bytes, from a model
of the bytes per sample by scale, rate and tile content (+0.85 / +0.49 / +0.22 dB at equal bytes at r0.05 / 0.3 / 0.5
over pricing in samples, 8 photos). Every scale in use is a layer (mode 11), decoded separately and composited; the 2x
layer carries the anchors (mode 19), placed by decoding its container in memory and flagging the worst 16x16 blocks.

Container v4 (`container_format.hpp`): a 48-byte header (version 4, image geometry, mode, prefix and anchor
payload lengths, sample bits, sample-coding flag), then the body. Sample positions come from MT19937 seeded with the
fixed public seed `cs_sample_seed`, so encodes are deterministic. Luma samples are 6-bit and chroma 8-bit, and the
samples are entropy-coded (`docs/sample-coding.md`). A container file holds only the container bytes (name it
`.cs`).

## Build (Windows)

```
.\build_cpu.bat          # ImgReconstruct_backend.exe and cs_tests.exe in x64\Cpu
.\build_cpu.bat main     # the exe only
```

Needs LLVM 23 `clang-cl`, the MSVC tools (VS 2022 or VS 18 Build Tools), OpenCV 4.90 at `C:\opencv` for the exe,
and the static OpenCV 4.13 set with contrib (`quality`) for `cs_tests.exe` (paths at the top of the script).

## Usage

```
ImgReconstruct_backend encode photo.png photo.cs --ratio 0.3
ImgReconstruct_backend decode photo.cs photo.dec.png
ImgReconstruct_backend compare photo.png photo.dec.png report --amp 4
```

- `--ratio` (default 1) is the byte budget as a sampling ratio in [0.001, 1]. The anchor count follows it.
- `--ms-coarse a` (encode, 0..1, default 0) pushes tiles to the coarse scales at the same budget: the plan's
  coarseness (area mean of log(s/2)/log 3, 0 = all 2x, 1 = all 6x) moves the fraction a of the way to all-6x.
- `--show-mask` (encode) writes `<out>.mask.png` (source | sample mask | overlay) and shows it in a window;
  `--no-preview` skips the windows (the decoder's live tile preview, the mask window).
- Batch: `encode|decode @pairs.tsv [options]` with one `input<TAB>output` per line, one process for all.

## Layout

- `ImgReconstruct_backend/`: the CLI (`ImgReconstruct_backend.cpp`), the encoder (`image_encoder`,
  `cs_sampler`), the decoder (`image_decoder`), the FISTA solver and helpers (`helper_functions`,
  `fista_kernels.hpp`), the 2x stage (`superres.cpp`), the container format, the sample coder, tiling and the
  preview window. `avir*.h` is the bundled AVIR resizer.
- `tests/`: `cs_tests` (header and format, tiling, sample bits and coding, anchors, solver, roundtrips).
- `docs/`: the sample coder and the flat CPU tile loop.
