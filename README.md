# Compressive-sensing image compression (CPU)

A photo is stored as a sparse set of pixel samples (luma at full resolution, Cr / Cb on the 4:2:0 grid) plus a few
exact native-luma "anchor" pixels placed on the blocks the decoder gets most wrong. The decoder reconstructs the
photo tile by tile with reweighted-L1 FISTA in the DCT basis, then upscales each tile 2x with a PnP-ADMM stage
(`cs-sr`) that also honours the anchors. This branch (`production-cpu`) is the production format only, CPU only,
with no learned (neural-network) stage.

## Formats

- **Multi-scale container (mode 20, default):** each 120 px tile is encoded at its own scale (2x..6x), planned for
  the byte budget of `--ratio`; every scale is a layer (mode 11), decoded separately and composited.
- **Single-scale seg8-tile container (mode 19, `--multiscale off`):** k-means scene classes, each split into 8
  detail levels of 64x64 tiles with their own sampling rate.

Container v4 (`container_format.hpp`): a 48-byte header (version 4, image geometry, sampling mode, seg prefix and
anchor payload lengths, sample bits, sample-coding flag), then the body. No password, key or encryption: sample
positions come from MT19937 seeded with the fixed public seed `cs_sample_seed`, so encodes are deterministic. Luma
samples are 6-bit and chroma 8-bit by default, and the samples are entropy-coded (`docs/sample-coding.md`). A
container file holds only the container bytes (name it `.cs`). The commands keep the names `encrypt` / `decrypt`.

## Build (Windows)

```
.\build_cpu.bat          # ImgReconstruct_backend.exe and cs_tests.exe in x64\Cpu
.\build_cpu.bat main     # the exe only
```

Needs LLVM 23 `clang-cl`, the MSVC tools (VS 2022 or VS 18 Build Tools), OpenCV 4.90 at `C:\opencv` for the exe,
and the static OpenCV 4.13 set with contrib (`quality`) for `cs_tests.exe` (paths at the top of the script).

## Usage

```
ImgReconstruct_backend encrypt photo.png photo.cs --seg-auto 8 --ycc420 --seg-tile --ratio 0.3 --hr-detail 450000
set CS_ADMM_PROX_OFF=1
ImgReconstruct_backend decrypt photo.cs photo.dec.png --tv 0 --photo-upscaler cs-sr --sr-pnp admm --sr-pnp-iters 2 --hr-weight 1
ImgReconstruct_backend compare photo.png photo.dec.png report --amp 4
```

- `CS_ADMM_PROX_OFF=1` (decode) runs the ADMM stage with an identity prox; the decoder then scales its automatic
  regularization x0.2. It is the production decode.
- Batch: `encrypt|decrypt @pairs.tsv [options]` with one `input<TAB>output` per line, one process for all.
- Running the exe without arguments prints every option.

## Layout

- `ImgReconstruct_backend/`: the CLI (`ImgReconstruct_backend.cpp`), the encoder (`image_encryption`,
  `CS_encryption`), the decoder (`image_decryption`), the FISTA solvers and helpers (`helper_functions`,
  `fista_kernels.hpp`), the 2x stage (`superres.cpp`), the container format, the sample coder, tiling and the
  preview window. `avir*.h` and `lancir.h` are the bundled AVIR / LANCIR resizers.
- `tests/`: `cs_tests` (header and format, tiling, sample bits and coding, segmentation, anchors, production
  roundtrips).
- `docs/`: the sample coder and the flat CPU tile loop.
