# CPU tile solve: flat loop instead of the wavefront (2026-11-19, branch production-cpu; within 2 levels on < 0.003% of values)

What changed (`image_decryption.cpp`, `decrypt_tiles_ycc420`): the CPU LR tile solve ran in anti-diagonal waves
(2 (n - 1) + 1 barriers on an n x n grid) because `build_neighbor_warm_start` started each tile from the YCrCb of its
solved west / north neighbours' overlap strips. Since the sample fill (`fill_start`) replaced that start on every
plane with samples, the neighbour start only reached planes without samples (and the Cr -> Cb hand-off when Cb has
none). Now every tile starts from `build_generic_warm_start` (exactly the old function's no-neighbour branch: the
resized generic Y ref, flat 0.5 chroma, /10 DCT convention), no tile reads another tile's result, and all tiles run
in one `omp parallel for ... schedule(dynamic)`. No switch: the wavefront and the neighbour start are
gone.

Why it is correct: the only cross-tile read was the warm start, so the loop has no ordering dependency left (no
race: a tile writes only its own `mats_out` / `hr_out` entry, the latter under the existing mutex). The solve and
the cs-sr stage are unchanged; only planes that had no samples see a different start.

Workload and method: CPU build (`build_cpu.bat`), prox-off production decode (`--tv 0 --photo-upscaler cs-sr
--sr-pnp admm --sr-pnp-iters 2 --hr-weight 1 --device cpu`, `CS_ADMM_PROX_OFF=1`) of multi-scale containers
(`--seg-auto 8 --ycc420 --seg-tile --hr-detail 450000`, no `--enhance`) at r0.1 / 0.3 / 0.7 of IMG_3690 and 5
phone photos from a curated set (7.2 to 15.9 MP). One pinned exe
(`abflat_a.exe`) with a temporary `CS_FLAT_TILES=1` switch, each container decoded wf / flat / flat / wf with 0.5 s
pauses, wall time of the whole process. Scores: PSNR, SSIM (skimage), CIEDE2000 p95 against the source PNG.

| ratio | wavefront | flat | time | dPSNR | dSSIM | dDE95 |
| --- | --- | --- | --- | --- | --- | --- |
| r0.1 | 10757 ms | 9309 ms | -13.5% | +0.000 dB | -0.00000 | -0.0000 |
| r0.3 | 10745 ms | 9423 ms | -12.3% | +0.000 dB | -0.00000 | +0.0000 |
| r0.7 | 10839 ms | 9448 ms | -12.8% | +0.000 dB | +0.00000 | +0.0000 |

- Every one of the 18 decodes was faster (-5% to -18%). Output: 7 of 18 decodes pixel-identical; the rest differ by
  at most 2 levels on at most 0.0021% of the values (IMG_3690 r0.1: 751 values; r0.3: 138; r0.7: none). Equal to
  3 decimals in PSNR, 4 in SSIM and 3 in dE95 on every photo and ratio.
- Encode (its anchor pass decodes in memory through the same loop): IMG_3690 r0.3 multi-scale 9951 / 9968 ->
  7669 / 7502 ms (ABBA, -23%), and the container is byte-identical to the wavefront encoder's (the anchor ranking
  does not see the few changed pixels).
- The committed form (no switch) decodes byte-identically to the switch run on IMG_3690 r0.1, 00500 r0.3 and
  07000 r0.7. `cs_tests` passes.
