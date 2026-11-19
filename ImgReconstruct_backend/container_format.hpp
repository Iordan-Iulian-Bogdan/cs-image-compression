#ifndef CS_CONTAINER_FORMAT_HPP
#define CS_CONTAINER_FORMAT_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Container format v4 (compression only: no password, no encryption, no MAC)
//
// A container is a byte string, written to disk as-is (cs_write_container_file). In memory it is a 1 x N CV_8UC3
// Mat, so the measurement code keeps addressing it in 3-byte pixels.
//
// Header, 16 pixels = 48 bytes:
//   [0]      version byte (4)
//   [1..32]  metadata (32 bytes: "m|rows|cols|h|w" '-' padded, plain text)
//   [33..47] pad: sampling mode and its fields (CS_OFF_PAD + CS_OFF_*)
//
// Sample positions: every random draw (luma, chroma, anchors) is seeded from the fixed public cs_sample_seed, so
// both sides regenerate the same positions from the container alone. MT19937 via std::seed_seq from the 32 seed
// bytes; the Fisher-Yates shuffle seed is the first 4 seed bytes. Encodes are deterministic: the same photo and
// options give the same bytes.
//
// Older containers (v3: password, AES-CTR, HMAC, PNG wrapper) are not read; re-encode them from the photo.
// ---------------------------------------------------------------------------

#define CS_HEADER_PIXELS 16          // header size in container pixels
#define CS_HEADER_BYTES (CS_HEADER_PIXELS * 3)
#define CS_METADATA_BYTES 32
#define CS_VERSION 4                 // current container format version

// Offsets inside the 48-byte header
#define CS_OFF_VERSION 0
#define CS_OFF_META 1
#define CS_OFF_PAD 33                // 15 bytes: the highest field is CS_OFF_LOD_FULL (14)

// The fixed seed of every sample-position draw (not a secret).
extern const uint8_t cs_sample_seed[32];

// Builds the 32-byte '-'-padded metadata string "m|rows|cols|h|w", plus "|p" when odd_parity != 0
// (bit 0: native height odd, bit 1: native width odd; the encoder padded that side by one replicated
// row/column and decrypt crops it back). Throws std::runtime_error if the unpadded text exceeds 32 bytes.
std::string cs_build_metadata(int m, int rows, int cols, int org_h, int org_w, int odd_parity = 0);

// Writes the header into buf (the first CS_HEADER_BYTES bytes of a container): version, metadata, zeroed pad.
// Returns false if buf is too small or the metadata is not CS_METADATA_BYTES long.
bool cs_write_header(uint8_t* buf, size_t buf_bytes, const std::string& metadata);

// Container files: the container's bytes, nothing else.
bool cs_write_container_file(const std::string& path, const uint8_t* data, size_t bytes);
// Reads a container file; false if it cannot be read.
bool cs_read_container_file(const std::string& path, std::vector<uint8_t>& out);

// Sampling modes recorded in the pad region (CS_OFF_PAD)
#define CS_MODE_RANDOM    0            // uniform random global indices (pad = 0)
#define CS_MODE_PERIODIC  1            // one pattern repeated per tile (pad = 1)
#define CS_MODE_ADAPTIVE  2            // LOD-based per-tile counts (pad = 2)
#define CS_MODE_YCC420    3            // luma/chroma-split 4:2:0 sampling (pad = 3)
#define CS_MODE_HF_FOCUS  5            // encrypted thumbnail + HF-weighted BGR sampling
#define CS_MODE_YCC420_HF 6            // HF-weighted luma + uniform coarse chroma
#define CS_MODE_YCC422    7            // luma/chroma-split 4:2:2 sampling (pad = 7)
#define CS_MODE_YCC422_HF 8            // HF-weighted luma + uniform 4:2:2 chroma (pad = 8)
#define CS_MODE_BGR_HRDETAIL 4         // uniform random BGR + HR-detail payload (pad = 4)
#define CS_MODE_YCC420_HRDETAIL 9      // uniform YCC420 + HR-detail payload (pad = 9)
#define CS_MODE_SEGMASK 10             // per-segment BGR sampling from a label map (pad = 10)
#define CS_MODE_YCC420_SEG 11          // seg-mask luma + uniform 4:2:0 chroma (pad = 11)
#define CS_MODE_YCC422_SEG 12          // seg-mask luma + uniform 4:2:2 chroma (pad = 12)
#define CS_MODE_TCLUSTER 13            // random T-tetromino point clusters, BGR (pad = 13)
#define CS_MODE_TMEANS 14              // random T-tetromino mean groups, BGR (pad = 14)
#define CS_MODE_YCC420_TCLUSTER 15     // T-cluster luma + uniform 4:2:0 chroma (pad = 15)
#define CS_MODE_YCC420_TMEANS 16       // T-means luma + uniform 4:2:0 chroma (pad = 16)
#define CS_MODE_YCC422_TCLUSTER 17     // T-cluster luma + uniform 4:2:2 chroma (pad = 17)
#define CS_MODE_YCC422_TMEANS 18       // T-means luma + uniform 4:2:2 chroma (pad = 18)
#define CS_MODE_YCC420_SEG_HRDETAIL 19 // seg-mask luma + uniform 4:2:0 chroma + HR-detail payload (pad = 19)
// Multi-scale (pad = 20): the photo split into T x T tiles (full resolution), each tile coded at one scale
// s = 2..6 (the photo pre-shrunk by s/2 before the usual 2x pipeline). One complete seg-mask layer container
// (mode 11, seg prefix version 2) per scale in use carries that scale's tiles; the decoder decodes every layer,
// scales it to full size and blends the layers by the tile scale map. Body after the header:
//   [descriptor; byte length (u32) at CS_OFF_PAD + CS_OFF_SEGPRE_LEN]
//     [u8 version = 1][u16 W][u16 H][u16 T][u8 n]{[u8 scale][u32 layer bytes]} x n [u32 map bytes][map PNG]
//     (map: 8-bit, ceil(H / T) x ceil(W / T), the tile's scale)
//   [layer containers' bytes, in descriptor order (each a whole number of 3-byte pixels)]
#define CS_MODE_MULTISCALE 20

// Per-segment sampling (mode 10) body prefix, little-endian, right after the
// header:
//   [0]      nclasses (u8, 1..255)
//   [1..n]   per-class quantized rates (u8: count_c = round(pix_c * q/255))
//   [1+n..]  label-map PNG length (u32) with the --seg-blend width (0..128)
//            stashed in the top byte; the PNG itself is the low 24 bits
//   [...]    label-map PNG bytes (8-bit grayscale class ids, pipeline geometry)
// Measurements (3 bytes each, BGR) follow, padded to whole pixels. Rates are
// quantized AFTER the global --ratio scale on encrypt so both sides derive
// identical counts from the shipped bytes; decrypt needs no extra flags.
// The above is the legacy layout (including flat boosted bands). New smooth
// blends use [0][version=1][width][nclasses][rates][u32 PNG length][PNG].
// The zero marker is invalid to older readers, preventing silent mis-sampling.
// The prefix's byte length ([0] through the PNG) rides in the pad at CS_OFF_SEGPRE_LEN.
// Version 2 of the extended prefix ([0][2][width 0..128]...) also draws the
// chroma positions only inside the label map's support (chroma pixels whose
// luma block holds a label with rate > 0; cs_seg_chroma_support), for the
// layers of a multi-scale container; the chroma count is unchanged.
#define CS_SEGBLEND_MAX 128
struct CsSegPrefix {
    uint8_t nclasses = 0;
    std::vector<uint8_t> rates_q;
    uint32_t seglen = 0;
    uint8_t blend = 0;
    bool smooth_blend = false;
    bool chroma_masked = false;
    bool tcluster = false;   // luma drawn as T-tetromino groups (versions 3 and 4)
    size_t total = 0; // 1 + nclasses + 4 + seglen
};
inline bool cs_segmask_parse_prefix(const uint8_t* body, size_t body_bytes, CsSegPrefix& out) {
    out = CsSegPrefix{};
    if (body_bytes < 5) return false;
    // Extended prefix: [0][version=1|2|3|4][width][nclasses][rates][u32 PNG length].
    // A separate version keeps existing flat-band containers readable.
    const bool extended = body[0] == 0;
    if (extended && (body_bytes < 8 || (body[1] < 1 || body[1] > 4) || (body[1] == 1 && body[2] == 0) ||
                     body[2] > CS_SEGBLEND_MAX)) return false;
    out.chroma_masked = extended && (body[1] == 2 || body[1] == 3);
    out.tcluster = extended && body[1] >= 3;
    const size_t offset = extended ? 3 : 0;
    const uint8_t n = body[offset];
    if (n == 0) return false;
    const size_t head = offset + 1 + n + 4;
    if (body_bytes < head) return false;
    out.nclasses = n;
    out.rates_q.assign(body + offset + 1, body + offset + 1 + n);
    const uint8_t* len = body + offset + 1 + n;
    const uint32_t raw = (uint32_t)len[0] | ((uint32_t)len[1] << 8) |
        ((uint32_t)len[2] << 16) | ((uint32_t)len[3] << 24);
    out.smooth_blend = extended;
    out.blend = extended ? body[2] : (uint8_t)(raw >> 24);
    out.seglen = extended ? raw : raw & 0x00ffffffu;
    if (out.blend > CS_SEGBLEND_MAX) return false;
    if (out.seglen == 0 || out.seglen > (1u << 24)) return false;
    if (head + out.seglen > body_bytes) return false;
    out.total = head + out.seglen;
    return true;
}
// Quantize a [0,1] sampling rate to the shipped u8 (both sides derive counts
// from these bytes, so encrypt must quantize before drawing, not after).
inline uint8_t cs_segmask_quantize_rate(float r) {
    if (!(r > 0.0f)) return 0;
    if (r >= 1.0f) return 255;
    return (uint8_t)std::lround(r * 255.0f);
}

// HR-detail payload (--hr-detail): native-resolution luma anchors stored as a trailing body section (the compact
// flag map + luma of cs_anchor_payload in mode 19). Decrypt applies the anchors as exact luma data terms.
#define CS_HRDETAIL_SAMPLE_BYTES 6
#define CS_HRDETAIL_MAX_SAMPLES (1 << 24)   // 16.8 M: --hr-detail scales with the photo area (37 x 450k at 12.2 MP)
// Offset from CS_OFF_PAD for the modes 4/9 HR sample count (u32 LE,
// 1..CS_HRDETAIL_MAX_SAMPLES). Aliases the HF thumbnail-length field,
// which is unused outside modes 5/6/8.
#define CS_OFF_HR_COUNT 7

// One HR-detail anchor: linear index into the decrypt output geometry
// (row-major, output width stride) + native luma value.
struct CsHrSample {
    uint32_t idx = 0;
    uint8_t y = 0;
};

// Adaptive pad layout (mode = 2), all little-endian:
//   [0]      mode (= 2)
//   [1..2]   tile_size
//   [3..4]   lod byte count (sanity vs geometry)
//   [5..6]   weight_base (uint16): w = tile_pixels * (weight_base + lod)
//            0 is treated as the default 256 on parse (pre-strength containers)
#define CS_OFF_ADAPTIVE_BASE 5         // offset from CS_OFF_PAD
#define CS_OFF_HF_THUMBLEN 7          // modes 5/6/8: encrypted thumbnail length (u32 LE)
#define CS_OFF_SAMPLE_BITS 11         // offset from CS_OFF_PAD: bits per stored
                                      // sample value, 1..8 (0 = legacy 8-bit)
#define CS_OFF_SAMPLE_BITS_CHROMA 12  // chroma bits; 0 = legacy/same as luma
#define CS_OFF_SAMPLE_BITS_MARKER 13  // bit-packing extension marker
#define CS_SAMPLE_BITS_MARKER 0xA7
// --sample-coding (seg modes 11 / 19, which never read the adaptive base at this offset): 1 = the luma / Cr / Cb
// sections are entropy-coded (sample_coder.hpp) instead of bit-packed. Body after the seg prefix:
//   [u32 luma bytes][u32 Cr bytes][u32 Cb bytes][luma stream][Cr stream][Cb stream], then the anchor payload as
//   before. The streams hold the values quantized to the stamped sample bits, so the decoded samples equal the
//   bit-packed container's.
#define CS_OFF_SAMPLE_CODING 5
#define CS_OFF_LOD_FULL 14          // offset from CS_OFF_PAD: --lod-full
                                    // detail threshold 0..255 (0 = off)
#define CS_OFF_ADAPTIVE_CUTOFF 7    // offset from CS_OFF_PAD: --adaptive-cutoff
                                    // hard LOD floor 0..255 (0 = off). Aliases
                                    // the HF-thumbnail/HR-count u32 field,
                                    // unused by adaptive-family modes (2 and
                                    // LOD-luma mode 3); honored only from
                                    // stamped headers.
#define CS_OFF_SEGPRE_LEN 7         // offset from CS_OFF_PAD: seg prefix + label
                                    // PNG byte length (u32 LE) for modes 10/11/
                                    // 12. Aliases the HF-thumbnail/HR-count u32
                                    // slot, which seg modes never read (offsets
                                    // 1..4 must stay 0 for YCC-seg tile checks).


// Maps --adaptive-strength in [0, 1] to the integer weight_base.
//   0.0 -> 65535 (essentially uniform: density ratio ~1.004)
//   0.5 -> 256   (tuned default: density ratio ~1.8x)
//   1.0 -> 1     (max LOD bias: density ratio ~7.8x)
inline int cs_adaptive_base_from_strength(double strength) {
    if (!(strength > 0.0)) return 65535;
    if (strength >= 1.0) return 1;
    const double b = std::exp(std::log(65535.0) * (1.0 - strength));
    int v = (int)std::lround(b);
    if (v < 1) v = 1;
    if (v > 65535) v = 65535;
    return v;
}

// LOD contrast exponent (--adaptive-gamma): reshapes shipped detail bytes
// as lod_min + round(((v - lod_min) / (255 - lod_min))^gamma * (255 - lod_min)).
// gamma == 1 is identity; > 1 pushes mid-detail down (stronger edge focus,
// past what --adaptive-strength 1.0 + --adaptive-floor 0 can express);
// < 1 flattens toward uniform. Floor and ceiling are fixed points.
// Applied encrypt-side to the final shipped bytes (after scoring, region
// blend, and smoothing), so decrypt recomputes identical counts with no
// format change. No-op when gamma == 1 or the range is degenerate.
inline void cs_apply_lod_gamma(std::vector<uint8_t>& lod, int lod_min, double gamma) {
    if (!(gamma > 0.0) || gamma == 1.0) return;
    const double den = 255.0 - (double)lod_min;
    if (den <= 0.0) return;
    for (size_t i = 0; i < lod.size(); ++i) {
        if (lod[i] <= lod_min) continue; // floor fixed, sub-floor preserved
        const double t = ((double)lod[i] - (double)lod_min) / den;
        const double tc = t > 1.0 ? 1.0 : t;
        int v = lod_min + (int)std::lround(std::pow(tc, gamma) * den);
        if (v < 0) v = 0;
        if (v > 255) v = 255;
        lod[i] = (uint8_t)v;
    }
}

// lod region: tiles_x * tiles_y bytes after the header, padded to a
// whole number of container pixels so measurements stay pixel-aligned.
inline int cs_lod_tile_count(int rows, int cols, int tile_size) {
    if (tile_size <= 0) return 0;
    const int tx = (cols + tile_size - 1) / tile_size;
    const int ty = (rows + tile_size - 1) / tile_size;
    return tx * ty;
}
inline int cs_lod_bytes(int rows, int cols, int tile_size) {
    return cs_lod_tile_count(rows, cols, tile_size);
}
inline int cs_lod_pixels(int rows, int cols, int tile_size) {
    const int b = cs_lod_bytes(rows, cols, tile_size);
    return (b + 2) / 3;               // ceil(b / 3)
}

// YCC 4:2:0 split sampling (mode = 3): luma is sampled at full resolution
// (m_luma positions) while each chroma plane is first downsampled 2x and
// then sampled on its own coarse grid. The per-plane chroma count derives
// deterministically from the stored luma budget so no extra header fields
// are needed — both sides must use cs_chroma_count. Container body
// layout after the header is raw bytes:
//   [Y x m_luma][Cr x m_chroma][Cb x m_chroma], zero-padded to whole pixels.
// At the same --ratio this stores ~1.5 bytes/px instead of 3 (BGR modes).
//
// YCC 4:2:2 (modes 7/8) is the same construction with chroma downsampled
// horizontally only (full height, half width): ~2 bytes/px, sharper chroma
// edges than 4:2:0 at a larger container. cs_chroma_dims/cs_chroma_count
// parameterize all four split modes (3/6 halve both axes, 7/8 halve width
// only); the cs_ycc420_* names remain for older 4:2:0-only call sites.
inline void cs_chroma_dims(int mode, int rows, int cols, int& crows, int& ccols) {
    if (mode == CS_MODE_YCC422 || mode == CS_MODE_YCC422_HF ||
        mode == CS_MODE_YCC422_TCLUSTER || mode == CS_MODE_YCC422_TMEANS) {
        crows = rows;
        ccols = (cols + 1) / 2;
    } else {
        crows = (rows + 1) / 2;
        ccols = (cols + 1) / 2;
    }
}
inline int cs_chroma_count(int mode, int m_luma, int rows, int cols) {
    int crows, ccols;
    cs_chroma_dims(mode, rows, cols, crows, ccols);
    if (rows <= 0 || cols <= 0) return 0;
    int mc = (int)(((long long)m_luma * crows * ccols) / ((long long)rows * cols));
    if (mc < 1) mc = 1;
    const int cap = crows * ccols;
    if (mc > cap) mc = cap;
    return mc;
}
// Any luma/chroma-split container (uniform, adaptive-luma, HF-luma, or
// HR-detail).
inline bool cs_is_ycc(int mode) {
    return mode == CS_MODE_YCC420 || mode == CS_MODE_YCC422 ||
        mode == CS_MODE_YCC420_HF || mode == CS_MODE_YCC422_HF ||
        mode == CS_MODE_YCC420_HRDETAIL || mode == CS_MODE_YCC420_SEG_HRDETAIL ||
        mode == CS_MODE_YCC420_SEG || mode == CS_MODE_YCC422_SEG ||
        mode == CS_MODE_YCC420_TCLUSTER || mode == CS_MODE_YCC420_TMEANS ||
        mode == CS_MODE_YCC422_TCLUSTER || mode == CS_MODE_YCC422_TMEANS;
}
// T-tetromino sampling in any container (BGR modes 13/14, T-luma YCC
// modes 15-18). T-means modes need the averaging (weight-aware) solver.
inline bool cs_is_tcluster(int mode) {
    return mode == CS_MODE_TCLUSTER || mode == CS_MODE_YCC420_TCLUSTER ||
        mode == CS_MODE_YCC422_TCLUSTER;
}
inline bool cs_is_tmeans(int mode) {
    return mode == CS_MODE_TMEANS || mode == CS_MODE_YCC420_TMEANS ||
        mode == CS_MODE_YCC422_TMEANS;
}
// Any container carrying a trailing HR-detail payload.
inline bool cs_is_hrdetail(int mode) {
    return mode == CS_MODE_BGR_HRDETAIL || mode == CS_MODE_YCC420_HRDETAIL ||
        mode == CS_MODE_YCC420_SEG_HRDETAIL;
}
inline bool cs_is_ycc422(int mode) {
    return mode == CS_MODE_YCC422 || mode == CS_MODE_YCC422_HF ||
        mode == CS_MODE_YCC422_SEG || mode == CS_MODE_YCC422_TCLUSTER ||
        mode == CS_MODE_YCC422_TMEANS;
}
inline bool cs_is_ycc_hf(int mode) {
    return mode == CS_MODE_YCC420_HF || mode == CS_MODE_YCC422_HF;
}
// Domain separation for the three per-plane RNG streams: luma uses the seed
// as-is (tag 0), chroma planes use tagged copies. Deterministic on both sides.
inline void cs_ycc420_channel_seed(const uint8_t in[32], uint8_t tag, uint8_t out[32]) {
    for (int i = 0; i < 32; ++i) out[i] = in[i];
    out[0] ^= tag;
    out[16] ^= tag;
}
#define CS_YCC420_TAG_CR 0xA5
#define CS_YCC420_TAG_CB 0x5A

struct CsHeaderInfo {
    int m = 0, rows = 0, cols = 0, org_h = 0, org_w = 0;
    int odd_parity = 0;              // optional 6th metadata field (cs_build_metadata), 0 when absent
    int version = 0;                 // CS_VERSION
    bool legacy = false;             // not a v4 container (an older or unknown format)
    int sampling_mode = CS_MODE_RANDOM; // CS_MODE_* from the pad byte
    int periodic_tile = 0;           // > 0: tile size (periodic or adaptive)
    int periodic_samples = 0;        // mode 1: samples per tile; mode 2: lod byte count (sanity)
    int adaptive_base = 256;         // mode 2: weight_base (w = cap * (base + lod))
    uint32_t thumb_len = 0;          // modes 5/6: encrypted thumbnail bytes in body
    bool hr_detail = false;          // modes 4/9: trailing HR-detail payload present
    uint32_t hr_count = 0;           // modes 4/9: HR sample count (payload = 6 * count bytes)
    int sample_bits = 8;             // bits per stored sample value (pad+11, 0 = legacy 8)
    int sample_bits_chroma = 8;      // Cr/Cb (pad+12, 0 = same as luma)
    int full_threshold = 0;          // --lod-full detail guarantee (pad+14, 0 = off)
    int adaptive_cutoff = 0;         // --adaptive-cutoff hard floor (pad+7, 0 = off)
};

// Sample bit-depth reduction (--sample-bits): measurement values are
// plaintext in secret order, so the sequence is incompressible as stored —
// fewer bits per value is the structural size lever (besides ratio/planes).
// Quantize maps 8-bit values onto N bits (rounded), dequantize scales back;
// N = 8 is the identity in both directions. Pack/unpack lay values out
// MSB-first, each section independently whole-byte padded (pad bits zero).
inline int cs_sample_bits_valid(int bits) { return bits >= 1 && bits <= 8; }
inline size_t cs_packed_bytes(size_t count, int bits) {
    if (count == 0 || bits >= 8) return count;
    return (count * (size_t)bits + 7) / 8;
}
inline size_t cs_packed_bgr_bytes(size_t samples, int luma_bits, int chroma_bits) {
    return (samples * (size_t)(luma_bits + 2 * chroma_bits) + 7) / 8;
}
inline uint8_t cs_quantize_sample(uint8_t v, int bits) {
    if (bits >= 8) return v;
    const int maxv = (1 << bits) - 1;
    return (uint8_t)((v * maxv + 127) / 255);
}
inline uint8_t cs_dequantize_sample(uint8_t q, int bits) {
    if (bits >= 8) return q;
    const int maxv = (1 << bits) - 1;
    return (uint8_t)((q * 255 + maxv / 2) / maxv);
}

// Bit-level sample codec (MSB-first; out must hold cs_packed_bytes /
// cs_packed_bgr_bytes). The _bgr variants interleave per-value depths: G
// uses luma_bits, B/R use chroma_bits (equal depths collapse to the plain
// path, keeping the legacy stream bit-identical).
void cs_pack_samples(const uint8_t* in, size_t count, int bits, uint8_t* out);
void cs_unpack_samples(const uint8_t* in, size_t count, int bits, uint8_t* out);
void cs_pack_samples_bgr(const uint8_t* in, size_t samples, int luma_bits,
    int chroma_bits, uint8_t* out);
void cs_unpack_samples_bgr(const uint8_t* in, size_t samples, int luma_bits,
    int chroma_bits, uint8_t* out);

// One packed body section: raw byte offset + value count + plane depth(s).
// interleaved_bgr = flat BGR triplets (G = luma depth, B/R = chroma depth);
// otherwise a single plane at bits. Sections must be ascending; everything
// before the first section copies verbatim (header + lod/thumbnail prefixes).
struct cs_body_section {
    size_t offset;
    size_t count;
    int bits;
    int chroma_bits;
    bool interleaved_bgr;
};

// The sections of a production container whose samples start at byte y0: [Y x mY][Cr x mC][Cb x mC], then the
// anchor payload (anchor_len bytes, kept as bytes) when there is one.
inline std::vector<cs_body_section> cs_ycc_body_sections(size_t y0, size_t mY, size_t mC, int bits, int chroma_bits,
    size_t anchor_len) {
    std::vector<cs_body_section> sections = {
        { y0, mY, bits, 0, false },
        { y0 + mY, mC, chroma_bits, 0, false },
        { y0 + mY + mC, mC, chroma_bits, 0, false } };
    if (anchor_len > 0) sections.push_back({ y0 + mY + 2 * mC, anchor_len, 8, 0, false });
    return sections;
}

// Parses the header from the bytes of a container. Returns false for other formats (out.legacy) and corrupt
// headers.
bool cs_parse_header(const uint8_t* buf, size_t buf_bytes, CsHeaderInfo& out);

#endif
