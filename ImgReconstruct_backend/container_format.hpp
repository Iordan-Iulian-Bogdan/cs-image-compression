#ifndef CS_CONTAINER_FORMAT_HPP
#define CS_CONTAINER_FORMAT_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Container format v4
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
// Containers of other versions are not read; re-encode them from the photo.
// ---------------------------------------------------------------------------

#define CS_HEADER_PIXELS 16          // header size in container pixels
#define CS_HEADER_BYTES (CS_HEADER_PIXELS * 3)
#define CS_METADATA_BYTES 32
#define CS_VERSION 4                 // current container format version

// Offsets inside the 48-byte header
#define CS_OFF_VERSION 0
#define CS_OFF_META 1
#define CS_OFF_PAD 33                // 15 bytes: the mode and its fields

// The fixed seed of every sample-position draw.
extern const uint8_t cs_sample_seed[32];

// Builds the 32-byte '-'-padded metadata string "m|rows|cols|h|w", plus "|p" when odd_parity != 0
// (bit 0: native height odd, bit 1: native width odd; the encoder padded that side by one replicated
// row/column and decode crops it back). Throws std::runtime_error if the unpadded text exceeds 32 bytes.
std::string cs_build_metadata(int m, int rows, int cols, int org_h, int org_w, int odd_parity = 0);

// Writes the header into buf (the first CS_HEADER_BYTES bytes of a container): version, metadata, zeroed pad.
// Returns false if buf is too small or the metadata is not CS_METADATA_BYTES long.
bool cs_write_header(uint8_t* buf, size_t buf_bytes, const std::string& metadata);

// Container files: the container's bytes, nothing else.
bool cs_write_container_file(const std::string& path, const uint8_t* data, size_t bytes);
// Reads a container file; false if it cannot be read.
bool cs_read_container_file(const std::string& path, std::vector<uint8_t>& out);

// Container modes, recorded in the pad byte (CS_OFF_PAD)
#define CS_MODE_YCC420_SEG 11          // a layer: seg-mask luma + 4:2:0 chroma
#define CS_MODE_YCC420_SEG_HRDETAIL 19 // the 2x layer: the same + the anchor payload
// Multi-scale (pad = 20), the container the encoder writes: the photo split into T x T tiles (full resolution),
// each tile coded at one scale s = 2..6 (the photo pre-shrunk by s/2 before the usual 2x pipeline). One complete
// layer container (mode 11, or 19 for the anchored 2x layer) per scale in use carries that scale's tiles; the
// decoder decodes every layer, scales it to full size and blends the layers by the tile scale map. Body after the
// header:
//   [descriptor; byte length (u32) at CS_OFF_PAD + CS_OFF_SEGPRE_LEN]
//     [u8 version = 1][u16 W][u16 H][u16 T][u8 n]{[u8 scale][u32 layer bytes]} x n [u32 map bytes][map PNG]
//     (map: 8-bit, ceil(H / T) x ceil(W / T), the tile's scale)
//   [layer containers' bytes, in descriptor order (each a whole number of 3-byte pixels)]
#define CS_MODE_MULTISCALE 20

// A layer's body prefix, little-endian, right after the header:
//   [0][v][0]  marker, prefix version v (5; 2 in containers from before 2026-11-20), a zero byte
//   [3]        nclasses (u8, 1..255)
//   [4..3+n]   per-class quantized rates (u8: count_c = round(pix_c * q/255))
//   [4+n..]    label-map PNG length (u32)
//   [...]      label-map PNG bytes (8-bit grayscale class ids, pipeline geometry)
// Rates are quantized before the draw so both sides derive identical counts from the shipped bytes. The chroma
// positions are drawn only inside the label map's support (chroma pixels whose luma block holds a label with
// rate > 0; cs_seg_chroma_support). Version 5 draws them with chroma draw v2 (CsSampler::returnYcc420ChromaIndices:
// a count floor and, on coarse layers, a stratified draw); version 2 with the plain random draw. The prefix's byte length ([0] through the PNG) rides in the pad at
// CS_OFF_SEGPRE_LEN.
struct CsSegPrefix {
    uint8_t nclasses = 0;
    std::vector<uint8_t> rates_q;
    uint32_t seglen = 0;
    bool chroma_draw_v2 = false;   // version 5
    size_t total = 0; // 3 + 1 + nclasses + 4 + seglen
};
inline bool cs_segmask_parse_prefix(const uint8_t* body, size_t body_bytes, CsSegPrefix& out) {
    out = CsSegPrefix{};
    if (body_bytes < 8 || body[0] != 0 || (body[1] != 2 && body[1] != 5) || body[2] != 0) return false;
    out.chroma_draw_v2 = body[1] == 5;
    const uint8_t n = body[3];
    if (n == 0) return false;
    const size_t head = 3 + 1 + (size_t)n + 4;
    if (body_bytes < head) return false;
    out.nclasses = n;
    out.rates_q.assign(body + 4, body + 4 + n);
    const uint8_t* len = body + 4 + n;
    out.seglen = (uint32_t)len[0] | ((uint32_t)len[1] << 8) | ((uint32_t)len[2] << 16) | ((uint32_t)len[3] << 24);
    if (out.seglen == 0 || out.seglen > (1u << 24)) return false;
    if (head + out.seglen > body_bytes) return false;
    out.total = head + out.seglen;
    return true;
}
// Quantize a [0,1] sampling rate to the shipped u8 (both sides derive counts
// from these bytes, so encode must quantize before drawing, not after).
inline uint8_t cs_segmask_quantize_rate(float r) {
    if (!(r > 0.0f)) return 0;
    if (r >= 1.0f) return 255;
    return (uint8_t)std::lround(r * 255.0f);
}

// HR-detail anchors (mode 19): native-resolution luma anchors stored as a trailing body section (the compact
// flag map + luma of cs_anchor_payload). Decode applies them as exact luma data terms and then pins them.
#define CS_HRDETAIL_MAX_SAMPLES (1 << 24)

// One HR-detail anchor: linear index into the decode output geometry
// (row-major, output width stride) + native luma value.
struct CsHrSample {
    uint32_t idx = 0;
    uint8_t y = 0;
};

// Pad fields (offsets from CS_OFF_PAD):
//   [1..4]   mode 19: the anchor payload's byte length (u32 LE)
//   [5]      1 = the luma / Cr / Cb sections are entropy-coded (sample_coder.hpp) instead of bit-packed. Body
//            after the seg prefix: [u32 luma bytes][u32 Cr bytes][u32 Cb bytes][luma stream][Cr stream][Cb stream],
//            then the anchor payload. The streams hold the values quantized to the stamped sample bits, so the
//            decoded samples equal the bit-packed container's. Every written container is coded; the encoder's
//            in-memory anchor pass decodes a bit-packed one.
//   [7..10]  the seg prefix's (mode 20: the descriptor's) byte length (u32 LE)
//   [11]     luma sample bits (1..8)
//   [12]     chroma sample bits (1..8)
//   [13]     CS_SAMPLE_BITS_MARKER
#define CS_OFF_SAMPLE_CODING 5
#define CS_OFF_SEGPRE_LEN 7
#define CS_OFF_SAMPLE_BITS 11
#define CS_OFF_SAMPLE_BITS_CHROMA 12
#define CS_OFF_SAMPLE_BITS_MARKER 13
#define CS_SAMPLE_BITS_MARKER 0xA7

// Luma is sampled at full resolution (m positions); each chroma plane is downsampled 2x per axis and sampled on its
// own grid. The per-plane chroma count derives from the luma count (cs_chroma_count). Samples after the prefix:
//   [Y x m][Cr x m_chroma][Cb x m_chroma]
inline void cs_chroma_dims(int rows, int cols, int& crows, int& ccols) {
    crows = (rows + 1) / 2;
    ccols = (cols + 1) / 2;
}
inline int cs_chroma_count(int m_luma, int rows, int cols) {
    int crows, ccols;
    cs_chroma_dims(rows, cols, crows, ccols);
    if (rows <= 0 || cols <= 0) return 0;
    int mc = (int)(((long long)m_luma * crows * ccols) / ((long long)rows * cols));
    if (mc < 1) mc = 1;
    const int cap = crows * ccols;
    if (mc > cap) mc = cap;
    return mc;
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
    bool legacy = false;             // not a v4 container (an older or unknown format)
    int sampling_mode = 0;           // CS_MODE_* from the pad byte
    bool hr_detail = false;          // mode 19: trailing anchor payload present
    uint32_t hr_count = 0;           // mode 19: the anchor payload's byte length
    int sample_bits = 8;             // bits per stored luma sample value
    int sample_bits_chroma = 8;      // Cr/Cb
};

// Sample bit depths: bit-packed values are stored in draw order, MSB-first, each section whole-byte padded (pad
// bits zero). Quantize maps 8-bit values onto N bits (rounded), dequantize scales back; N = 8 is the identity.
inline int cs_sample_bits_valid(int bits) { return bits >= 1 && bits <= 8; }
inline size_t cs_packed_bytes(size_t count, int bits) {
    if (count == 0 || bits >= 8) return count;
    return (count * (size_t)bits + 7) / 8;
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

// Bit-level sample codec (MSB-first; out must hold cs_packed_bytes).
void cs_pack_samples(const uint8_t* in, size_t count, int bits, uint8_t* out);
void cs_unpack_samples(const uint8_t* in, size_t count, int bits, uint8_t* out);

// One packed body section: raw byte offset + value count + its depth. Sections must be ascending; everything
// before the first section copies verbatim (header + prefix).
struct cs_body_section {
    size_t offset;
    size_t count;
    int bits;
};

// The sections of a layer whose samples start at byte y0: [Y x mY][Cr x mC][Cb x mC], then the anchor payload
// (anchor_len bytes, kept as bytes) when there is one.
inline std::vector<cs_body_section> cs_ycc_body_sections(size_t y0, size_t mY, size_t mC, int bits, int chroma_bits,
    size_t anchor_len) {
    std::vector<cs_body_section> sections = {
        { y0, mY, bits },
        { y0 + mY, mC, chroma_bits },
        { y0 + mY + mC, mC, chroma_bits } };
    if (anchor_len > 0) sections.push_back({ y0 + mY + 2 * mC, anchor_len, 8 });
    return sections;
}

// Parses the header from the bytes of a container. Returns false for other formats (out.legacy) and corrupt
// headers, and for modes other than 11, 19 and 20.
bool cs_parse_header(const uint8_t* buf, size_t buf_bytes, CsHeaderInfo& out);

#endif
