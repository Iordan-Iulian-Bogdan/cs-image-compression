#include "container_format.hpp"
#include <cstring>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <algorithm>

// The fixed seed of every sample-position draw ("cs-sample-seed-v4", zero padded). Any constant works; changing it
// changes every container.
const uint8_t cs_sample_seed[32] = { 'c', 's', '-', 's', 'a', 'm', 'p', 'l', 'e', '-', 's', 'e', 'e', 'd', '-', 'v', '4' };

bool cs_write_container_file(const std::string& path, const uint8_t* data, size_t bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write((const char*)data, (std::streamsize)bytes);
    return (bool)f;
}

bool cs_read_container_file(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const std::streamoff n = f.tellg();
    if (n <= 0) return false;
    out.resize((size_t)n);
    f.seekg(0);
    f.read((char*)out.data(), n);
    return (bool)f;
}

std::string cs_build_metadata(int m, int rows, int cols, int org_h, int org_w, int odd_parity) {
    std::string text = std::to_string(m) + "|" + std::to_string(rows) + "|" + std::to_string(cols) + "|" +
        std::to_string(org_h) + "|" + std::to_string(org_w);
    if (odd_parity != 0) text += "|" + std::to_string(odd_parity);   // even images keep the 5-field text
    if (text.size() > CS_METADATA_BYTES) {
        throw std::runtime_error("metadata too large for header (image dimensions out of range)");
    }
    return std::string(CS_METADATA_BYTES - text.size(), '-') + text;
}

namespace {

// Parse the '-'-padded "m|rows|cols|h|w" metadata string. Returns false on
// malformed content instead of throwing, so header errors surface uniformly.
// Five fields "m|rows|cols|h|w", or six with the odd-size parity (vals[5], 0 when absent, 0..3).
bool parse_metadata_fields(const uint8_t* meta, int (&vals)[6]) {
    std::string text((const char*)meta, CS_METADATA_BYTES);
    const size_t first = text.find_first_not_of('-');
    if (first == std::string::npos) return false;
    text = text.substr(first);

    int vi = 0;
    std::string cur;
    for (char c : text) {
        if (c == '|') {
            if (vi >= 5) return false;
            try { vals[vi++] = std::stoi(cur); } catch (...) { return false; }
            cur.clear();
        }
        else if (c == '-' || (c >= '0' && c <= '9')) {
            cur.push_back(c);
        }
        else {
            return false;
        }
    }
    if (vi != 4 && vi != 5) return false;
    try { vals[vi++] = std::stoi(cur); } catch (...) { return false; }
    if (vi == 5) vals[5] = 0;
    return vals[5] >= 0 && vals[5] <= 3;
}

} // namespace

bool cs_write_header(uint8_t* buf, size_t buf_bytes, const std::string& metadata) {
    if (buf_bytes < CS_HEADER_BYTES || metadata.size() != CS_METADATA_BYTES) return false;
    buf[CS_OFF_VERSION] = CS_VERSION;
    std::memcpy(buf + CS_OFF_META, metadata.data(), CS_METADATA_BYTES);
    std::memset(buf + CS_OFF_PAD, 0, CS_HEADER_BYTES - CS_OFF_PAD);
    return true;
}

void cs_pack_samples(const uint8_t* in, size_t count, int bits, uint8_t* out) {
    if (count == 0) return;
    if (bits >= 8) {
        std::memcpy(out, in, count);
        return;
    }
    const int maxv = (1 << bits) - 1;
    uint32_t acc = 0;
    int acc_bits = 0;
    size_t o = 0;
    for (size_t i = 0; i < count; ++i) {
        const uint32_t q = (uint32_t)((in[i] * maxv + 127) / 255);
        acc = (acc << (uint32_t)bits) | q;
        acc_bits += bits;
        while (acc_bits >= 8) {
            acc_bits -= 8;
            out[o++] = (uint8_t)((acc >> acc_bits) & 0xFF);
        }
    }
    if (acc_bits > 0) {
        out[o++] = (uint8_t)((acc << (8 - acc_bits)) & 0xFF);
    }
    (void)o;
}

void cs_unpack_samples(const uint8_t* in, size_t count, int bits, uint8_t* out) {
    if (count == 0) return;
    if (bits >= 8) {
        std::memcpy(out, in, count);
        return;
    }
    const int maxv = (1 << bits) - 1;
    uint32_t acc = 0;
    int acc_bits = 0;
    size_t i = 0;
    for (size_t k = 0; k < count; ++k) {
        while (acc_bits < bits) {
            acc = (acc << 8) | in[i++];
            acc_bits += 8;
        }
        acc_bits -= bits;
        const uint32_t q = (acc >> acc_bits) & (uint32_t)maxv;
        acc &= (acc_bits > 0) ? ((1u << acc_bits) - 1u) : 0u;
        out[k] = (uint8_t)((q * 255 + (uint32_t)(maxv / 2)) / (uint32_t)maxv);
    }
}

void cs_pack_samples_bgr(const uint8_t* in, size_t samples, int luma_bits,
    int chroma_bits, uint8_t* out) {
    if (luma_bits == chroma_bits) {
        cs_pack_samples(in, samples * 3, luma_bits, out);
        return;
    }
    uint32_t acc = 0;
    int acc_bits = 0;
    size_t o = 0;
    for (size_t i = 0; i < samples * 3; ++i) {
        const int bits = (i % 3 == 1) ? luma_bits : chroma_bits; // G is luma-weighted
        const int maxv = (1 << bits) - 1;
        const uint32_t q = (uint32_t)((in[i] * maxv + 127) / 255);
        acc = (acc << (uint32_t)bits) | q;
        acc_bits += bits;
        while (acc_bits >= 8) {
            acc_bits -= 8;
            out[o++] = (uint8_t)((acc >> acc_bits) & 0xFF);
        }
    }
    if (acc_bits > 0) out[o] = (uint8_t)((acc << (8 - acc_bits)) & 0xFF);
}

void cs_unpack_samples_bgr(const uint8_t* in, size_t samples, int luma_bits,
    int chroma_bits, uint8_t* out) {
    if (luma_bits == chroma_bits) {
        cs_unpack_samples(in, samples * 3, luma_bits, out);
        return;
    }
    uint32_t acc = 0;
    int acc_bits = 0;
    size_t i = 0;
    for (size_t k = 0; k < samples * 3; ++k) {
        const int bits = (k % 3 == 1) ? luma_bits : chroma_bits;
        const int maxv = (1 << bits) - 1;
        while (acc_bits < bits) {
            acc = (acc << 8) | in[i++];
            acc_bits += 8;
        }
        acc_bits -= bits;
        const uint32_t q = (acc >> acc_bits) & (uint32_t)maxv;
        acc &= (acc_bits > 0) ? ((1u << acc_bits) - 1u) : 0u;
        out[k] = (uint8_t)((q * 255 + (uint32_t)(maxv / 2)) / (uint32_t)maxv);
    }
}

bool cs_parse_header(const uint8_t* buf, size_t buf_bytes, CsHeaderInfo& out) {
    out = CsHeaderInfo{};
    if (buf_bytes < CS_HEADER_BYTES || buf[CS_OFF_VERSION] != CS_VERSION) {
        out.legacy = true;
        return false;
    }
    int vals[6];
    if (!parse_metadata_fields(buf + CS_OFF_META, vals)) return false;
    out.version = CS_VERSION;
    out.m = vals[0]; out.rows = vals[1]; out.cols = vals[2];
    out.org_h = vals[3]; out.org_w = vals[4]; out.odd_parity = vals[5];

    // sampling-mode flags live in the pad region:
    // [CS_OFF_PAD] = mode (0 random, 1 periodic, 2 adaptive), followed by
    // tile_size and (mode 1) samples_per_tile as 16-bit little-endian values.
    // Mode 2 stores lod byte count in the second slot as a sanity check.
    const uint8_t mode = buf[CS_OFF_PAD];
    if (mode == CS_MODE_BGR_HRDETAIL) {
        // Uniform BGR sampling + trailing HR-detail payload. Tile/aux pad
        // fields must be zero; the sample count rides at CS_OFF_HR_COUNT.
        out.sampling_mode = mode;
        out.hr_detail = true;
        const int tile = buf[CS_OFF_PAD + 1] | (buf[CS_OFF_PAD + 2] << 8);
        const int aux = buf[CS_OFF_PAD + 3] | (buf[CS_OFF_PAD + 4] << 8) |
            (buf[CS_OFF_PAD + 5] << 16) | (buf[CS_OFF_PAD + 6] << 24);
        out.hr_count = (uint32_t)buf[CS_OFF_PAD + CS_OFF_HR_COUNT] |
            ((uint32_t)buf[CS_OFF_PAD + CS_OFF_HR_COUNT + 1] << 8) |
            ((uint32_t)buf[CS_OFF_PAD + CS_OFF_HR_COUNT + 2] << 16) |
            ((uint32_t)buf[CS_OFF_PAD + CS_OFF_HR_COUNT + 3] << 24);
        if (tile != 0 || aux != 0 || out.hr_count == 0 ||
            out.hr_count > CS_HRDETAIL_MAX_SAMPLES) {
            out.legacy = true;
            return false;
        }
    }
    else if (mode == CS_MODE_YCC420_SEG_HRDETAIL) {
        // Seg-mask luma + uniform chroma + compact anchor payload. The seg
        // geometry is read by the sampler; hr_count here is the payload BYTE
        // length (cs_read_anchor_payload checks it), in the pad bytes after
        // the mode byte (the seg prefix length occupies CS_OFF_SEGPRE_LEN,
        // so CS_OFF_HR_COUNT cannot be reused).
        out.sampling_mode = mode;
        out.hr_detail = true;
        out.hr_count = (uint32_t)buf[CS_OFF_PAD + 1] |
            ((uint32_t)buf[CS_OFF_PAD + 2] << 8) |
            ((uint32_t)buf[CS_OFF_PAD + 3] << 16) |
            ((uint32_t)buf[CS_OFF_PAD + 4] << 24);
        if (out.hr_count == 0) {
            out.legacy = true;
            return false;
        }
    }
    else if (mode == CS_MODE_YCC420 || mode == CS_MODE_YCC422 || mode == CS_MODE_YCC420_HRDETAIL ||
        mode == CS_MODE_YCC420_SEG || mode == CS_MODE_YCC422_SEG ||
        mode == CS_MODE_YCC420_TCLUSTER || mode == CS_MODE_YCC420_TMEANS ||
        mode == CS_MODE_YCC422_TCLUSTER || mode == CS_MODE_YCC422_TMEANS) {
        // luma/chroma-split container. Pure-uniform modes 3/7 leave the tile
        // fields zero; the LOD-luma variant reuses the adaptive pad layout
        // (tile_size, lod byte count, weight_base). Accept the LOD variant
        // only when the lod count matches the geometry, otherwise fall back
        // to uniform (tolerant read of pre-LOD containers). Mode 9
        // (HR-detail) is uniform-only: any tile grid is rejected.
        out.sampling_mode = mode;
        if (mode == CS_MODE_YCC420_HRDETAIL) out.hr_detail = true;
        const int tile = buf[CS_OFF_PAD + 1] | (buf[CS_OFF_PAD + 2] << 8);
        const int lodc = buf[CS_OFF_PAD + 3] | (buf[CS_OFF_PAD + 4] << 8);
        if (mode == CS_MODE_YCC420_HRDETAIL) {
            // Uniform-only: any tile grid is rejected, and the sample
            // count rides at CS_OFF_HR_COUNT.
            out.hr_count = (uint32_t)buf[CS_OFF_PAD + CS_OFF_HR_COUNT] |
                ((uint32_t)buf[CS_OFF_PAD + CS_OFF_HR_COUNT + 1] << 8) |
                ((uint32_t)buf[CS_OFF_PAD + CS_OFF_HR_COUNT + 2] << 16) |
                ((uint32_t)buf[CS_OFF_PAD + CS_OFF_HR_COUNT + 3] << 24);
            if (tile != 0 || out.hr_count == 0 ||
                out.hr_count > CS_HRDETAIL_MAX_SAMPLES) {
                out.legacy = true;
                return false;
            }
        }
        else if (tile > 0 && lodc == cs_lod_bytes(out.rows, out.cols, tile) &&
            mode != CS_MODE_YCC420_HRDETAIL) {
            out.periodic_tile = tile;
            out.periodic_samples = lodc;
            const int base = buf[CS_OFF_PAD + CS_OFF_ADAPTIVE_BASE] |
                (buf[CS_OFF_PAD + CS_OFF_ADAPTIVE_BASE + 1] << 8);
            out.adaptive_base = (base > 0) ? base : 256;
        }
        else if (tile > 0) {
            // Tile grid present but the lod count disagrees with the
            // geometry: the 16-bit count field truncated (grids over 65535
            // tiles, only writable by the pre-guard writer). Falling back
            // to uniform would silently scatter every sample (6 dB
            // garbage, exit 0), so reject for re-encoding instead.
            // tile == 0
            // still means a pre-LOD uniform container (tolerant read).
            out.legacy = true;
            return false;
        }
    }
    else if (mode == CS_MODE_YCC420_HF || mode == CS_MODE_YCC422_HF) {
        out.sampling_mode = mode;
        out.periodic_tile = buf[CS_OFF_PAD + 1] | (buf[CS_OFF_PAD + 2] << 8);
        const int lodc = buf[CS_OFF_PAD + 3] | (buf[CS_OFF_PAD + 4] << 8);
        const int base = buf[CS_OFF_PAD + CS_OFF_ADAPTIVE_BASE] |
            (buf[CS_OFF_PAD + CS_OFF_ADAPTIVE_BASE + 1] << 8);
        out.adaptive_base = base > 0 ? base : 65535;
        out.thumb_len = (uint32_t)buf[CS_OFF_PAD + CS_OFF_HF_THUMBLEN] |
            ((uint32_t)buf[CS_OFF_PAD + CS_OFF_HF_THUMBLEN + 1] << 8) |
            ((uint32_t)buf[CS_OFF_PAD + CS_OFF_HF_THUMBLEN + 2] << 16) |
            ((uint32_t)buf[CS_OFF_PAD + CS_OFF_HF_THUMBLEN + 3] << 24);
        if (out.periodic_tile <= 0 || lodc != 0) {
            out.legacy = true;
            return false;
        }
    }
    else if (mode == CS_MODE_SEGMASK) {
        // Per-segment sampling: per-class rates + PNG label map ride in the
        // body prefix (parsed by the sampler); no pad geometry to validate.
        out.sampling_mode = mode;
    }
    else if (mode == CS_MODE_TCLUSTER || mode == CS_MODE_TMEANS) {
        // T-tetromino sampling: the seed + geometry regenerate the pair
        // stream; no pad geometry to validate.
        out.sampling_mode = mode;
    }
    else if (mode == CS_MODE_PERIODIC || mode == CS_MODE_ADAPTIVE || mode == CS_MODE_HF_FOCUS) {
        out.sampling_mode = mode;
        out.periodic_tile = buf[CS_OFF_PAD + 1] | (buf[CS_OFF_PAD + 2] << 8);
        out.periodic_samples = buf[CS_OFF_PAD + 3] | (buf[CS_OFF_PAD + 4] << 8);
        if (mode == CS_MODE_ADAPTIVE || mode == CS_MODE_HF_FOCUS) {
            int base = buf[CS_OFF_PAD + CS_OFF_ADAPTIVE_BASE] |
                (buf[CS_OFF_PAD + CS_OFF_ADAPTIVE_BASE + 1] << 8);
            out.adaptive_base = (base > 0) ? base : 256; // 0 = pre-strength default
        }
        const bool lod_ok = (mode != CS_MODE_ADAPTIVE && mode != CS_MODE_HF_FOCUS) ||
            (out.periodic_tile > 0 &&
                out.periodic_samples == cs_lod_bytes(out.rows, out.cols, out.periodic_tile));
        if (out.periodic_tile <= 0 ||
            (mode == CS_MODE_PERIODIC && out.periodic_samples <= 0) ||
            !lod_ok) {
            if ((mode == CS_MODE_ADAPTIVE || mode == CS_MODE_HF_FOCUS) &&
                out.periodic_tile > 0) {
                // Lod count disagrees with geometry: the 16-bit count field
                // truncated (grids over 65535 tiles, only writable by the
                // pre-guard writer). Falling back to RANDOM would silently
                // scatter every sample (6 dB garbage, exit 0), so reject
                // for re-encoding instead.
                out.legacy = true;
                return false;
            }
            out.sampling_mode = CS_MODE_RANDOM;
            out.periodic_tile = 0;
            out.periodic_samples = 0;
            out.adaptive_base = 256;
            out.thumb_len = 0;
        }
        if (mode == CS_MODE_HF_FOCUS && out.sampling_mode == mode) {
            out.thumb_len = (uint32_t)buf[CS_OFF_PAD + CS_OFF_HF_THUMBLEN] |
                ((uint32_t)buf[CS_OFF_PAD + CS_OFF_HF_THUMBLEN + 1] << 8) |
                ((uint32_t)buf[CS_OFF_PAD + CS_OFF_HF_THUMBLEN + 2] << 16) |
                ((uint32_t)buf[CS_OFF_PAD + CS_OFF_HF_THUMBLEN + 3] << 24);
        }
    }
    // sample bit-depth travels in the pad for every mode: 0 = legacy writer
    // (zeroed pad) decodes as 8; anything outside 1..8 is a corrupt header.
    {
        if (buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_MARKER] != 0 &&
            buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_MARKER] != CS_SAMPLE_BITS_MARKER) {
            return false;
        }
        const int sb = buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS];
        if (sb == 0) out.sample_bits = 8;
        else if (sb >= 1 && sb <= 8) out.sample_bits = sb;
        else return false;
        const int cb = buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_CHROMA];
        if (cb == 0) out.sample_bits_chroma = out.sample_bits;
        else if (cb >= 1 && cb <= 8) out.sample_bits_chroma = cb;
        else return false;
        // Marker + 8/8 is the default writer output (depths always stamped,
        // body left unpacked): plain 8-bit.
    }
    // --lod-full detail guarantee travels in the pad for adaptive-family
    // modes; 0 = off. Only honored from stamped headers so a
    // legacy zeroed pad always decodes as off.
    out.full_threshold = (buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_MARKER] == CS_SAMPLE_BITS_MARKER)
        ? buf[CS_OFF_PAD + CS_OFF_LOD_FULL] : 0;
    // --adaptive-cutoff hard floor: only adaptive-family writers stamp it
    // (mode 2, LOD-luma mode 3); other modes alias this byte for thumbnail
    // lengths / HR counts, so gate strictly by mode.
    out.adaptive_cutoff = 0;
    if (buf[CS_OFF_PAD + CS_OFF_SAMPLE_BITS_MARKER] == CS_SAMPLE_BITS_MARKER &&
        (out.sampling_mode == CS_MODE_ADAPTIVE ||
            (out.sampling_mode == CS_MODE_YCC420 && out.periodic_tile > 0))) {
        out.adaptive_cutoff = buf[CS_OFF_PAD + CS_OFF_ADAPTIVE_CUTOFF];
    }
    return true;
}
