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

bool cs_parse_header(const uint8_t* buf, size_t buf_bytes, CsHeaderInfo& out) {
    out = CsHeaderInfo{};
    if (buf_bytes < CS_HEADER_BYTES || buf[CS_OFF_VERSION] != CS_VERSION) {
        out.legacy = true;
        return false;
    }
    int vals[6];
    if (!parse_metadata_fields(buf + CS_OFF_META, vals)) return false;
    out.m = vals[0]; out.rows = vals[1]; out.cols = vals[2];
    out.org_h = vals[3]; out.org_w = vals[4]; out.odd_parity = vals[5];

    const uint8_t mode = buf[CS_OFF_PAD];
    if (mode == CS_MODE_YCC420_SEG_HRDETAIL) {
        // The anchor payload's byte length (cs_read_anchor_payload checks it) in the pad bytes after the mode byte.
        out.hr_detail = true;
        out.hr_count = (uint32_t)buf[CS_OFF_PAD + 1] |
            ((uint32_t)buf[CS_OFF_PAD + 2] << 8) |
            ((uint32_t)buf[CS_OFF_PAD + 3] << 16) |
            ((uint32_t)buf[CS_OFF_PAD + 4] << 24);
        if (out.hr_count == 0) return false;
    }
    else if (mode != CS_MODE_YCC420_SEG && mode != CS_MODE_MULTISCALE) {
        return false;
    }
    out.sampling_mode = mode;
    // The sample bit depths (a layer stamps them; mode 20 leaves them zero: 8).
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
    return true;
}
