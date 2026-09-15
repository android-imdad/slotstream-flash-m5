#include "affine.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

void slotstream_affine_widen4to6(const uint8_t *restrict source,
                               uint8_t *restrict target,
                               size_t source_bytes) {
#if defined(__aarch64__)
    const uint8x16_t low = vdupq_n_u8(0x0f);
    const uint8x16_t middle = vdupq_n_u8(0x30);
    const uint8x16_t high = vdupq_n_u8(0xf0);
    while (source_bytes >= 32) {
        /* Deinterleave sixteen independent two-byte groups, expand each
         * group to three bytes, then interleave directly into the output.
         * The structure store writes exactly 48 bytes; no padding or
         * over-wide tail store is needed. */
        const uint8x16x2_t input = vld2q_u8(source);
        const uint8x16_t a = input.val[0], b = input.val[1];
        uint8x16x3_t output;
        output.val[0] = vorrq_u8(vandq_u8(a, low),
                               vshlq_n_u8(vandq_u8(a, middle), 2));
        output.val[1] = vorrq_u8(vshrq_n_u8(a, 6), vshlq_n_u8(b, 4));
        output.val[2] = vshrq_n_u8(vandq_u8(b, high), 2);
        vst3q_u8(target, output);
        source += 32;
        target += 48;
        source_bytes -= 32;
    }
#endif
    /* Portable fallback and the even-byte tail of the NEON path. */
    while (source_bytes >= 2) {
        const uint8_t a = source[0], b = source[1];
        target[0] = (a & 0x0f) | ((a & 0x30) << 2);
        target[1] = (a >> 6) | ((b & 0x0f) << 4);
        target[2] = (b & 0xf0) >> 2;
        source += 2;
        target += 3;
        source_bytes -= 2;
    }
}
