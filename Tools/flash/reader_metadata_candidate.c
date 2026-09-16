#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

/* Scalar integer expansion also specifies NaN payload/sign and quieting. */
static uint32_t fp16_to_fp32_bits(uint16_t half) {
    uint32_t sign = (uint32_t)(half & 0x8000) << 16;
    uint32_t exponent = (half >> 10) & 31;
    uint32_t fraction = half & 1023;
    if (exponent == 31) return sign | 0x7f800000u | (fraction << 13)
        | (fraction ? 0x00400000u : 0);
    if (exponent) return sign | ((exponent + 112) << 23) | (fraction << 13);
    if (!fraction) return sign;
    exponent = 113;
    while (!(fraction & 1024)) { fraction <<= 1; --exponent; }
    return sign | (exponent << 23) | ((fraction & 1023) << 13);
}

void slotstream_affine_fp16_to_fp32(const uint8_t *restrict source,
                                   uint8_t *restrict target, size_t count) {
#if defined(__aarch64__)
    while (count >= 8) {
        /* Byte loads/stores permit unaligned source and destination. */
        float16x8_t half = vreinterpretq_f16_u8(vld1q_u8(source));
        vst1q_u8(target, vreinterpretq_u8_f32(vcvt_f32_f16(vget_low_f16(half))));
        vst1q_u8(target + 16, vreinterpretq_u8_f32(vcvt_f32_f16(vget_high_f16(half))));
        source += 16; target += 32; count -= 8;
    }
#endif
    while (count--) {
        uint16_t half = (uint16_t)source[0] | ((uint16_t)source[1] << 8);
        uint32_t bits = fp16_to_fp32_bits(half);
        memcpy(target, &bits, sizeof(bits));
        source += 2; target += 4;
    }
}
