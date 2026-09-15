#ifndef SLOTSTREAM_AFFINE_H
#define SLOTSTREAM_AFFINE_H

#include <stddef.h>
#include <stdint.h>

/* Exact 4-bit to 6-bit code expansion. The caller validates an even source
 * byte count, non-overlapping buffers, and source_bytes * 3 / 2 output bytes.
 * Unaligned addresses are supported. No scales or quantized values change. */
void slotstream_affine_widen4to6(const uint8_t *source, uint8_t *target,
                               size_t source_bytes);

#endif
