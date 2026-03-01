#pragma once
#ifndef BLS12_381_BASE_PARAMS_H
#define BLS12_381_BASE_PARAMS_H

#include "fields/storage.cuh"
#include "fields/params_gen.cuh"

namespace bls12_381 {
  struct fq_config {
    static constexpr storage<12> modulus = {0xffffaaab, 0xb9feffff, 0xb153ffff, 0x1eabfffe, 0xf6b0f624, 0x6730d2a0,
                                            0xf38512bf, 0x64774b84, 0x434bacd7, 0x4b1ba7b6, 0x397fe69a, 0x1a0111ea};
    PARAMS(modulus)

    // nonresidue to generate the extension field
    static constexpr uint32_t nonresidue = 1;
    // true if nonresidue is negative
    static constexpr bool nonresidue_is_negative = true;
  };
} // namespace bls12_381

#endif
