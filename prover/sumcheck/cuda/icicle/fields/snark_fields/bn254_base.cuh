#pragma once
#ifndef BN254_BASE_PARAMS_H
#define BN254_BASE_PARAMS_H

#include "fields/storage.cuh"
#include "fields/params_gen.cuh"

namespace bn254 {
  struct fq_config {
    static constexpr storage<8> modulus = {0xd87cfd47, 0x3c208c16, 0x6871ca8d, 0x97816a91,
                                           0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72};
    PARAMS(modulus)

    // nonresidue to generate the extension field
    static constexpr uint32_t nonresidue = 1;
    // true if nonresidue is negative
    static constexpr bool nonresidue_is_negative = true;
  };
} // namespace bn254

#endif
