#pragma once
#ifndef BLS12_377_BASE_PARAMS_H
#define BLS12_377_BASE_PARAMS_H

#include "fields/storage.cuh"
#include "fields/params_gen.cuh"

namespace bls12_377 {
  struct fq_config {
    static constexpr storage<12> modulus = {0x00000001, 0x8508c000, 0x30000000, 0x170b5d44, 0xba094800, 0x1ef3622f,
                                            0x00f5138f, 0x1a22d9f3, 0x6ca1493b, 0xc63b05c0, 0x17c510ea, 0x01ae3a46};
    PARAMS(modulus)

    static constexpr storage<12> rou = {0xc563b9a1, 0x7eca603c, 0x06fe0bc3, 0x06df0a43, 0x0ddff8c6, 0xb44d994a,
                                        0x4512a3d4, 0x40fbe05b, 0x8aeffc9b, 0x30f15248, 0x05198a80, 0x0036a92e};
    TWIDDLES(modulus, rou)

    // nonresidue to generate the extension field
    static constexpr uint32_t nonresidue = 5;
    // true if nonresidue is negative
    static constexpr bool nonresidue_is_negative = true;
  };
} // namespace bls12_377

#endif
