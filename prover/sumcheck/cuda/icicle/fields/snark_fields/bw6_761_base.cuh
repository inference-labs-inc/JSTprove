#pragma once
#ifndef BW6_761_BASE_BASE_H
#define BW6_761_BASE_BASE_H

#include "fields/storage.cuh"
#include "fields/params_gen.cuh"

namespace bw6_761 {
  struct fq_config {
    static constexpr storage<24> modulus = {0x0000008b, 0xf49d0000, 0x70000082, 0xe6913e68, 0xeaf0a437, 0x160cf8ae,
                                            0x5667a8f8, 0x98a116c2, 0x73ebff2e, 0x71dcd3dc, 0x12f9fd90, 0x8689c8ed,
                                            0x25b42304, 0x03cebaff, 0xe584e919, 0x707ba638, 0x8087be41, 0x528275ef,
                                            0x81d14688, 0xb926186a, 0x04faff3e, 0xd187c940, 0xfb83ce0a, 0x0122e824};
    PARAMS(modulus)
  };
} // namespace bw6_761

#endif
