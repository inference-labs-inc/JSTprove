#pragma once
#ifndef GRUMPKIN_PARAMS_H
#define GRUMPKIN_PARAMS_H

#include "fields/storage.cuh"

#include "curves/macro.h"
#include "curves/projective.cuh"
#include "fields/snark_fields/grumpkin_base.cuh"
#include "fields/snark_fields/grumpkin_scalar.cuh"

namespace grumpkin {
  typedef bn254::fp_config fq_config;
  // G1 generator
  static constexpr storage<fq_config::limbs_count> g1_gen_x = {0x00000001, 0x00000000, 0x00000000, 0x00000000,
                                                               0x00000000, 0x00000000, 0x00000000, 0x00000000};
  static constexpr storage<fq_config::limbs_count> g1_gen_y = {0x823f272c, 0x833fc48d, 0xf1181294, 0x2d270d45,
                                                               0x6a45d63,  0xcf135e75, 0x00000002, 0x00000000};

  static constexpr storage<fq_config::limbs_count> weierstrass_b = {0xeffffff0, 0x43e1f593, 0x79b97091, 0x2833e848,
                                                                    0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72};

  CURVE_DEFINITIONS
} // namespace grumpkin

#endif
