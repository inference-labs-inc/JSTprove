#pragma once
#ifndef BN254_PARAMS_H
#define BN254_PARAMS_H

#include "fields/storage.cuh"

#include "curves/macro.h"
#include "curves/projective.cuh"
#include "fields/snark_fields/bn254_base.cuh"
#include "fields/snark_fields/bn254_scalar.cuh"
#include "fields/quadratic_extension.cuh"

namespace bn254 {
  // G1 and G2 generators
  static constexpr storage<fq_config::limbs_count> g1_gen_x = {0x00000001, 0x00000000, 0x00000000, 0x00000000,
                                                               0x00000000, 0x00000000, 0x00000000, 0x00000000};
  static constexpr storage<fq_config::limbs_count> g1_gen_y = {0x00000002, 0x00000000, 0x00000000, 0x00000000,
                                                               0x00000000, 0x00000000, 0x00000000, 0x00000000};
  static constexpr storage<fq_config::limbs_count> g2_gen_x_re = {0xd992f6ed, 0x46debd5c, 0xf75edadd, 0x674322d4,
                                                                  0x5e5c4479, 0x426a0066, 0x121f1e76, 0x1800deef};
  static constexpr storage<fq_config::limbs_count> g2_gen_x_im = {0xaef312c2, 0x97e485b7, 0x35a9e712, 0xf1aa4933,
                                                                  0x31fb5d25, 0x7260bfb7, 0x920d483a, 0x198e9393};
  static constexpr storage<fq_config::limbs_count> g2_gen_y_re = {0x66fa7daa, 0x4ce6cc01, 0x0c43d37b, 0xe3d1e769,
                                                                  0x8dcb408f, 0x4aab7180, 0xdb8c6deb, 0x12c85ea5};
  static constexpr storage<fq_config::limbs_count> g2_gen_y_im = {0xd122975b, 0x55acdadc, 0x70b38ef3, 0xbc4b3133,
                                                                  0x690c3395, 0xec9e99ad, 0x585ff075, 0x090689d0};

  static constexpr storage<fq_config::limbs_count> weierstrass_b = {0x00000003, 0x00000000, 0x00000000, 0x00000000,
                                                                    0x00000000, 0x00000000, 0x00000000, 0x00000000};
  static constexpr storage<fq_config::limbs_count> weierstrass_b_g2_re = {
    0x24a138e5, 0x3267e6dc, 0x59dbefa3, 0xb5b4c5e5, 0x1be06ac3, 0x81be1899, 0xceb8aaae, 0x2b149d40};
  static constexpr storage<fq_config::limbs_count> weierstrass_b_g2_im = {
    0x85c315d2, 0xe4a2bd06, 0xe52d1852, 0xa74fa084, 0xeed8fdf4, 0xcd2cafad, 0x3af0fed4, 0x009713b0};

  CURVE_DEFINITIONS
  G2_CURVE_DEFINITIONS
} // namespace bn254

#endif
