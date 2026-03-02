#pragma once
#ifndef BLS12_377_PARAMS_H
#define BLS12_377_PARAMS_H

#include "fields/storage.cuh"

#include "curves/macro.h"
#include "curves/projective.cuh"
#include "fields/snark_fields/bls12_377_base.cuh"
#include "fields/snark_fields/bls12_377_scalar.cuh"
#include "fields/quadratic_extension.cuh"

namespace bls12_377 {
  // G1 and G2 generators
  static constexpr storage<fq_config::limbs_count> g1_gen_x = {0xb21be9ef, 0xeab9b16e, 0xffcd394e, 0xd5481512,
                                                               0xbd37cb5c, 0x188282c8, 0xaa9d41bb, 0x85951e2c,
                                                               0xbf87ff54, 0xc8fc6225, 0xfe740a67, 0x008848de};
  static constexpr storage<fq_config::limbs_count> g1_gen_y = {0x559c8ea6, 0xfd82de55, 0x34a9591a, 0xc2fe3d36,
                                                               0x4fb82305, 0x6d182ad4, 0xca3e52d9, 0xbd7fb348,
                                                               0x30afeec4, 0x1f674f5d, 0xc5102eff, 0x01914a69};
  static constexpr storage<fq_config::limbs_count> g2_gen_x_re = {0x7c005196, 0x74e3e48f, 0xbb535402, 0x71889f52,
                                                                  0x57db6b9b, 0x7ea501f5, 0x203e5031, 0xc565f071,
                                                                  0xa3841d01, 0xc89630a2, 0x71c785fe, 0x018480be};
  static constexpr storage<fq_config::limbs_count> g2_gen_x_im = {0x6ea16afe, 0xb26bfefa, 0xbff76fe6, 0x5cf89984,
                                                                  0x0799c9de, 0xe7223ece, 0x6651cecb, 0x532777ee,
                                                                  0xb1b140d5, 0x70dc5a51, 0xe7004031, 0x00ea6040};
  static constexpr storage<fq_config::limbs_count> g2_gen_y_re = {0x09fd4ddf, 0xf0940944, 0x6d8c7c2e, 0xf2cf8888,
                                                                  0xf832d204, 0xe458c282, 0x74b49a58, 0xde03ed72,
                                                                  0xcbb2efb4, 0xd960736b, 0x5d446f7b, 0x00690d66};
  static constexpr storage<fq_config::limbs_count> g2_gen_y_im = {0x85eb8f93, 0xd9a1cdd1, 0x5e52270b, 0x4279b83f,
                                                                  0xcee304c2, 0x2463b01a, 0x3d591bf1, 0x61ef11ac,
                                                                  0x151a70aa, 0x9e549da3, 0xd2835518, 0x00f8169f};

  static constexpr storage<fq_config::limbs_count> weierstrass_b = {0x00000001, 0x00000000, 0x00000000, 0x00000000,
                                                                    0x00000000, 0x00000000, 0x00000000, 0x00000000,
                                                                    0x00000000, 0x00000000, 0x00000000, 0x00000000};
  static constexpr storage<fq_config::limbs_count> weierstrass_b_g2_re = {
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
  static constexpr storage<fq_config::limbs_count> weierstrass_b_g2_im = {
    0x9999999a, 0x1c9ed999, 0x1ccccccd, 0x0dd39e5c, 0x3c6bf800, 0x129207b6,
    0xcd5fd889, 0xdc7b4f91, 0x7460c589, 0x43bd0373, 0xdb0fd6f3, 0x010222f6};

  CURVE_DEFINITIONS
  G2_CURVE_DEFINITIONS
} // namespace bls12_377

#endif
