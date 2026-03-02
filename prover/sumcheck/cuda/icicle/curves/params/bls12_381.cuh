#pragma once
#ifndef BLS12_381_PARAMS_H
#define BLS12_381_PARAMS_H

#include "fields/storage.cuh"

#include "curves/macro.h"
#include "curves/projective.cuh"
#include "fields/snark_fields/bls12_381_base.cuh"
#include "fields/snark_fields/bls12_381_scalar.cuh"
#include "fields/quadratic_extension.cuh"

namespace bls12_381 {
  // G1 and G2 generators
  static constexpr storage<fq_config::limbs_count> g1_gen_x = {0xdb22c6bb, 0xfb3af00a, 0xf97a1aef, 0x6c55e83f,
                                                               0x171bac58, 0xa14e3a3f, 0x9774b905, 0xc3688c4f,
                                                               0x4fa9ac0f, 0x2695638c, 0x3197d794, 0x17f1d3a7};
  static constexpr storage<fq_config::limbs_count> g1_gen_y = {0x46c5e7e1, 0x0caa2329, 0xa2888ae4, 0xd03cc744,
                                                               0x2c04b3ed, 0x00db18cb, 0xd5d00af6, 0xfcf5e095,
                                                               0x741d8ae4, 0xa09e30ed, 0xe3aaa0f1, 0x08b3f481};
  static constexpr storage<fq_config::limbs_count> g2_gen_x_re = {0xc121bdb8, 0xd48056c8, 0xa805bbef, 0x0bac0326,
                                                                  0x7ae3d177, 0xb4510b64, 0xfa403b02, 0xc6e47ad4,
                                                                  0x2dc51051, 0x26080527, 0xf08f0a91, 0x024aa2b2};
  static constexpr storage<fq_config::limbs_count> g2_gen_x_im = {0x5d042b7e, 0xe5ac7d05, 0x13945d57, 0x334cf112,
                                                                  0xdc7f5049, 0xb5da61bb, 0x9920b61a, 0x596bd0d0,
                                                                  0x88274f65, 0x7dacd3a0, 0x52719f60, 0x13e02b60};
  static constexpr storage<fq_config::limbs_count> g2_gen_y_re = {0x08b82801, 0xe1935486, 0x3baca289, 0x923ac9cc,
                                                                  0x5160d12c, 0x6d429a69, 0x8cbdd3a7, 0xadfd9baa,
                                                                  0xda2e351a, 0x8cc9cdc6, 0x727d6e11, 0x0ce5d527};
  static constexpr storage<fq_config::limbs_count> g2_gen_y_im = {0xf05f79be, 0xaaa9075f, 0x5cec1da1, 0x3f370d27,
                                                                  0x572e99ab, 0x267492ab, 0x85a763af, 0xcb3e287e,
                                                                  0x2bc28b99, 0x32acd2b0, 0x2ea734cc, 0x0606c4a0};

  static constexpr storage<fq_config::limbs_count> weierstrass_b = {0x00000004, 0x00000000, 0x00000000, 0x00000000,
                                                                    0x00000000, 0x00000000, 0x00000000, 0x00000000,
                                                                    0x00000000, 0x00000000, 0x00000000, 0x00000000};
  static constexpr storage<fq_config::limbs_count> weierstrass_b_g2_re = {
    0x00000004, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
  static constexpr storage<fq_config::limbs_count> weierstrass_b_g2_im = {
    0x00000004, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};

  CURVE_DEFINITIONS
  G2_CURVE_DEFINITIONS
} // namespace bls12_381

#endif
