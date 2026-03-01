#pragma once
#ifndef BW6_761_PARAMS_H
#define BW6_761_PARAMS_H

#include "fields/storage.cuh"

#include "curves/macro.h"
#include "curves/projective.cuh"
#include "fields/snark_fields/bw6_761_base.cuh"
#include "fields/snark_fields/bw6_761_scalar.cuh"
#include "fields/quadratic_extension.cuh"

namespace bw6_761 {
  // G1 and G2 generators
  static constexpr storage<fq_config::limbs_count> g1_gen_x = {
    0x66e5b43d, 0x4088f3af, 0xa6af603f, 0x055928ac, 0x56133e82, 0x6750dd03, 0x280ca27f, 0x03758f9a,
    0xc9ea0971, 0x5bd71fa0, 0x47729b90, 0xa17a54ce, 0x94c2e746, 0x11dbfcd2, 0xc15520ac, 0x79017ffa,
    0x85f56fc7, 0xee05c54b, 0x551b27f0, 0xe6a0cfb7, 0xa477beae, 0xb277ce98, 0x0ea190c8, 0x01075b02};
  static constexpr storage<fq_config::limbs_count> g1_gen_y = {
    0xb4e95363, 0xbafc8f2d, 0x0b20d2a1, 0xad1cb2be, 0xcad0fb93, 0xb2b08119, 0xb3053253, 0x9f9df141,
    0x6fc2cdd4, 0xbe3fb90b, 0x717a4c55, 0xcc685d31, 0x71b5b806, 0xc5b8fa17, 0xaf7e0dba, 0x265909f1,
    0xa2e573a3, 0x1a7348d2, 0x884c9ec6, 0x0f952589, 0x45cc2a42, 0xe6fd637b, 0x0a6fc574, 0x0058b84e};
  static constexpr storage<fq_config::limbs_count> g2_gen_x = {
    0xcd025f1c, 0xa830c194, 0xe1bf995b, 0x6410cf4f, 0xc2ad54b0, 0x00e96efb, 0x3cd208d7, 0xce6948cb,
    0x00e1b6ba, 0x963317a3, 0xac70e7c7, 0xc5bbcae9, 0xf09feb58, 0x734ec3f1, 0xab3da268, 0x26b41c5d,
    0x13890f6d, 0x4c062010, 0xc5a7115f, 0xd61053aa, 0x69d660f9, 0xc852a82e, 0x41d9b816, 0x01101332};
  static constexpr storage<fq_config::limbs_count> g2_gen_y = {
    0x28c73b61, 0xeb70a167, 0xf9eac689, 0x91ec0594, 0x3c5a02a5, 0x58aa2d3a, 0x504affc7, 0x3ea96fcd,
    0xffa82300, 0x8906c170, 0xd2c712b8, 0x64f293db, 0x33293fef, 0x94c97eb7, 0x0b95a59c, 0x0a1d86c8,
    0x53ffe316, 0x81a78e27, 0xcec2181c, 0x26b7cf9a, 0xe4b6d2dc, 0x8179eb10, 0x7761369f, 0x0017c335};

  static constexpr storage<fq_config::limbs_count> weierstrass_b = {
    0x0000008a, 0xf49d0000, 0x70000082, 0xe6913e68, 0xeaf0a437, 0x160cf8ae, 0x5667a8f8, 0x98a116c2,
    0x73ebff2e, 0x71dcd3dc, 0x12f9fd90, 0x8689c8ed, 0x25b42304, 0x03cebaff, 0xe584e919, 0x707ba638,
    0x8087be41, 0x528275ef, 0x81d14688, 0xb926186a, 0x04faff3e, 0xd187c940, 0xfb83ce0a, 0x0122e824};
  static constexpr storage<fq_config::limbs_count> g2_weierstrass_b = {
    0x00000004, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};

  CURVE_DEFINITIONS

  typedef point_field_t g2_point_field_t;
  static constexpr g2_point_field_t g2_generator_x = g2_point_field_t{g2_gen_x};
  static constexpr g2_point_field_t g2_generator_y = g2_point_field_t{g2_gen_y};
  static constexpr g2_point_field_t g2_b = g2_point_field_t{g2_weierstrass_b};

  /**
   * [Projective representation](https://hyperelliptic.org/EFD/g1p/auto-shortw-projective.html) of G2 curve.
   */
  typedef Projective<g2_point_field_t, scalar_t, g2_b, g2_generator_x, g2_generator_y> g2_projective_t;
  /**
   * Affine representation of G1 curve.
   */
  typedef Affine<g2_point_field_t> g2_affine_t;
} // namespace bw6_761

#endif
