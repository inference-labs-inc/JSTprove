#pragma once
#ifndef BLS12_381_SCALAR_PARAMS_H
#define BLS12_381_SCALAR_PARAMS_H

#include "fields/storage.cuh"
#include "fields/field.cuh"
#include "fields/params_gen.cuh"

namespace bls12_381 {
  struct fp_config {
    static constexpr storage<8> modulus = {0x00000001, 0xffffffff, 0xfffe5bfe, 0x53bda402,
                                           0x09a1d805, 0x3339d808, 0x299d7d48, 0x73eda753};
    PARAMS(modulus)

    static constexpr storage<8> rou = {0x0b912f1f, 0x1b788f50, 0x70b3e094, 0xc4024ff2,
                                       0xd168d6c0, 0x0fd56dc8, 0x5b416b6f, 0x0212d79e};
    TWIDDLES(modulus, rou)
  };

  /**
   * Scalar field. Is always a prime field.
   */
  typedef Field<fp_config> scalar_t;
} // namespace bls12_381

#endif
