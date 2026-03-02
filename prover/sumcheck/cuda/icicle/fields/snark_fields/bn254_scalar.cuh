#pragma once
#ifndef BN254_SCALAR_PARAMS_H
#define BN254_SCALAR_PARAMS_H

#include "fields/storage.cuh"
#include "fields/field.cuh"
#include "fields/params_gen.cuh"

namespace bn254 {
  struct fp_config {
    static constexpr storage<8> modulus = {0xf0000001, 0x43e1f593, 0x79b97091, 0x2833e848,
                                           0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72};
    PARAMS(modulus)

    static constexpr storage<8> rou = {0x725b19f0, 0x9bd61b6e, 0x41112ed4, 0x402d111e,
                                       0x8ef62abc, 0x00e0a7eb, 0xa58a7e85, 0x2a3c09f0};
    TWIDDLES(modulus, rou)
  };

  /**
   * Scalar field. Is always a prime field.
   */
  typedef Field<fp_config> scalar_t;
} // namespace bn254

#endif
