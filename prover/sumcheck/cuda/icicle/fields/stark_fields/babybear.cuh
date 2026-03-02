#pragma once

#include "fields/storage.cuh"
#include "fields/field.cuh"
#include "fields/quartic_extension.cuh"
#include "fields/params_gen.cuh"

namespace babybear {
  struct fp_config {
    static constexpr storage<1> modulus = {0x78000001};
    PARAMS(modulus)

    static constexpr storage<1> rou = {0x00000089};
    TWIDDLES(modulus, rou)

    // nonresidue to generate the extension field
    static constexpr uint32_t nonresidue = 11;
    // true if nonresidue is negative.
    static constexpr bool nonresidue_is_negative = false;
  };

  /**
   * Scalar field. Is always a prime field.
   */
  typedef Field<fp_config> scalar_t;

  /**
   * Extension field of `scalar_t` enabled if `-DEXT_FIELD` env variable is.
   */
  typedef ExtensionField<fp_config, scalar_t> extension_t;
} // namespace babybear
