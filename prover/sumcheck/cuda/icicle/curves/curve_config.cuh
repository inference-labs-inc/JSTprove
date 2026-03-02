#pragma once
#ifndef CURVE_CONFIG_H
#define CURVE_CONFIG_H

#include "fields/id.h"
#include "curves/projective.cuh"

/**
 * @namespace curve_config
 * Namespace with type definitions for short Weierstrass pairing-friendly [elliptic
 * curves](https://hyperelliptic.org/EFD/g1p/auto-shortw.html). Here, concrete types are created in accordance
 * with the `-DCURVE` env variable passed during build.
 */
#if CURVE_ID == BN254
#include "curves/params/bn254.cuh"
namespace curve_config = bn254;

#elif CURVE_ID == BLS12_381
#include "curves/params/bls12_381.cuh"
namespace curve_config = bls12_381;

#elif CURVE_ID == BLS12_377
#include "curves/params/bls12_377.cuh"
namespace curve_config = bls12_377;

#elif CURVE_ID == BW6_761
#include "curves/params/bw6_761.cuh"
namespace curve_config = bw6_761;

#elif CURVE_ID == GRUMPKIN
#include "curves/params/grumpkin.cuh"
namespace curve_config = grumpkin;
#endif
#endif
