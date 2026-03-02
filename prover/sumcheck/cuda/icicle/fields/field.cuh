/**
 * This file contains methods for working with elements of a prime field. It is based on and evolved from Matter Labs'
 * [Zprize
 * submission](https://github.com/matter-labs/z-prize-msm-gpu/blob/main/bellman-cuda-rust/bellman-cuda-sys/native/ff_dispatch_st.cuh).
 *
 * TODO: DmytroTym: current version needs refactoring (e.g. there's no reason to have different classes Field and
 * ff_storage among other issues). But because this is an internal file and correctness and performance are unaffected,
 * refactoring it is low in the priority list.
 *
 * Documentation of methods is intended to explain inner workings to developers working on icicle. In its current sha3_state
 * it mostly explains modular multiplication and related methods. One important quirk of modern CUDA that's affecting
 * most methods is explained by [Niall Emmart](https://youtu.be/KAWlySN7Hm8?si=h7nzDujnvubWXeDX&t=4039). In short, when
 * 64-bit MAD (`r = a * b + c`) instructions get compiled down to SASS (CUDA assembly) they require two-register values
 * `r` and `c` to start from even register (e.g. `r` can live in registers 20 and 21, or 14 and 15, but not 15 and 16).
 * This complicates implementations forcing us to segregate terms into two categories depending on their alignment.
 * Which is where `even` and `odd` arrays across the codebase come from.
 */

#pragma once

#include "gpu-utils/error_handler.cuh"
#include "gpu-utils/modifiers.cuh"
#include "gpu-utils/sharedmem.cuh"
#include "host_math.cuh"
#include "ptx.cuh"
#include "storage.cuh"

#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

template <class CONFIG>
class Field
{
public:
  static constexpr unsigned TLC = CONFIG::limbs_count;
  static constexpr unsigned NBITS = CONFIG::modulus_bit_count;

  static constexpr HOST_DEVICE_INLINE Field zero() { return Field{CONFIG::zero}; }

  static constexpr HOST_DEVICE_INLINE Field one() { return Field{CONFIG::one}; }

  static constexpr HOST_DEVICE_INLINE Field from(uint32_t value)
  {
    storage<TLC> scalar{};
    scalar.limbs[0] = value;
    for (int i = 1; i < TLC; i++) {
      scalar.limbs[i] = 0;
    }
    return Field{scalar};
  }

  static constexpr HOST_DEVICE_INLINE Field from_u128(__uint128_t value)
  {
    storage<TLC> scalar{};
    scalar.limbs[0] = value;
    scalar.limbs[1] = value >> 32;
    scalar.limbs[2] = value >> 64;
    scalar.limbs[3] = value >> 3 * 32;
    for (int i = 4; i < TLC; i++) {
      scalar.limbs[i] = 0;
    }
    return Field{scalar};
  }

  static HOST_INLINE Field omega(uint32_t logn)
  {
    if (logn == 0) { return Field{CONFIG::one}; }

    if (logn > CONFIG::omegas_count) { THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "Field: Invalid omega index"); }

    Field omega = Field{CONFIG::rou};
    for (int i = 0; i < CONFIG::omegas_count - logn; i++)
      omega = sqr(omega);
    return omega;
  }

  static HOST_INLINE Field omega_inv(uint32_t logn)
  {
    if (logn == 0) { return Field{CONFIG::one}; }

    if (logn > CONFIG::omegas_count) {
      THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "Field: Invalid omega_inv index");
    }

    Field omega = inverse(Field{CONFIG::rou});
    for (int i = 0; i < CONFIG::omegas_count - logn; i++)
      omega = sqr(omega);
    return omega;
  }

  static HOST_DEVICE_INLINE Field inv_log_size(uint32_t logn)
  {
    if (logn == 0) { return Field{CONFIG::one}; }
#ifndef __CUDA_ARCH__
    if (logn > CONFIG::omegas_count) THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "Field: Invalid inv index");
#else
    if (logn > CONFIG::omegas_count) {
      printf(
        "CUDA ERROR: field.cuh: error on inv_log_size(logn): logn(=%u) > omegas_count (=%u)", logn,
        CONFIG::omegas_count);
      assert(false);
    }
#endif // __CUDA_ARCH__
    storage_array<CONFIG::omegas_count, TLC> const inv = CONFIG::inv;
    return Field{inv.storages[logn - 1]};
  }

  static constexpr HOST_INLINE unsigned get_omegas_count()
  {
    if constexpr (has_member_omegas_count<CONFIG>()) {
      return CONFIG::omegas_count;
    } else {
      return 0;
    }
  }

  template <typename T>
  static constexpr bool has_member_omegas_count()
  {
    return sizeof(T::omegas_count) > 0;
  }

  // private:
  typedef storage<TLC> ff_storage;
  typedef storage<2 * TLC> ff_wide_storage;

  /**
   * A new addition to the config file - \f$ 2^{32 \cdot num\_limbs} - p \f$.
   */
  static constexpr HOST_DEVICE_INLINE ff_storage get_neg_modulus() { return CONFIG::neg_modulus; }

  /**
   * A new addition to the config file - the number of times to reduce in [reduce](@ref reduce) function.
   */
  static constexpr HOST_DEVICE_INLINE unsigned num_of_reductions() { return CONFIG::num_of_reductions; }

  static constexpr unsigned slack_bits = 32 * TLC - NBITS;

  struct Wide {
    ff_wide_storage limbs_storage;

    static constexpr Wide HOST_DEVICE_INLINE from_field(const Field& xs)
    {
      Wide out{};
#ifdef __CUDA_ARCH__
      UNROLL
#endif
      for (unsigned i = 0; i < TLC; i++)
        out.limbs_storage.limbs[i] = xs.limbs_storage.limbs[i];
      return out;
    }

    static constexpr Field HOST_DEVICE_INLINE get_lower(const Wide& xs)
    {
      Field out{};
#ifdef __CUDA_ARCH__
      UNROLL
#endif
      for (unsigned i = 0; i < TLC; i++)
        out.limbs_storage.limbs[i] = xs.limbs_storage.limbs[i];
      return out;
    }

    static constexpr Field HOST_DEVICE_INLINE get_higher(const Wide& xs)
    {
      Field out{};
#ifdef __CUDA_ARCH__
      UNROLL
#endif
      for (unsigned i = 0; i < TLC; i++)
        out.limbs_storage.limbs[i] = xs.limbs_storage.limbs[i + TLC];
      return out;
    }

    static constexpr Field HOST_DEVICE_INLINE get_higher_with_slack(const Wide& xs)
    {
      Field out{};
#ifdef __CUDA_ARCH__
      UNROLL
#endif
      for (unsigned i = 0; i < TLC; i++) {
#ifdef __CUDA_ARCH__
        out.limbs_storage.limbs[i] =
          __funnelshift_lc(xs.limbs_storage.limbs[i + TLC - 1], xs.limbs_storage.limbs[i + TLC], 2 * slack_bits);
#else
        out.limbs_storage.limbs[i] = (xs.limbs_storage.limbs[i + TLC] << 2 * slack_bits) +
                                     (xs.limbs_storage.limbs[i + TLC - 1] >> (32 - 2 * slack_bits));
#endif
      }
      return out;
    }

    template <unsigned REDUCTION_SIZE = 1>
    static constexpr HOST_DEVICE_INLINE Wide sub_modulus_squared(const Wide& xs)
    {
      if (REDUCTION_SIZE == 0) return xs;
      const ff_wide_storage modulus = get_modulus_squared<REDUCTION_SIZE>();
      Wide rs = {};
      return sub_limbs<2 * TLC, true>(xs.limbs_storage, modulus, rs.limbs_storage) ? xs : rs;
    }

    template <unsigned MODULUS_MULTIPLE = 1>
    static constexpr HOST_DEVICE_INLINE Wide neg(const Wide& xs)
    {
      const ff_wide_storage modulus = get_modulus_squared<MODULUS_MULTIPLE>();
      Wide rs = {};
      sub_limbs<2 * TLC, false>(modulus, xs.limbs_storage, rs.limbs_storage);
      return rs;
    }

    friend HOST_DEVICE_INLINE Wide operator+(Wide xs, const Wide& ys)
    {
      Wide rs = {};
      add_limbs<2 * TLC, false>(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
      return sub_modulus_squared<1>(rs);
    }

    friend HOST_DEVICE_INLINE Wide operator-(Wide xs, const Wide& ys)
    {
      Wide rs = {};
      uint32_t carry = sub_limbs<2 * TLC, true>(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
      if (carry == 0) return rs;
      const ff_wide_storage modulus = get_modulus_squared<1>();
      add_limbs<2 * TLC, false>(rs.limbs_storage, modulus, rs.limbs_storage);
      return rs;
    }
  };

  // return modulus multiplied by 1, 2 or 4
  template <unsigned MULTIPLIER = 1>
  static constexpr HOST_DEVICE_INLINE ff_storage get_modulus()
  {
    switch (MULTIPLIER) {
    case 1:
      return CONFIG::modulus;
    case 2:
      return CONFIG::modulus_2;
    case 4:
      return CONFIG::modulus_4;
    default:
      return {};
    }
  }

  // return m
  static constexpr HOST_DEVICE_INLINE ff_storage get_m() { return CONFIG::m; }

  // return modulus^2, helpful for ab +/- cd
  template <unsigned MULTIPLIER = 1>
  static constexpr HOST_DEVICE_INLINE ff_wide_storage get_modulus_squared()
  {
    switch (MULTIPLIER) {
    case 1:
      return CONFIG::modulus_squared;
    case 2:
      return CONFIG::modulus_squared_2;
    case 4:
      return CONFIG::modulus_squared_4;
    default:
      return {};
    }
  }

  template <unsigned NLIMBS, bool SUBTRACT, bool CARRY_OUT>
  static constexpr DEVICE_INLINE uint32_t add_sub_u32_device(const uint32_t* x, const uint32_t* y, uint32_t* r)
  {
    r[0] = SUBTRACT ? ptx::sub_cc(x[0], y[0]) : ptx::add_cc(x[0], y[0]);
    for (unsigned i = 1; i < NLIMBS; i++)
      r[i] = SUBTRACT ? ptx::subc_cc(x[i], y[i]) : ptx::addc_cc(x[i], y[i]);
    if (!CARRY_OUT) {
      ptx::addc(0, 0);
      return 0;
    }
    return SUBTRACT ? ptx::subc(0, 0) : ptx::addc(0, 0);
  }

  template <unsigned NLIMBS, bool SUBTRACT, bool CARRY_OUT>
  static constexpr DEVICE_INLINE uint32_t
  add_sub_limbs_device(const storage<NLIMBS>& xs, const storage<NLIMBS>& ys, storage<NLIMBS>& rs)
  {
    const uint32_t* x = xs.limbs;
    const uint32_t* y = ys.limbs;
    uint32_t* r = rs.limbs;
    return add_sub_u32_device<NLIMBS, SUBTRACT, CARRY_OUT>(x, y, r);
  }

  template <unsigned NLIMBS, bool CARRY_OUT>
  static constexpr HOST_DEVICE_INLINE uint32_t
  add_limbs(const storage<NLIMBS>& xs, const storage<NLIMBS>& ys, storage<NLIMBS>& rs)
  {
#ifdef __CUDA_ARCH__
    return add_sub_limbs_device<NLIMBS, false, CARRY_OUT>(xs, ys, rs);
#else
    return host_math::template add_sub_limbs<NLIMBS, false, CARRY_OUT>(xs, ys, rs);
#endif
  }

  template <unsigned NLIMBS, bool CARRY_OUT>
  static constexpr HOST_DEVICE_INLINE uint32_t
  sub_limbs(const storage<NLIMBS>& xs, const storage<NLIMBS>& ys, storage<NLIMBS>& rs)
  {
#ifdef __CUDA_ARCH__
    return add_sub_limbs_device<NLIMBS, true, CARRY_OUT>(xs, ys, rs);
#else
    return host_math::template add_sub_limbs<NLIMBS, true, CARRY_OUT>(xs, ys, rs);
#endif
  }

  static DEVICE_INLINE void mul_n(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n = TLC)
  {
    UNROLL
    for (size_t i = 0; i < n; i += 2) {
      acc[i] = ptx::mul_lo(a[i], bi);
      acc[i + 1] = ptx::mul_hi(a[i], bi);
    }
  }

  static DEVICE_INLINE void mul_n_msb(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n = TLC, size_t start_i = 0)
  {
    UNROLL
    for (size_t i = start_i; i < n; i += 2) {
      acc[i] = ptx::mul_lo(a[i], bi);
      acc[i + 1] = ptx::mul_hi(a[i], bi);
    }
  }

  template <bool CARRY_IN = false>
  static DEVICE_INLINE void
  cmad_n(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n = TLC, uint32_t optional_carry = 0)
  {
    if (CARRY_IN) ptx::add_cc(UINT32_MAX, optional_carry);
    acc[0] = CARRY_IN ? ptx::madc_lo_cc(a[0], bi, acc[0]) : ptx::mad_lo_cc(a[0], bi, acc[0]);
    acc[1] = ptx::madc_hi_cc(a[0], bi, acc[1]);

    UNROLL
    for (size_t i = 2; i < n; i += 2) {
      acc[i] = ptx::madc_lo_cc(a[i], bi, acc[i]);
      acc[i + 1] = ptx::madc_hi_cc(a[i], bi, acc[i + 1]);
    }
  }

  template <bool EVEN_PHASE>
  static DEVICE_INLINE void cmad_n_msb(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n = TLC)
  {
    if (EVEN_PHASE) {
      acc[0] = ptx::mad_lo_cc(a[0], bi, acc[0]);
      acc[1] = ptx::madc_hi_cc(a[0], bi, acc[1]);
    } else {
      acc[1] = ptx::mad_hi_cc(a[0], bi, acc[1]);
    }

    UNROLL
    for (size_t i = 2; i < n; i += 2) {
      acc[i] = ptx::madc_lo_cc(a[i], bi, acc[i]);
      acc[i + 1] = ptx::madc_hi_cc(a[i], bi, acc[i + 1]);
    }
  }

  static DEVICE_INLINE void cmad_n_lsb(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n = TLC)
  {
    if (n > 1)
      acc[0] = ptx::mad_lo_cc(a[0], bi, acc[0]);
    else
      acc[0] = ptx::mad_lo(a[0], bi, acc[0]);

    size_t i;
    UNROLL
    for (i = 1; i < n - 1; i += 2) {
      acc[i] = ptx::madc_hi_cc(a[i - 1], bi, acc[i]);
      if (i == n - 2)
        acc[i + 1] = ptx::madc_lo(a[i + 1], bi, acc[i + 1]);
      else
        acc[i + 1] = ptx::madc_lo_cc(a[i + 1], bi, acc[i + 1]);
    }
    if (i == n - 1) acc[i] = ptx::madc_hi(a[i - 1], bi, acc[i]);
  }

  template <bool CARRY_OUT = false, bool CARRY_IN = false>
  static DEVICE_INLINE uint32_t mad_row(
    uint32_t* odd,
    uint32_t* even,
    const uint32_t* a,
    uint32_t bi,
    size_t n = TLC,
    uint32_t ci = 0,
    uint32_t di = 0,
    uint32_t carry_for_high = 0,
    uint32_t carry_for_low = 0)
  {
    cmad_n<CARRY_IN>(odd, a + 1, bi, n - 2, carry_for_low);
    odd[n - 2] = ptx::madc_lo_cc(a[n - 1], bi, ci);
    odd[n - 1] = CARRY_OUT ? ptx::madc_hi_cc(a[n - 1], bi, di) : ptx::madc_hi(a[n - 1], bi, di);
    uint32_t cr = CARRY_OUT ? ptx::addc(0, 0) : 0;
    cmad_n(even, a, bi, n);
    if (CARRY_OUT) {
      odd[n - 1] = ptx::addc_cc(odd[n - 1], carry_for_high);
      cr = ptx::addc(cr, 0);
    } else
      odd[n - 1] = ptx::addc(odd[n - 1], carry_for_high);
    return cr;
  }

  template <bool EVEN_PHASE>
  static DEVICE_INLINE void mad_row_msb(uint32_t* odd, uint32_t* even, const uint32_t* a, uint32_t bi, size_t n = TLC)
  {
    cmad_n_msb<!EVEN_PHASE>(odd, EVEN_PHASE ? a : (a + 1), bi, n - 2);
    odd[EVEN_PHASE ? (n - 1) : (n - 2)] = ptx::madc_lo_cc(a[n - 1], bi, 0);
    odd[EVEN_PHASE ? n : (n - 1)] = ptx::madc_hi(a[n - 1], bi, 0);
    cmad_n_msb<EVEN_PHASE>(even, EVEN_PHASE ? (a + 1) : a, bi, n - 1);
    odd[EVEN_PHASE ? n : (n - 1)] = ptx::addc(odd[EVEN_PHASE ? n : (n - 1)], 0);
  }

  static DEVICE_INLINE void mad_row_lsb(uint32_t* odd, uint32_t* even, const uint32_t* a, uint32_t bi, size_t n = TLC)
  {
    // bi here is constant so we can do a compile-time check for zero (which does happen once for bls12-381 scalar field
    // modulus)
    if (bi != 0) {
      if (n > 1) cmad_n_lsb(odd, a + 1, bi, n - 1);
      cmad_n_lsb(even, a, bi, n);
    }
    return;
  }

  static DEVICE_INLINE uint32_t
  mul_n_and_add(uint32_t* acc, const uint32_t* a, uint32_t bi, uint32_t* extra, size_t n = (TLC >> 1))
  {
    acc[0] = ptx::mad_lo_cc(a[0], bi, extra[0]);

    UNROLL
    for (size_t i = 1; i < n - 1; i += 2) {
      acc[i] = ptx::madc_hi_cc(a[i - 1], bi, extra[i]);
      acc[i + 1] = ptx::madc_lo_cc(a[i + 1], bi, extra[i + 1]);
    }

    acc[n - 1] = ptx::madc_hi_cc(a[n - 2], bi, extra[n - 1]);
    return ptx::addc(0, 0);
  }

  /**
   * A function that computes wide product \f$ rs = as \cdot bs \f$ that's correct for the higher TLC + 1 limbs with a
   * small maximum error.
   *
   * The way this function saves computations (as compared to regular school-book multiplication) is by not including
   * terms that are too small. Namely, limb product \f$ a_i \cdot b_j \f$ is excluded if \f$ i + j < TLC - 2 \f$ and
   * only the higher half is included if \f$ i + j = TLC - 2 \f$. All other limb products are included. So, the error
   * i.e. difference between true product and the result of this function written to `rs` is exactly the sum of all
   * dropped limbs products, which we can bound: \f$ a_0 \cdot b_0 + 2^{32}(a_0 \cdot b_1 + a_1 \cdot b_0) + \dots +
   * 2^{32(TLC - 3)}(a_{TLC - 3} \cdot b_0 + \dots + a_0 \cdot b_{TLC - 3}) + 2^{32(TLC - 2)}(\floor{\frac{a_{TLC - 2}
   * \cdot b_0}{2^{32}}} + \dots + \floor{\frac{a_0 \cdot b_{TLC - 2}}{2^{32}}}) \leq 2^{64} + 2\cdot 2^{96} + \dots +
   * (TLC - 2) \cdot 2^{32(TLC - 1)} + (TLC - 1) \cdot 2^{32(TLC - 1)} \leq 2(TLC - 1) \cdot 2^{32(TLC - 1)}\f$.
   */
  static DEVICE_INLINE void multiply_msb_raw_device(const ff_storage& as, const ff_storage& bs, ff_wide_storage& rs)
  {
    if constexpr (TLC > 1) {
      const uint32_t* a = as.limbs;
      const uint32_t* b = bs.limbs;
      uint32_t* even = rs.limbs;
      __align__(16) uint32_t odd[2 * TLC - 2];

      even[TLC - 1] = ptx::mul_hi(a[TLC - 2], b[0]);
      odd[TLC - 2] = ptx::mul_lo(a[TLC - 1], b[0]);
      odd[TLC - 1] = ptx::mul_hi(a[TLC - 1], b[0]);
      size_t i;
      UNROLL
      for (i = 2; i < TLC - 1; i += 2) {
        mad_row_msb<true>(&even[TLC - 2], &odd[TLC - 2], &a[TLC - i - 1], b[i - 1], i + 1);
        mad_row_msb<false>(&odd[TLC - 2], &even[TLC - 2], &a[TLC - i - 2], b[i], i + 2);
      }
      mad_row(&even[TLC], &odd[TLC - 2], a, b[TLC - 1]);

      // merge |even| and |odd|
      ptx::add_cc(even[TLC - 1], odd[TLC - 2]);
      for (i = TLC - 1; i < 2 * TLC - 2; i++)
        even[i + 1] = ptx::addc_cc(even[i + 1], odd[i]);
      even[i + 1] = ptx::addc(even[i + 1], 0);
    } else {
      multiply_raw_device(as, bs, rs);
    }
  }

  /**
   * A function that computes the low half of the fused multiply-and-add \f$ rs = as \cdot bs + cs \f$ where
   * \f$ bs = 2^{32*nof_limbs} \f$.
   *
   * For efficiency, this method does not include terms that are too large. Namely, limb product \f$ a_i \cdot b_j \f$
   * is excluded if \f$ i + j > TLC - 1 \f$ and only the lower half is included if \f$ i + j = TLC - 1 \f$. All other
   * limb products are included.
   */
  static DEVICE_INLINE void
  multiply_and_add_lsb_neg_modulus_raw_device(const ff_storage& as, ff_storage& cs, ff_storage& rs)
  {
    ff_storage bs = get_neg_modulus();
    const uint32_t* a = as.limbs;
    const uint32_t* b = bs.limbs;
    uint32_t* c = cs.limbs;
    uint32_t* even = rs.limbs;

    if constexpr (TLC > 2) {
      __align__(16) uint32_t odd[TLC - 1];
      size_t i;
      // `b[0]` is \f$ 2^{32} \f$ minus the last limb of prime modulus. Because most scalar (and some base) primes
      // are necessarily NTT-friendly, `b[0]` often turns out to be \f$ 2^{32} - 1 \f$. This actually leads to
      // less efficient SASS generated by nvcc, so this case needed separate handling.
      if (b[0] == UINT32_MAX) {
        add_sub_u32_device<TLC, true, false>(c, a, even);
        for (i = 0; i < TLC - 1; i++)
          odd[i] = a[i];
      } else {
        mul_n_and_add(even, a, b[0], c, TLC);
        mul_n(odd, a + 1, b[0], TLC - 1);
      }
      mad_row_lsb(&even[2], &odd[0], a, b[1], TLC - 1);
      UNROLL
      for (i = 2; i < TLC - 1; i += 2) {
        mad_row_lsb(&odd[i], &even[i], a, b[i], TLC - i);
        mad_row_lsb(&even[i + 2], &odd[i], a, b[i + 1], TLC - i - 1);
      }

      // merge |even| and |odd|
      even[1] = ptx::add_cc(even[1], odd[0]);
      for (i = 1; i < TLC - 2; i++)
        even[i + 1] = ptx::addc_cc(even[i + 1], odd[i]);
      even[i + 1] = ptx::addc(even[i + 1], odd[i]);
    } else if (TLC == 2) {
      even[0] = ptx::mad_lo(a[0], b[0], c[0]);
      even[1] = ptx::mad_hi(a[0], b[0], c[0]);
      even[1] = ptx::mad_lo(a[0], b[1], even[1]);
      even[1] = ptx::mad_lo(a[1], b[0], even[1]);
    } else if (TLC == 1) {
      even[0] = ptx::mad_lo(a[0], b[0], c[0]);
    }
  }

  /**
   * This method multiplies `a` and `b` (both assumed to have TLC / 2 limbs) and adds `in1` and `in2` (TLC limbs each)
   * to the result which is written to `even`.
   *
   * It is used to compute the "middle" part of Karatsuba: \f$ a_{lo} \cdot b_{hi} + b_{lo} \cdot a_{hi} =
   * (a_{hi} - a_{lo})(b_{lo} - b_{hi}) + a_{lo} \cdot b_{lo} + a_{hi} \cdot b_{hi} \f$. Currently this method assumes
   * that the top bit of \f$ a_{hi} \f$ and \f$ b_{hi} \f$ are unset. This ensures correctness by allowing to keep the
   * result inside TLC limbs and ignore the carries from the highest limb.
   */
  static DEVICE_INLINE void
  multiply_and_add_short_raw_device(const uint32_t* a, const uint32_t* b, uint32_t* even, uint32_t* in1, uint32_t* in2)
  {
    __align__(16) uint32_t odd[TLC - 2];
    uint32_t first_row_carry = mul_n_and_add(even, a, b[0], in1);
    uint32_t carry = mul_n_and_add(odd, a + 1, b[0], &in2[1]);

    size_t i;
    UNROLL
    for (i = 2; i < ((TLC >> 1) - 1); i += 2) {
      carry = mad_row<true, false>(
        &even[i], &odd[i - 2], a, b[i - 1], TLC >> 1, in1[(TLC >> 1) + i - 2], in1[(TLC >> 1) + i - 1], carry);
      carry =
        mad_row<true, false>(&odd[i], &even[i], a, b[i], TLC >> 1, in2[(TLC >> 1) + i - 1], in2[(TLC >> 1) + i], carry);
    }
    mad_row<false, true>(
      &even[TLC >> 1], &odd[(TLC >> 1) - 2], a, b[(TLC >> 1) - 1], TLC >> 1, in1[TLC - 2], in1[TLC - 1], carry,
      first_row_carry);
    // merge |even| and |odd| plus the parts of `in2` we haven't added yet (first and last limbs)
    even[0] = ptx::add_cc(even[0], in2[0]);
    for (i = 0; i < (TLC - 2); i++)
      even[i + 1] = ptx::addc_cc(even[i + 1], odd[i]);
    even[i + 1] = ptx::addc(even[i + 1], in2[i + 1]);
  }

  /**
   * This method multiplies `a` and `b` and writes the result into `even`. It assumes that `a` and `b` are TLC/2 limbs
   * long. The usual schoolbook algorithm is used.
   */
  static DEVICE_INLINE void multiply_short_raw_device(const uint32_t* a, const uint32_t* b, uint32_t* even)
  {
    __align__(16) uint32_t odd[TLC - 2];
    mul_n(even, a, b[0], TLC >> 1);
    mul_n(odd, a + 1, b[0], TLC >> 1);
    mad_row(&even[2], &odd[0], a, b[1], TLC >> 1);

    size_t i;
    UNROLL
    for (i = 2; i < ((TLC >> 1) - 1); i += 2) {
      mad_row(&odd[i], &even[i], a, b[i], TLC >> 1);
      mad_row(&even[i + 2], &odd[i], a, b[i + 1], TLC >> 1);
    }
    // merge |even| and |odd|
    even[1] = ptx::add_cc(even[1], odd[0]);
    for (i = 1; i < TLC - 2; i++)
      even[i + 1] = ptx::addc_cc(even[i + 1], odd[i]);
    even[i + 1] = ptx::addc(even[i + 1], 0);
  }

  /**
   * This method multiplies `as` and `bs` and writes the (wide) result into `rs`.
   *
   * It is assumed that the highest bits of `as` and `bs` are unset which is true for all the numbers icicle had to deal
   * with so far. This method implements [subtractive
   * Karatsuba](https://en.wikipedia.org/wiki/Karatsuba_algorithm#Implementation).
   */
  static DEVICE_INLINE void multiply_raw_device(const ff_storage& as, const ff_storage& bs, ff_wide_storage& rs)
  {
    const uint32_t* a = as.limbs;
    const uint32_t* b = bs.limbs;
    uint32_t* r = rs.limbs;
    if constexpr (TLC > 2) {
      // Next two lines multiply high and low halves of operands (\f$ a_{lo} \cdot b_{lo}; a_{hi} \cdot b_{hi} \$f) and
      // write the results into `r`.
      multiply_short_raw_device(a, b, r);
      multiply_short_raw_device(&a[TLC >> 1], &b[TLC >> 1], &r[TLC]);
      __align__(16) uint32_t middle_part[TLC];
      __align__(16) uint32_t diffs[TLC];
      // Differences of halves \f$ a_{hi} - a_{lo}; b_{lo} - b_{hi} \$f are written into `diffs`, signs written to
      // `carry1` and `carry2`.
      uint32_t carry1 = add_sub_u32_device<(TLC >> 1), true, true>(&a[TLC >> 1], a, diffs);
      uint32_t carry2 = add_sub_u32_device<(TLC >> 1), true, true>(b, &b[TLC >> 1], &diffs[TLC >> 1]);
      // Compute the "middle part" of Karatsuba: \f$ a_{lo} \cdot b_{hi} + b_{lo} \cdot a_{hi} \f$.
      // This is where the assumption about unset high bit of `a` and `b` is relevant.
      multiply_and_add_short_raw_device(diffs, &diffs[TLC >> 1], middle_part, r, &r[TLC]);
      // Corrections that need to be performed when differences are negative.
      // Again, carry doesn't need to be propagated due to unset high bits of `a` and `b`.
      if (carry1)
        add_sub_u32_device<(TLC >> 1), true, false>(&middle_part[TLC >> 1], &diffs[TLC >> 1], &middle_part[TLC >> 1]);
      if (carry2) add_sub_u32_device<(TLC >> 1), true, false>(&middle_part[TLC >> 1], diffs, &middle_part[TLC >> 1]);
      // Now that middle part is fully correct, it can be added to the result.
      add_sub_u32_device<TLC, false, true>(&r[TLC >> 1], middle_part, &r[TLC >> 1]);

      // Carry from adding middle part has to be propagated to the highest limb.
      for (size_t i = TLC + (TLC >> 1); i < 2 * TLC; i++)
        r[i] = ptx::addc_cc(r[i], 0);
    } else if (TLC == 2) {
      __align__(8) uint32_t odd[2];
      r[0] = ptx::mul_lo(a[0], b[0]);
      r[1] = ptx::mul_hi(a[0], b[0]);
      r[2] = ptx::mul_lo(a[1], b[1]);
      r[3] = ptx::mul_hi(a[1], b[1]);
      odd[0] = ptx::mul_lo(a[0], b[1]);
      odd[1] = ptx::mul_hi(a[0], b[1]);
      odd[0] = ptx::mad_lo(a[1], b[0], odd[0]);
      odd[1] = ptx::mad_hi(a[1], b[0], odd[1]);
      r[1] = ptx::add_cc(r[1], odd[0]);
      r[2] = ptx::addc_cc(r[2], odd[1]);
      r[3] = ptx::addc(r[3], 0);
    } else if (TLC == 1) {
      r[0] = ptx::mul_lo(a[0], b[0]);
      r[1] = ptx::mul_hi(a[0], b[0]);
    }
  }

  static HOST_DEVICE_INLINE void multiply_raw(const ff_storage& as, const ff_storage& bs, ff_wide_storage& rs)
  {
#ifdef __CUDA_ARCH__
    return multiply_raw_device(as, bs, rs);
#else
    return host_math::template multiply_raw<TLC>(as, bs, rs);
#endif
  }

  static HOST_DEVICE_INLINE void
  multiply_and_add_lsb_neg_modulus_raw(const ff_storage& as, ff_storage& cs, ff_storage& rs)
  {
#ifdef __CUDA_ARCH__
    return multiply_and_add_lsb_neg_modulus_raw_device(as, cs, rs);
#else
    Wide r_wide = {};
    host_math::template multiply_raw<TLC>(as, get_neg_modulus(), r_wide.limbs_storage);
    Field r = Wide::get_lower(r_wide);
    add_limbs<TLC, false>(cs, r.limbs_storage, rs);
#endif
  }

  static HOST_DEVICE_INLINE void multiply_msb_raw(const ff_storage& as, const ff_storage& bs, ff_wide_storage& rs)
  {
#ifdef __CUDA_ARCH__
    return multiply_msb_raw_device(as, bs, rs);
#else
    return host_math::template multiply_raw<TLC>(as, bs, rs);
#endif
  }

public:
  ff_storage limbs_storage;

  HOST_DEVICE_INLINE uint32_t* export_limbs() { return (uint32_t*)limbs_storage.limbs; }

  HOST_DEVICE_INLINE unsigned get_scalar_digit(unsigned digit_num, unsigned digit_width) const
  {
    const uint32_t limb_lsb_idx = (digit_num * digit_width) / 32;
    const uint32_t shift_bits = (digit_num * digit_width) % 32;
    unsigned rv = limbs_storage.limbs[limb_lsb_idx] >> shift_bits;
    if ((shift_bits + digit_width > 32) && (limb_lsb_idx + 1 < TLC)) {
      rv += limbs_storage.limbs[limb_lsb_idx + 1] << (32 - shift_bits);
    }
    rv &= ((1 << digit_width) - 1);
    return rv;
  }

  static HOST_INLINE Field rand_host()
  {
    std::random_device rd;
    std::mt19937_64 generator(rd());
    std::uniform_int_distribution<unsigned> distribution;
    Field value{};
    for (unsigned i = 0; i < TLC; i++)
      value.limbs_storage.limbs[i] = distribution(generator);
    while (lt(Field{get_modulus()}, value))
      value = value - Field{get_modulus()};
    return value;
  }

  static void rand_host_many(Field* out, int size)
  {
    for (int i = 0; i < size; i++)
      out[i] = rand_host();
  }

  template <unsigned REDUCTION_SIZE = 1>
  static constexpr HOST_DEVICE_INLINE Field sub_modulus(const Field& xs)
  {
    if (REDUCTION_SIZE == 0) return xs;
    const ff_storage modulus = get_modulus<REDUCTION_SIZE>();
    Field rs = {};
    return sub_limbs<TLC, true>(xs.limbs_storage, modulus, rs.limbs_storage) ? xs : rs;
  }

  friend std::ostream& operator<<(std::ostream& os, const Field& xs)
  {
    std::stringstream hex_string;
    hex_string << std::hex << std::setfill('0');

    for (int i = 0; i < TLC; i++) {
      hex_string << std::setw(8) << xs.limbs_storage.limbs[TLC - i - 1];
    }

    os << "0x" << hex_string.str();
    return os;
  }

  friend HOST_DEVICE_INLINE Field operator+(Field xs, const Field& ys)
  {
    Field rs = {};
    add_limbs<TLC, false>(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
    return sub_modulus<1>(rs);
  }

  friend HOST_DEVICE_INLINE Field operator-(Field xs, const Field& ys)
  {
    Field rs = {};
    uint32_t carry = sub_limbs<TLC, true>(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
    if (carry == 0) return rs;
    const ff_storage modulus = get_modulus<1>();
    add_limbs<TLC, false>(rs.limbs_storage, modulus, rs.limbs_storage);
    return rs;
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Wide mul_wide(const Field& xs, const Field& ys)
  {
    Wide rs = {};
    multiply_raw(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
    return rs;
  }

  /**
   * This method reduces a Wide number `xs` modulo `p` and returns the result as a Field element.
   *
   * It is assumed that the high `2 * slack_bits` bits of `xs` are unset which is always the case for the product of 2
   * numbers with their high `slack_bits` unset. Larger Wide numbers should be reduced by subtracting an appropriate
   * factor of `modulus_squared` first.
   *
   * This function implements ["multi-precision Barrett"](https://github.com/ingonyama-zk/modular_multiplication). As
   * opposed to Montgomery reduction, it doesn't require numbers to have a special representation but lets us work with
   * them as-is. The general idea of Barrett reduction is to estimate the quotient \f$ l \approx \floor{\frac{xs}{p}}
   * \f$ and return \f$ xs - l \cdot p \f$. But since \f$ l \f$ is inevitably computed with an error (it's always less
   * or equal than the real quotient). So the modulus `p` might need to be subtracted several times before the result is
   * in the desired range \f$ [0;p-1] \f$. The estimate of the error is as follows: \f[ \frac{xs}{p} - l = \frac{xs}{p}
   * - \frac{xs \cdot m}{2^{2n}} + \frac{xs \cdot m}{2^{2n}} - \floor{\frac{xs}{2^k}}\frac{m}{2^{2n-k}}
   *  + \floor{\frac{xs}{2^k}}\frac{m}{2^{2n-k}} - l \leq p^2(\frac{1}{p}-\frac{m}{2^{2n}}) + \frac{m}{2^{2n-k}} + 2(TLC
   * - 1) \cdot 2^{-32} \f] Here \f$ l \f$ is the result of [multiply_msb_raw](@ref multiply_msb_raw) function and the
   * last term in the error is due to its approximation. \f$ n \f$ is the number of bits in \f$ p \f$ and \f$ k = 2n -
   * 32\cdot TLC \f$. Overall, the error is always less than 2 so at most 2 reductions are needed. However, in most
   * cases it's less than 1, so setting the [num_of_reductions](@ref num_of_reductions) variable for a field equal to 1
   * will cause only 1 reduction to be performed.
   */
  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Field reduce(const Wide& xs)
  {
    // `xs` is left-shifted by `2 * slack_bits` and higher half is written to `xs_hi`
    Field xs_hi = Wide::get_higher_with_slack(xs);
    Wide l = {};
    multiply_msb_raw(xs_hi.limbs_storage, get_m(), l.limbs_storage); // MSB mult by `m`
    Field l_hi = Wide::get_higher(l);
    Field r = {};
    Field xs_lo = Wide::get_lower(xs);
    // Here we need to compute the lsb of `xs - l \cdot p` and to make use of fused multiply-and-add, we rewrite it as
    // `xs + l \cdot (2^{32 \cdot TLC}-p)` which is the same as original (up to higher limbs which we don't care about).
    multiply_and_add_lsb_neg_modulus_raw(l_hi.limbs_storage, xs_lo.limbs_storage, r.limbs_storage);
    ff_storage r_reduced = {};
    uint32_t carry;
    // As mentioned, either 2 or 1 reduction can be performed depending on the field in question.
    if (num_of_reductions() == 2) {
      carry = sub_limbs<TLC, true>(r.limbs_storage, get_modulus<2>(), r_reduced);
      if (carry == 0) r = Field{r_reduced};
    }
    carry = sub_limbs<TLC, true>(r.limbs_storage, get_modulus<1>(), r_reduced);
    if (carry == 0) r = Field{r_reduced};

    return r;
  }

  HOST_DEVICE_INLINE Field& operator=(Field const& other)
  {
    for (int i = 0; i < TLC; i++) {
      this->limbs_storage.limbs[i] = other.limbs_storage.limbs[i];
    }
    return *this;
  }

  friend HOST_DEVICE_INLINE Field operator*(const Field& xs, const Field& ys)
  {
    Wide xy = mul_wide(xs, ys); // full mult
    return reduce(xy);          // reduce mod p
  }

  friend HOST_DEVICE_INLINE bool operator==(const Field& xs, const Field& ys)
  {
#ifdef __CUDA_ARCH__
    const uint32_t* x = xs.limbs_storage.limbs;
    const uint32_t* y = ys.limbs_storage.limbs;
    uint32_t limbs_or = x[0] ^ y[0];
    UNROLL
    for (unsigned i = 1; i < TLC; i++)
      limbs_or |= x[i] ^ y[i];
    return limbs_or == 0;
#else
    for (unsigned i = 0; i < TLC; i++)
      if (xs.limbs_storage.limbs[i] != ys.limbs_storage.limbs[i]) return false;
    return true;
#endif
  }

  friend HOST_DEVICE_INLINE bool operator!=(const Field& xs, const Field& ys) { return !(xs == ys); }

  template <const Field& multiplier>
  static HOST_DEVICE_INLINE Field mul_const(const Field& xs)
  {
    Field mul = multiplier;
    static bool is_u32 = true;
#ifdef __CUDA_ARCH__
    UNROLL
#endif
    for (unsigned i = 1; i < TLC; i++)
      is_u32 &= (mul.limbs_storage.limbs[i] == 0);

    if (is_u32) return mul_unsigned<multiplier.limbs_storage.limbs[0], Field>(xs);
    return mul * xs;
  }

  template <uint32_t multiplier, class T, unsigned REDUCTION_SIZE = 1>
  static constexpr HOST_DEVICE_INLINE T mul_unsigned(const T& xs)
  {
    T rs = {};
    T temp = xs;
    bool is_zero = true;
#ifdef __CUDA_ARCH__
    UNROLL
#endif
    for (unsigned i = 0; i < 32; i++) {
      if (multiplier & (1 << i)) {
        rs = is_zero ? temp : (rs + temp);
        is_zero = false;
      }
      if (multiplier & ((1 << (31 - i) - 1) << (i + 1))) break;
      temp = temp + temp;
    }
    return rs;
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Wide sqr_wide(const Field& xs)
  {
    // TODO: change to a more efficient squaring
    return mul_wide<MODULUS_MULTIPLE>(xs, xs);
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Field sqr(const Field& xs)
  {
    // TODO: change to a more efficient squaring
    return xs * xs;
  }

  static constexpr HOST_DEVICE_INLINE Field to_montgomery(const Field& xs) { return xs * Field{CONFIG::montgomery_r}; }

  static constexpr HOST_DEVICE_INLINE Field from_montgomery(const Field& xs)
  {
    return xs * Field{CONFIG::montgomery_r_inv};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Field neg(const Field& xs)
  {
    const ff_storage modulus = get_modulus<MODULUS_MULTIPLE>();
    Field rs = {};
    sub_limbs<TLC, false>(modulus, xs.limbs_storage, rs.limbs_storage);
    return rs;
  }

  // Assumes the number is even!
  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Field div2(const Field& xs)
  {
    const uint32_t* x = xs.limbs_storage.limbs;
    Field rs = {};
    uint32_t* r = rs.limbs_storage.limbs;
    if constexpr (TLC > 1) {
#ifdef __CUDA_ARCH__
      UNROLL
#endif
      for (unsigned i = 0; i < TLC - 1; i++) {
#ifdef __CUDA_ARCH__
        r[i] = __funnelshift_rc(x[i], x[i + 1], 1);
#else
        r[i] = (x[i] >> 1) | (x[i + 1] << 31);
#endif
      }
    }
    r[TLC - 1] = x[TLC - 1] >> 1;
    return sub_modulus<MODULUS_MULTIPLE>(rs);
  }

  static constexpr HOST_DEVICE_INLINE bool lt(const Field& xs, const Field& ys)
  {
    ff_storage dummy = {};
    uint32_t carry = sub_limbs<TLC, true>(xs.limbs_storage, ys.limbs_storage, dummy);
    return carry;
  }

  static constexpr HOST_DEVICE_INLINE bool is_odd(const Field& xs) { return xs.limbs_storage.limbs[0] & 1; }

  static constexpr HOST_DEVICE_INLINE bool is_even(const Field& xs) { return ~xs.limbs_storage.limbs[0] & 1; }

  static constexpr HOST_DEVICE_INLINE Field inverse(const Field& xs)
  {
    if (xs == zero()) return zero();
    constexpr Field one = Field{CONFIG::one};
    constexpr ff_storage modulus = CONFIG::modulus;
    Field u = xs;
    Field v = Field{modulus};
    Field b = one;
    Field c = {};
    while (!(u == one) && !(v == one)) {
      while (is_even(u)) {
        u = div2(u);
        if (is_odd(b)) add_limbs<TLC, false>(b.limbs_storage, modulus, b.limbs_storage);
        b = div2(b);
      }
      while (is_even(v)) {
        v = div2(v);
        if (is_odd(c)) add_limbs<TLC, false>(c.limbs_storage, modulus, c.limbs_storage);
        c = div2(c);
      }
      if (lt(v, u)) {
        u = u - v;
        b = b - c;
      } else {
        v = v - u;
        c = c - b;
      }
    }
    return (u == one) ? b : c;
  }

  static constexpr HOST_DEVICE_INLINE Field pow(Field base, int exp)
  {
    Field res = one();
    while (exp > 0) {
      if (exp & 1) res = res * base;
      base = base * base;
      exp >>= 1;
    }
    return res;
  }
};

template <class CONFIG>
struct std::hash<Field<CONFIG>> {
  std::size_t operator()(const Field<CONFIG>& key) const
  {
    std::size_t hash = 0;
    // boost hashing, see
    // https://stackoverflow.com/questions/35985960/c-why-is-boosthash-combine-the-best-way-to-combine-hash-values/35991300#35991300
    for (int i = 0; i < CONFIG::limbs_count; i++)
      hash ^= std::hash<uint32_t>()(key.limbs_storage.limbs[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
  }
};

template <class CONFIG>
struct SharedMemory<Field<CONFIG>> {
  __device__ Field<CONFIG>* getPointer()
  {
    extern __shared__ Field<CONFIG> s_scalar_[];
    return s_scalar_;
  }
};
