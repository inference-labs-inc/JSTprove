#pragma once
#ifndef HOST_MATH_H
#define HOST_MATH_H

#include <cstdint>
#include <cuda_runtime.h>
#include "gpu-utils/modifiers.cuh"
#include "storage.cuh"

namespace host_math {

  // return x + y with uint32_t operands
  static constexpr __host__ uint32_t add(const uint32_t x, const uint32_t y) { return x + y; }

  // return x + y + carry with uint32_t operands
  static constexpr __host__ uint32_t addc(const uint32_t x, const uint32_t y, const uint32_t carry)
  {
    return x + y + carry;
  }

  // return x + y and carry out with uint32_t operands
  static constexpr __host__ uint32_t add_cc(const uint32_t x, const uint32_t y, uint32_t& carry)
  {
    uint32_t result = x + y;
    carry = x > result;
    return result;
  }

  // return x + y + carry and carry out  with uint32_t operands
  static constexpr __host__ uint32_t addc_cc(const uint32_t x, const uint32_t y, uint32_t& carry)
  {
    const uint32_t result = x + y + carry;
    carry = carry && x >= result || !carry && x > result;
    return result;
  }

  // return x - y with uint32_t operands
  static constexpr __host__ uint32_t sub(const uint32_t x, const uint32_t y) { return x - y; }

  //    return x - y - borrow with uint32_t operands
  static constexpr __host__ uint32_t subc(const uint32_t x, const uint32_t y, const uint32_t borrow)
  {
    return x - y - borrow;
  }

  //    return x - y and borrow out with uint32_t operands
  static constexpr __host__ uint32_t sub_cc(const uint32_t x, const uint32_t y, uint32_t& borrow)
  {
    uint32_t result = x - y;
    borrow = x < result;
    return result;
  }

  //    return x - y - borrow and borrow out with uint32_t operands
  static constexpr __host__ uint32_t subc_cc(const uint32_t x, const uint32_t y, uint32_t& borrow)
  {
    const uint32_t result = x - y - borrow;
    borrow = borrow && x <= result || !borrow && x < result;
    return result;
  }

  // return x * y + z + carry and carry out with uint32_t operands
  static constexpr __host__ uint32_t madc_cc(const uint32_t x, const uint32_t y, const uint32_t z, uint32_t& carry)
  {
    uint64_t r = static_cast<uint64_t>(x) * y + z + carry;
    carry = (uint32_t)(r >> 32);
    uint32_t result = r & 0xffffffff;
    return result;
  }

  template <unsigned OPS_COUNT = UINT32_MAX, bool CARRY_IN = false, bool CARRY_OUT = false>
  struct carry_chain {
    unsigned index;

    constexpr HOST_INLINE carry_chain() : index(0) {}

    constexpr HOST_INLINE uint32_t add(const uint32_t x, const uint32_t y, uint32_t& carry)
    {
      index++;
      if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT)
        return host_math::add(x, y);
      else if (index == 1 && !CARRY_IN)
        return host_math::add_cc(x, y, carry);
      else if (index < OPS_COUNT || CARRY_OUT)
        return host_math::addc_cc(x, y, carry);
      else
        return host_math::addc(x, y, carry);
    }

    constexpr HOST_INLINE uint32_t sub(const uint32_t x, const uint32_t y, uint32_t& carry)
    {
      index++;
      if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT)
        return host_math::sub(x, y);
      else if (index == 1 && !CARRY_IN)
        return host_math::sub_cc(x, y, carry);
      else if (index < OPS_COUNT || CARRY_OUT)
        return host_math::subc_cc(x, y, carry);
      else
        return host_math::subc(x, y, carry);
    }
  };

  template <unsigned NLIMBS_A, unsigned NLIMBS_B = NLIMBS_A>
  static constexpr HOST_INLINE void
  multiply_raw(const storage<NLIMBS_A>& as, const storage<NLIMBS_B>& bs, storage<NLIMBS_A + NLIMBS_B>& rs)
  {
    const uint32_t* a = as.limbs;
    const uint32_t* b = bs.limbs;
    uint32_t* r = rs.limbs;
    for (unsigned i = 0; i < NLIMBS_B; i++) {
      uint32_t carry = 0;
      for (unsigned j = 0; j < NLIMBS_A; j++)
        r[j + i] = host_math::madc_cc(a[j], b[i], r[j + i], carry);
      r[NLIMBS_A + i] = carry;
    }
  }

  template <unsigned NLIMBS, bool SUBTRACT, bool CARRY_OUT>
  static constexpr HOST_INLINE uint32_t
  add_sub_limbs(const storage<NLIMBS>& xs, const storage<NLIMBS>& ys, storage<NLIMBS>& rs)
  {
    const uint32_t* x = xs.limbs;
    const uint32_t* y = ys.limbs;
    uint32_t* r = rs.limbs;
    uint32_t carry = 0;
    carry_chain<NLIMBS, false, CARRY_OUT> chain;
    for (unsigned i = 0; i < NLIMBS; i++)
      r[i] = SUBTRACT ? chain.sub(x[i], y[i], carry) : chain.add(x[i], y[i], carry);
    return CARRY_OUT ? carry : 0;
  }

  template <unsigned NLIMBS, unsigned BITS>
  static constexpr HOST_INLINE storage<NLIMBS> left_shift(const storage<NLIMBS>& xs)
  {
    if constexpr (BITS == 0)
      return xs;
    else {
      constexpr unsigned BITS32 = BITS % 32;
      constexpr unsigned LIMBS_GAP = BITS / 32;
      storage<NLIMBS> out{};
      if constexpr (LIMBS_GAP < NLIMBS) {
        out.limbs[LIMBS_GAP] = xs.limbs[0] << BITS32;
        for (unsigned i = 1; i < NLIMBS - LIMBS_GAP; i++)
          out.limbs[i + LIMBS_GAP] = (xs.limbs[i] << BITS32) + (xs.limbs[i - 1] >> (32 - BITS32));
      }
      return out;
    }
  }

  template <unsigned NLIMBS, unsigned BITS>
  static constexpr HOST_INLINE storage<NLIMBS> right_shift(const storage<NLIMBS>& xs)
  {
    if constexpr (BITS == 0)
      return xs;
    else {
      constexpr unsigned BITS32 = BITS % 32;
      constexpr unsigned LIMBS_GAP = BITS / 32;
      storage<NLIMBS> out{};
      if constexpr (LIMBS_GAP < NLIMBS - 1) {
        for (unsigned i = 0; i < NLIMBS - LIMBS_GAP - 1; i++)
          out.limbs[i] = (xs.limbs[i + LIMBS_GAP] >> BITS32) + (xs.limbs[i + LIMBS_GAP + 1] << (32 - BITS32));
      }
      if constexpr (LIMBS_GAP < NLIMBS) out.limbs[NLIMBS - LIMBS_GAP - 1] = (xs.limbs[NLIMBS - 1] >> BITS32);
      return out;
    }
  }

  template <unsigned NLIMBS_NUM, unsigned NLIMBS_DENOM, unsigned NLIMBS_Q = (NLIMBS_NUM - NLIMBS_DENOM)>
  static constexpr HOST_INLINE void integer_division(
    const storage<NLIMBS_NUM>& num, const storage<NLIMBS_DENOM>& denom, storage<NLIMBS_Q>& q, storage<NLIMBS_DENOM>& r)
  {
    storage<NLIMBS_DENOM> temp = {};
    for (int limb_idx = NLIMBS_NUM - 1; limb_idx >= 0; limb_idx--) {
      for (int bit_idx = 31; bit_idx >= 0; bit_idx--) {
        r = left_shift<NLIMBS_DENOM, 1>(r);
        r.limbs[0] |= ((num.limbs[limb_idx] >> bit_idx) & 1);
        uint32_t c = add_sub_limbs<NLIMBS_DENOM, true, true>(r, denom, temp);
        if (limb_idx < NLIMBS_Q & !c) {
          r = temp;
          q.limbs[limb_idx] |= 1 << bit_idx;
        }
      }
    }
  }
} // namespace host_math

#endif
