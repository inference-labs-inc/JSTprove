#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace ptx {

  __device__ __forceinline__ uint32_t add(const uint32_t x, const uint32_t y)
  {
    uint32_t result;
    asm("add.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
    return result;
  }

  __device__ __forceinline__ uint32_t add_cc(const uint32_t x, const uint32_t y)
  {
    uint32_t result;
    asm volatile("add.cc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
    return result;
  }

  __device__ __forceinline__ uint32_t addc(const uint32_t x, const uint32_t y)
  {
    uint32_t result;
    asm volatile("addc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
    return result;
  }

  __device__ __forceinline__ uint32_t addc_cc(const uint32_t x, const uint32_t y)
  {
    uint32_t result;
    asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
    return result;
  }

  __device__ __forceinline__ uint32_t sub(const uint32_t x, const uint32_t y)
  {
    uint32_t result;
    asm("sub.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
    return result;
  }

  __device__ __forceinline__ uint32_t sub_cc(const uint32_t x, const uint32_t y)
  {
    uint32_t result;
    asm volatile("sub.cc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
    return result;
  }

  __device__ __forceinline__ uint32_t subc(const uint32_t x, const uint32_t y)
  {
    uint32_t result;
    asm volatile("subc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
    return result;
  }

  __device__ __forceinline__ uint32_t subc_cc(const uint32_t x, const uint32_t y)
  {
    uint32_t result;
    asm volatile("subc.cc.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
    return result;
  }

  __device__ __forceinline__ uint32_t mul_lo(const uint32_t x, const uint32_t y)
  {
    uint32_t result;
    asm("mul.lo.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
    return result;
  }

  __device__ __forceinline__ uint32_t mul_hi(const uint32_t x, const uint32_t y)
  {
    uint32_t result;
    asm("mul.hi.u32 %0, %1, %2;" : "=r"(result) : "r"(x), "r"(y));
    return result;
  }

  __device__ __forceinline__ uint32_t mad_lo(const uint32_t x, const uint32_t y, const uint32_t z)
  {
    uint32_t result;
    asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
    return result;
  }

  __device__ __forceinline__ uint32_t mad_hi(const uint32_t x, const uint32_t y, const uint32_t z)
  {
    uint32_t result;
    asm("mad.hi.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
    return result;
  }

  __device__ __forceinline__ uint32_t mad_lo_cc(const uint32_t x, const uint32_t y, const uint32_t z)
  {
    uint32_t result;
    asm volatile("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
    return result;
  }

  __device__ __forceinline__ uint32_t mad_hi_cc(const uint32_t x, const uint32_t y, const uint32_t z)
  {
    uint32_t result;
    asm volatile("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
    return result;
  }

  __device__ __forceinline__ uint32_t madc_lo(const uint32_t x, const uint32_t y, const uint32_t z)
  {
    uint32_t result;
    asm volatile("madc.lo.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
    return result;
  }

  __device__ __forceinline__ uint32_t madc_hi(const uint32_t x, const uint32_t y, const uint32_t z)
  {
    uint32_t result;
    asm volatile("madc.hi.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
    return result;
  }

  __device__ __forceinline__ uint32_t madc_lo_cc(const uint32_t x, const uint32_t y, const uint32_t z)
  {
    uint32_t result;
    asm volatile("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
    return result;
  }

  __device__ __forceinline__ uint32_t madc_hi_cc(const uint32_t x, const uint32_t y, const uint32_t z)
  {
    uint32_t result;
    asm volatile("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
    return result;
  }

  __device__ __forceinline__ uint64_t mov_b64(uint32_t lo, uint32_t hi)
  {
    uint64_t result;
    asm("mov.b64 %0, {%1,%2};" : "=l"(result) : "r"(lo), "r"(hi));
    return result;
  }

  // Gives u64 overloads a dedicated namespace.
  // Callers should know exactly what they're calling (no implicit conversions).
  namespace u64 {

    __device__ __forceinline__ uint64_t add(const uint64_t x, const uint64_t y)
    {
      uint64_t result;
      asm("add.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
      return result;
    }

    __device__ __forceinline__ uint64_t add_cc(const uint64_t x, const uint64_t y)
    {
      uint64_t result;
      asm volatile("add.cc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
      return result;
    }

    __device__ __forceinline__ uint64_t addc(const uint64_t x, const uint64_t y)
    {
      uint64_t result;
      asm volatile("addc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
      return result;
    }

    __device__ __forceinline__ uint64_t addc_cc(const uint64_t x, const uint64_t y)
    {
      uint64_t result;
      asm volatile("addc.cc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
      return result;
    }

    __device__ __forceinline__ uint64_t sub(const uint64_t x, const uint64_t y)
    {
      uint64_t result;
      asm("sub.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
      return result;
    }

    __device__ __forceinline__ uint64_t sub_cc(const uint64_t x, const uint64_t y)
    {
      uint64_t result;
      asm volatile("sub.cc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
      return result;
    }

    __device__ __forceinline__ uint64_t subc(const uint64_t x, const uint64_t y)
    {
      uint64_t result;
      asm volatile("subc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
      return result;
    }

    __device__ __forceinline__ uint64_t subc_cc(const uint64_t x, const uint64_t y)
    {
      uint64_t result;
      asm volatile("subc.cc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
      return result;
    }

    __device__ __forceinline__ uint64_t mul_lo(const uint64_t x, const uint64_t y)
    {
      uint64_t result;
      asm("mul.lo.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
      return result;
    }

    __device__ __forceinline__ uint64_t mul_hi(const uint64_t x, const uint64_t y)
    {
      uint64_t result;
      asm("mul.hi.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
      return result;
    }

    __device__ __forceinline__ uint64_t mad_lo(const uint64_t x, const uint64_t y, const uint64_t z)
    {
      uint64_t result;
      asm("mad.lo.u64 %0, %1, %2, %3;" : "=l"(result) : "l"(x), "l"(y), "l"(z));
      return result;
    }

    __device__ __forceinline__ uint64_t mad_hi(const uint64_t x, const uint64_t y, const uint64_t z)
    {
      uint64_t result;
      asm("mad.hi.u64 %0, %1, %2, %3;" : "=l"(result) : "l"(x), "l"(y), "l"(z));
      return result;
    }

    __device__ __forceinline__ uint64_t mad_lo_cc(const uint64_t x, const uint64_t y, const uint64_t z)
    {
      uint64_t result;
      asm volatile("mad.lo.cc.u64 %0, %1, %2, %3;" : "=l"(result) : "l"(x), "l"(y), "l"(z));
      return result;
    }

    __device__ __forceinline__ uint64_t mad_hi_cc(const uint64_t x, const uint64_t y, const uint64_t z)
    {
      uint64_t result;
      asm volatile("mad.hi.cc.u64 %0, %1, %2, %3;" : "=l"(result) : "l"(x), "l"(y), "l"(z));
      return result;
    }

    __device__ __forceinline__ uint64_t madc_lo(const uint64_t x, const uint64_t y, const uint64_t z)
    {
      uint64_t result;
      asm volatile("madc.lo.u64 %0, %1, %2, %3;" : "=l"(result) : "l"(x), "l"(y), "l"(z));
      return result;
    }

    __device__ __forceinline__ uint64_t madc_hi(const uint64_t x, const uint64_t y, const uint64_t z)
    {
      uint64_t result;
      asm volatile("madc.hi.u64 %0, %1, %2, %3;" : "=l"(result) : "l"(x), "l"(y), "l"(z));
      return result;
    }

    __device__ __forceinline__ uint64_t madc_lo_cc(const uint64_t x, const uint64_t y, const uint64_t z)
    {
      uint64_t result;
      asm volatile("madc.lo.cc.u64 %0, %1, %2, %3;" : "=l"(result) : "l"(x), "l"(y), "l"(z));
      return result;
    }

    __device__ __forceinline__ uint64_t madc_hi_cc(const uint64_t x, const uint64_t y, const uint64_t z)
    {
      uint64_t result;
      asm volatile("madc.hi.cc.u64 %0, %1, %2, %3;" : "=l"(result) : "l"(x), "l"(y), "l"(z));
      return result;
    }

  } // namespace u64

  __device__ __forceinline__ void bar_arrive(const unsigned name, const unsigned count)
  {
    asm volatile("bar.arrive %0, %1;" : : "r"(name), "r"(count) : "memory");
  }

  __device__ __forceinline__ void bar_sync(const unsigned name, const unsigned count)
  {
    asm volatile("bar.sync %0, %1;" : : "r"(name), "r"(count) : "memory");
  }

} // namespace ptx
