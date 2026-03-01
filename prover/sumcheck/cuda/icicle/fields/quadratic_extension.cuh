#pragma once

#include "field.cuh"
#include "gpu-utils/modifiers.cuh"
#include "gpu-utils/sharedmem.cuh"

template <typename CONFIG, class T>
class ExtensionField
{
private:
  friend T;

  typedef typename T::Wide FWide;

  struct ExtensionWide {
    FWide real;
    FWide imaginary;

    friend HOST_DEVICE_INLINE ExtensionWide operator+(ExtensionWide xs, const ExtensionWide& ys)
    {
      return ExtensionWide{xs.real + ys.real, xs.imaginary + ys.imaginary};
    }

    friend HOST_DEVICE_INLINE ExtensionWide operator-(ExtensionWide xs, const ExtensionWide& ys)
    {
      return ExtensionWide{xs.real - ys.real, xs.imaginary - ys.imaginary};
    }
  };

public:
  typedef T FF;
  static constexpr unsigned TLC = 2 * CONFIG::limbs_count;

  FF real;
  FF imaginary;

  static constexpr HOST_DEVICE_INLINE ExtensionField zero() { return ExtensionField{FF::zero(), FF::zero()}; }

  static constexpr HOST_DEVICE_INLINE ExtensionField one() { return ExtensionField{FF::one(), FF::zero()}; }

  static constexpr HOST_DEVICE_INLINE ExtensionField to_montgomery(const ExtensionField& xs)
  {
    return ExtensionField{xs.real * FF{CONFIG::montgomery_r}, xs.imaginary * FF{CONFIG::montgomery_r}};
  }

  static constexpr HOST_DEVICE_INLINE ExtensionField from_montgomery(const ExtensionField& xs)
  {
    return ExtensionField{xs.real * FF{CONFIG::montgomery_r_inv}, xs.imaginary * FF{CONFIG::montgomery_r_inv}};
  }

  static HOST_INLINE ExtensionField rand_host() { return ExtensionField{FF::rand_host(), FF::rand_host()}; }

  static void rand_host_many(ExtensionField* out, int size)
  {
    for (int i = 0; i < size; i++)
      out[i] = rand_host();
  }

  template <unsigned REDUCTION_SIZE = 1>
  static constexpr HOST_DEVICE_INLINE ExtensionField sub_modulus(const ExtensionField& xs)
  {
    return ExtensionField{FF::sub_modulus<REDUCTION_SIZE>(&xs.real), FF::sub_modulus<REDUCTION_SIZE>(&xs.imaginary)};
  }

  friend std::ostream& operator<<(std::ostream& os, const ExtensionField& xs)
  {
    os << "{ Real: " << xs.real << " }; { Imaginary: " << xs.imaginary << " }";
    return os;
  }

  friend HOST_DEVICE_INLINE ExtensionField operator+(ExtensionField xs, const ExtensionField& ys)
  {
    return ExtensionField{xs.real + ys.real, xs.imaginary + ys.imaginary};
  }

  friend HOST_DEVICE_INLINE ExtensionField operator-(ExtensionField xs, const ExtensionField& ys)
  {
    return ExtensionField{xs.real - ys.real, xs.imaginary - ys.imaginary};
  }

  friend HOST_DEVICE_INLINE ExtensionField operator+(FF xs, const ExtensionField& ys)
  {
    return ExtensionField{xs + ys.real, ys.imaginary};
  }

  friend HOST_DEVICE_INLINE ExtensionField operator-(FF xs, const ExtensionField& ys)
  {
    return ExtensionField{xs - ys.real, FF::neg(ys.imaginary)};
  }

  friend HOST_DEVICE_INLINE ExtensionField operator+(ExtensionField xs, const FF& ys)
  {
    return ExtensionField{xs.real + ys, xs.imaginary};
  }

  friend HOST_DEVICE_INLINE ExtensionField operator-(ExtensionField xs, const FF& ys)
  {
    return ExtensionField{xs.real - ys, xs.imaginary};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE ExtensionWide mul_wide(const ExtensionField& xs, const ExtensionField& ys)
  {
    FWide real_prod = FF::mul_wide(xs.real, ys.real);
    FWide imaginary_prod = FF::mul_wide(xs.imaginary, ys.imaginary);
    FWide prod_of_sums = FF::mul_wide(xs.real + xs.imaginary, ys.real + ys.imaginary);
    FWide nonresidue_times_im = FF::template mul_unsigned<CONFIG::nonresidue>(imaginary_prod);
    nonresidue_times_im = CONFIG::nonresidue_is_negative ? FWide::neg(nonresidue_times_im) : nonresidue_times_im;
    return ExtensionWide{real_prod + nonresidue_times_im, prod_of_sums - real_prod - imaginary_prod};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE ExtensionWide mul_wide(const ExtensionField& xs, const FF& ys)
  {
    return ExtensionWide{FF::mul_wide(xs.real, ys), FF::mul_wide(xs.imaginary, ys)};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE ExtensionWide mul_wide(const FF& xs, const ExtensionField& ys)
  {
    return mul_wide(ys, xs);
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE ExtensionField reduce(const ExtensionWide& xs)
  {
    return ExtensionField{
      FF::template reduce<MODULUS_MULTIPLE>(xs.real), FF::template reduce<MODULUS_MULTIPLE>(xs.imaginary)};
  }

  template <class T1, class T2>
  friend HOST_DEVICE_INLINE ExtensionField operator*(const T1& xs, const T2& ys)
  {
    ExtensionWide xy = mul_wide(xs, ys);
    return reduce(xy);
  }

  friend HOST_DEVICE_INLINE bool operator==(const ExtensionField& xs, const ExtensionField& ys)
  {
    return (xs.real == ys.real) && (xs.imaginary == ys.imaginary);
  }

  friend HOST_DEVICE_INLINE bool operator!=(const ExtensionField& xs, const ExtensionField& ys) { return !(xs == ys); }

  template <const ExtensionField& multiplier>
  static HOST_DEVICE_INLINE ExtensionField mul_const(const ExtensionField& xs)
  {
    static constexpr FF mul_real = multiplier.real;
    static constexpr FF mul_imaginary = multiplier.imaginary;
    const FF xs_real = xs.real;
    const FF xs_imaginary = xs.imaginary;
    FF real_prod = FF::template mul_const<mul_real>(xs_real);
    FF imaginary_prod = FF::template mul_const<mul_imaginary>(xs_imaginary);
    FF re_im = FF::template mul_const<mul_real>(xs_imaginary);
    FF im_re = FF::template mul_const<mul_imaginary>(xs_real);
    FF nonresidue_times_im = FF::template mul_unsigned<CONFIG::nonresidue>(imaginary_prod);
    nonresidue_times_im = CONFIG::nonresidue_is_negative ? FF::neg(nonresidue_times_im) : nonresidue_times_im;
    return ExtensionField{real_prod + nonresidue_times_im, re_im + im_re};
  }

  template <uint32_t multiplier, unsigned REDUCTION_SIZE = 1>
  static constexpr HOST_DEVICE_INLINE ExtensionField mul_unsigned(const ExtensionField& xs)
  {
    return {FF::template mul_unsigned<multiplier>(xs.real), FF::template mul_unsigned<multiplier>(xs.imaginary)};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE ExtensionWide sqr_wide(const ExtensionField& xs)
  {
    // TODO: change to a more efficient squaring
    return mul_wide<MODULUS_MULTIPLE>(xs, xs);
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE ExtensionField sqr(const ExtensionField& xs)
  {
    // TODO: change to a more efficient squaring
    return xs * xs;
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE ExtensionField neg(const ExtensionField& xs)
  {
    return ExtensionField{FF::neg(xs.real), FF::neg(xs.imaginary)};
  }

  // inverse of zero is set to be zero which is what we want most of the time
  static constexpr HOST_DEVICE_INLINE ExtensionField inverse(const ExtensionField& xs)
  {
    ExtensionField xs_conjugate = {xs.real, FF::neg(xs.imaginary)};
    FF nonresidue_times_im = FF::template mul_unsigned<CONFIG::nonresidue>(FF::sqr(xs.imaginary));
    nonresidue_times_im = CONFIG::nonresidue_is_negative ? FF::neg(nonresidue_times_im) : nonresidue_times_im;
    // TODO: wide here
    FF xs_norm_squared = FF::sqr(xs.real) - nonresidue_times_im;
    return xs_conjugate * ExtensionField{FF::inverse(xs_norm_squared), FF::zero()};
  }
};

template <typename CONFIG, class T>
struct SharedMemory<ExtensionField<CONFIG, T>> {
  __device__ ExtensionField<CONFIG, T>* getPointer()
  {
    extern __shared__ ExtensionField<CONFIG, T> s_ext2_scalar_[];
    return s_ext2_scalar_;
  }
};
