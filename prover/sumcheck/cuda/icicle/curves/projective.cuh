#pragma once

#include "affine.cuh"
#include "gpu-utils/sharedmem.cuh"

template <typename FF, class SCALAR_FF, const FF& B_VALUE, const FF& GENERATOR_X, const FF& GENERATOR_Y>
class Projective
{
  friend Affine<FF>;

public:
  typedef Affine<FF> Aff;
  typedef SCALAR_FF Scalar;

  static constexpr unsigned SCALAR_FF_NBITS = SCALAR_FF::NBITS;
  static constexpr unsigned FF_NBITS = FF::NBITS;

  FF x;
  FF y;
  FF z;

  static HOST_DEVICE_INLINE Projective zero() { return {FF::zero(), FF::one(), FF::zero()}; }

  static HOST_DEVICE_INLINE Affine<FF> to_affine(const Projective& point)
  {
    FF denom = FF::inverse(point.z);
    return {point.x * denom, point.y * denom};
  }

  static HOST_DEVICE_INLINE Projective from_affine(const Affine<FF>& point)
  {
    return point == Affine<FF>::zero() ? zero() : Projective{point.x, point.y, FF::one()};
  }

  static HOST_DEVICE_INLINE Projective to_montgomery(const Projective& point)
  {
    return {FF::to_montgomery(point.x), FF::to_montgomery(point.y), FF::to_montgomery(point.z)};
  }

  static HOST_DEVICE_INLINE Projective from_montgomery(const Projective& point)
  {
    return {FF::from_montgomery(point.x), FF::from_montgomery(point.y), FF::from_montgomery(point.z)};
  }

  static HOST_DEVICE_INLINE Projective generator() { return {GENERATOR_X, GENERATOR_Y, FF::one()}; }

  static HOST_DEVICE_INLINE Projective neg(const Projective& point) { return {point.x, FF::neg(point.y), point.z}; }

  static HOST_DEVICE_INLINE Projective dbl(const Projective& point)
  {
    const FF X = point.x;
    const FF Y = point.y;
    const FF Z = point.z;

    // TODO: Change to efficient dbl once implemented for field.cuh
    FF t0 = FF::sqr(Y);                                                     // 1. t0 ← Y · Y
    FF Z3 = t0 + t0;                                                        // 2. Z3 ← t0 + t0
    Z3 = Z3 + Z3;                                                           // 3. Z3 ← Z3 + Z3
    Z3 = Z3 + Z3;                                                           // 4. Z3 ← Z3 + Z3
    FF t1 = Y * Z;                                                          // 5. t1 ← Y · Z
    FF t2 = FF::sqr(Z);                                                     // 6. t2 ← Z · Z
    t2 = FF::template mul_unsigned<3>(FF::template mul_const<B_VALUE>(t2)); // 7. t2 ← b3 · t2
    FF X3 = t2 * Z3;                                                        // 8. X3 ← t2 · Z3
    FF Y3 = t0 + t2;                                                        // 9. Y3 ← t0 + t2
    Z3 = t1 * Z3;                                                           // 10. Z3 ← t1 · Z3
    t1 = t2 + t2;                                                           // 11. t1 ← t2 + t2
    t2 = t1 + t2;                                                           // 12. t2 ← t1 + t2
    t0 = t0 - t2;                                                           // 13. t0 ← t0 − t2
    Y3 = t0 * Y3;                                                           // 14. Y3 ← t0 · Y3
    Y3 = X3 + Y3;                                                           // 15. Y3 ← X3 + Y3
    t1 = X * Y;                                                             // 16. t1 ← X · Y
    X3 = t0 * t1;                                                           // 17. X3 ← t0 · t1
    X3 = X3 + X3;                                                           // 18. X3 ← X3 + X3
    return {X3, Y3, Z3};
  }

  friend HOST_DEVICE_INLINE Projective operator+(Projective p1, const Projective& p2)
  {
    const FF X1 = p1.x;                                                                //                   < 2
    const FF Y1 = p1.y;                                                                //                   < 2
    const FF Z1 = p1.z;                                                                //                   < 2
    const FF X2 = p2.x;                                                                //                   < 2
    const FF Y2 = p2.y;                                                                //                   < 2
    const FF Z2 = p2.z;                                                                //                   < 2
    const FF t00 = X1 * X2;                                                            // t00 ← X1 · X2     < 2
    const FF t01 = Y1 * Y2;                                                            // t01 ← Y1 · Y2     < 2
    const FF t02 = Z1 * Z2;                                                            // t02 ← Z1 · Z2     < 2
    const FF t03 = X1 + Y1;                                                            // t03 ← X1 + Y1     < 4
    const FF t04 = X2 + Y2;                                                            // t04 ← X2 + Y2     < 4
    const FF t05 = t03 * t04;                                                          // t03 ← t03 · t04   < 3
    const FF t06 = t00 + t01;                                                          // t06 ← t00 + t01   < 4
    const FF t07 = t05 - t06;                                                          // t05 ← t05 − t06   < 2
    const FF t08 = Y1 + Z1;                                                            // t08 ← Y1 + Z1     < 4
    const FF t09 = Y2 + Z2;                                                            // t09 ← Y2 + Z2     < 4
    const FF t10 = t08 * t09;                                                          // t10 ← t08 · t09   < 3
    const FF t11 = t01 + t02;                                                          // t11 ← t01 + t02   < 4
    const FF t12 = t10 - t11;                                                          // t12 ← t10 − t11   < 2
    const FF t13 = X1 + Z1;                                                            // t13 ← X1 + Z1     < 4
    const FF t14 = X2 + Z2;                                                            // t14 ← X2 + Z2     < 4
    const FF t15 = t13 * t14;                                                          // t15 ← t13 · t14   < 3
    const FF t16 = t00 + t02;                                                          // t16 ← t00 + t02   < 4
    const FF t17 = t15 - t16;                                                          // t17 ← t15 − t16   < 2
    const FF t18 = t00 + t00;                                                          // t18 ← t00 + t00   < 2
    const FF t19 = t18 + t00;                                                          // t19 ← t18 + t00   < 2
    const FF t20 = FF::template mul_unsigned<3>(FF::template mul_const<B_VALUE>(t02)); // t20 ← b3 · t02    < 2
    const FF t21 = t01 + t20;                                                          // t21 ← t01 + t20   < 2
    const FF t22 = t01 - t20;                                                          // t22 ← t01 − t20   < 2
    const FF t23 = FF::template mul_unsigned<3>(FF::template mul_const<B_VALUE>(t17)); // t23 ← b3 · t17    < 2
    const auto t24 = FF::mul_wide(t12, t23);                                           // t24 ← t12 · t23   < 2
    const auto t25 = FF::mul_wide(t07, t22);                                           // t25 ← t07 · t22   < 2
    const FF X3 = FF::reduce(t25 - t24);                                               // X3 ← t25 − t24    < 2
    const auto t27 = FF::mul_wide(t23, t19);                                           // t27 ← t23 · t19   < 2
    const auto t28 = FF::mul_wide(t22, t21);                                           // t28 ← t22 · t21   < 2
    const FF Y3 = FF::reduce(t28 + t27);                                               // Y3 ← t28 + t27    < 2
    const auto t30 = FF::mul_wide(t19, t07);                                           // t30 ← t19 · t07   < 2
    const auto t31 = FF::mul_wide(t21, t12);                                           // t31 ← t21 · t12   < 2
    const FF Z3 = FF::reduce(t31 + t30);                                               // Z3 ← t31 + t30    < 2
    return {X3, Y3, Z3};
  }

  friend HOST_DEVICE_INLINE Projective operator-(Projective p1, const Projective& p2) { return p1 + neg(p2); }

  friend HOST_DEVICE_INLINE Projective operator+(Projective p1, const Affine<FF>& p2)
  {
    const FF X1 = p1.x;                                                                //                   < 2
    const FF Y1 = p1.y;                                                                //                   < 2
    const FF Z1 = p1.z;                                                                //                   < 2
    const FF X2 = p2.x;                                                                //                   < 2
    const FF Y2 = p2.y;                                                                //                   < 2
    const FF t00 = X1 * X2;                                                            // t00 ← X1 · X2     < 2
    const FF t01 = Y1 * Y2;                                                            // t01 ← Y1 · Y2     < 2
    const FF t02 = Z1;                                                                 // t02 ← Z1          < 2
    const FF t03 = X1 + Y1;                                                            // t03 ← X1 + Y1     < 4
    const FF t04 = X2 + Y2;                                                            // t04 ← X2 + Y2     < 4
    const FF t05 = t03 * t04;                                                          // t03 ← t03 · t04   < 3
    const FF t06 = t00 + t01;                                                          // t06 ← t00 + t01   < 4
    const FF t07 = t05 - t06;                                                          // t05 ← t05 − t06   < 2
    const FF t08 = Y1 + Z1;                                                            // t08 ← Y1 + Z1     < 4
    const FF t09 = Y2 + FF::one();                                                     // t09 ← Y2 + 1      < 4
    const FF t10 = t08 * t09;                                                          // t10 ← t08 · t09   < 3
    const FF t11 = t01 + t02;                                                          // t11 ← t01 + t02   < 4
    const FF t12 = t10 - t11;                                                          // t12 ← t10 − t11   < 2
    const FF t13 = X1 + Z1;                                                            // t13 ← X1 + Z1     < 4
    const FF t14 = X2 + FF::one();                                                     // t14 ← X2 + 1      < 4
    const FF t15 = t13 * t14;                                                          // t15 ← t13 · t14   < 3
    const FF t16 = t00 + t02;                                                          // t16 ← t00 + t02   < 4
    const FF t17 = t15 - t16;                                                          // t17 ← t15 − t16   < 2
    const FF t18 = t00 + t00;                                                          // t18 ← t00 + t00   < 2
    const FF t19 = t18 + t00;                                                          // t19 ← t18 + t00   < 2
    const FF t20 = FF::template mul_unsigned<3>(FF::template mul_const<B_VALUE>(t02)); // t20 ← b3 · t02    < 2
    const FF t21 = t01 + t20;                                                          // t21 ← t01 + t20   < 2
    const FF t22 = t01 - t20;                                                          // t22 ← t01 − t20   < 2
    const FF t23 = FF::template mul_unsigned<3>(FF::template mul_const<B_VALUE>(t17)); // t23 ← b3 · t17    < 2
    const auto t24 = FF::mul_wide(t12, t23);                                           // t24 ← t12 · t23   < 2
    const auto t25 = FF::mul_wide(t07, t22);                                           // t25 ← t07 · t22   < 2
    const FF X3 = FF::reduce(t25 - t24);                                               // X3 ← t25 − t24    < 2
    const auto t27 = FF::mul_wide(t23, t19);                                           // t27 ← t23 · t19   < 2
    const auto t28 = FF::mul_wide(t22, t21);                                           // t28 ← t22 · t21   < 2
    const FF Y3 = FF::reduce(t28 + t27);                                               // Y3 ← t28 + t27    < 2
    const auto t30 = FF::mul_wide(t19, t07);                                           // t30 ← t19 · t07   < 2
    const auto t31 = FF::mul_wide(t21, t12);                                           // t31 ← t21 · t12   < 2
    const FF Z3 = FF::reduce(t31 + t30);                                               // Z3 ← t31 + t30    < 2
    return {X3, Y3, Z3};
  }

  friend HOST_DEVICE_INLINE Projective operator-(Projective p1, const Affine<FF>& p2)
  {
    return p1 + Affine<FF>::neg(p2);
  }

  friend HOST_DEVICE_INLINE Projective operator*(SCALAR_FF scalar, const Projective& point)
  {
    Projective res = zero();
#ifdef __CUDA_ARCH__
    UNROLL
#endif
    for (int i = 0; i < SCALAR_FF::NBITS; i++) {
      if (i > 0) { res = dbl(res); }
      if (scalar.get_scalar_digit(SCALAR_FF::NBITS - i - 1, 1)) { res = res + point; }
    }
    return res;
  }

  friend HOST_DEVICE_INLINE Projective operator*(const Projective& point, SCALAR_FF scalar) { return scalar * point; }

  friend HOST_DEVICE_INLINE bool operator==(const Projective& p1, const Projective& p2)
  {
    return (p1.x * p2.z == p2.x * p1.z) && (p1.y * p2.z == p2.y * p1.z);
  }

  friend HOST_DEVICE_INLINE bool operator!=(const Projective& p1, const Projective& p2) { return !(p1 == p2); }

  friend HOST_INLINE std::ostream& operator<<(std::ostream& os, const Projective& point)
  {
    os << "Point { x: " << point.x << "; y: " << point.y << "; z: " << point.z << " }";
    return os;
  }

  static HOST_DEVICE_INLINE bool is_zero(const Projective& point)
  {
    return point.x == FF::zero() && point.y != FF::zero() && point.z == FF::zero();
  }

  static HOST_DEVICE_INLINE bool is_on_curve(const Projective& point)
  {
    if (is_zero(point)) return true;
    bool eq_holds =
      (FF::template mul_const<B_VALUE>(FF::sqr(point.z) * point.z) + FF::sqr(point.x) * point.x ==
       point.z * FF::sqr(point.y));
    return point.z != FF::zero() && eq_holds;
  }

  static HOST_INLINE Projective rand_host()
  {
    SCALAR_FF rand_scalar = SCALAR_FF::rand_host();
    return rand_scalar * generator();
  }

  static void rand_host_many(Projective* out, int size)
  {
    for (int i = 0; i < size; i++)
      out[i] = (i % size < 100) ? rand_host() : out[i - 100];
  }

  static void rand_host_many_affine(Affine<FF>* out, int size)
  {
    for (int i = 0; i < size; i++)
      out[i] = (i % size < 100) ? to_affine(rand_host()) : out[i - 100];
  }
};

template <typename FF, class SCALAR_FF, const FF& B_VALUE, const FF& GENERATOR_X, const FF& GENERATOR_Y>
struct SharedMemory<Projective<FF, SCALAR_FF, B_VALUE, GENERATOR_X, GENERATOR_Y>> {
  __device__ Projective<FF, SCALAR_FF, B_VALUE, GENERATOR_X, GENERATOR_Y>* getPointer()
  {
    extern __shared__ Projective<FF, SCALAR_FF, B_VALUE, GENERATOR_X, GENERATOR_Y> s_projective_[];
    return s_projective_;
  }
};
