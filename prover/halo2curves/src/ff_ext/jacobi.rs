use core::cmp::PartialEq;
use std::ops::{Add, Mul, Neg, Shr, Sub};

/// Big signed (64 * L)-bit integer type, whose variables store
/// numbers in the two's complement code as arrays of 64-bit chunks.
/// The ordering of the chunks in these arrays is little-endian.
/// The arithmetic operations for this type are wrapping ones
#[derive(Clone)]
pub struct LInt<const L: usize>([u64; L]);

impl<const L: usize> LInt<L> {
    /// Representation of -1
    pub const MINUS_ONE: Self = Self([u64::MAX; L]);

    /// Representation of 0
    pub const ZERO: Self = Self([0; L]);

    /// Representation of 1
    pub const ONE: Self = {
        let mut data = [0; L];
        data[0] = 1;
        Self(data)
    };

    /// Returns the number, which is stored as the specified
    /// sequence padded with zeros to length L. If the input
    /// sequence is longer than L, the method panics
    pub fn new(data: &[u64]) -> Self {
        let mut number = Self::ZERO;
        number.0[..data.len()].copy_from_slice(data);
        number
    }

    /// Returns "true" iff the current number is negative
    #[inline]
    pub fn is_negative(&self) -> bool {
        self.0[L - 1] > (u64::MAX >> 1)
    }

    /// Returns a tuple representing the sum of the first two arguments and the bit
    /// described by the third argument. The first element of the tuple is this sum
    /// modulo 2^64, the second one indicates whether the sum is no less than 2^64
    #[inline]
    fn sum(first: u64, second: u64, carry: bool) -> (u64, bool) {
        // The implementation is inspired with the "carrying_add" function from this source:
        // https://github.com/rust-lang/rust/blob/master/library/core/src/num/uint_macros.rs
        let (second, carry) = second.overflowing_add(carry as u64);
        let (first, high) = first.overflowing_add(second);
        (first, carry || high)
    }

    /// Returns "(low, high)", where "high * 2^64 + low = first * second + carry + summand"
    #[inline]
    fn prodsum(first: u64, second: u64, summand: u64, carry: u64) -> (u64, u64) {
        let all = (first as u128) * (second as u128) + (carry as u128) + (summand as u128);
        (all as u64, (all >> u64::BITS) as u64)
    }
}

impl<const L: usize> PartialEq for LInt<L> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<const L: usize> Shr<u32> for &LInt<L> {
    type Output = LInt<L>;
    /// Returns the result of applying the arithmetic right shift to the current number.
    /// The specified bit quantity the number is shifted by must lie in {1, 2, ..., 63}.
    /// For the quantities outside of the range, the behavior of the method is undefined
    fn shr(self, bits: u32) -> Self::Output {
        debug_assert!(
            (bits > 0) && (bits < 64),
            "Cannot shift by 0 or more than 63 bits!"
        );
        let (mut data, right) = ([0; L], u64::BITS - bits);

        for (i, d) in data.iter_mut().enumerate().take(L - 1) {
            *d = (self.0[i] >> bits) | (self.0[i + 1] << right);
        }
        data[L - 1] = self.0[L - 1] >> bits;
        if self.is_negative() {
            data[L - 1] |= u64::MAX << right;
        }
        LInt::<L>(data)
    }
}

impl<const L: usize> Shr<u32> for LInt<L> {
    type Output = LInt<L>;
    fn shr(self, bits: u32) -> Self::Output {
        &self >> bits
    }
}

impl<const L: usize> Add for &LInt<L> {
    type Output = LInt<L>;
    fn add(self, other: Self) -> Self::Output {
        let (mut data, mut carry) = ([0; L], false);
        for (i, d) in data.iter_mut().enumerate().take(L) {
            (*d, carry) = Self::Output::sum(self.0[i], other.0[i], carry);
        }
        LInt::<L>(data)
    }
}

impl<const L: usize> Add<&LInt<L>> for LInt<L> {
    type Output = LInt<L>;
    fn add(self, other: &Self) -> Self::Output {
        &self + other
    }
}

impl<const L: usize> Add for LInt<L> {
    type Output = LInt<L>;
    fn add(self, other: Self) -> Self::Output {
        &self + &other
    }
}

impl<const L: usize> Sub for &LInt<L> {
    type Output = LInt<L>;
    fn sub(self, other: Self) -> Self::Output {
        // For the two's complement code the additive negation is the result of
        // adding 1 to the bitwise inverted argument's representation. Thus, for
        // any encoded integers x and y we have x - y = x + !y + 1, where "!" is
        // the bitwise inversion and addition is done according to the rules of
        // the code. The algorithm below uses this formula and is the modified
        // addition algorithm, where the carry flag is initialized with "true"
        // and the chunks of the second argument are bitwise inverted
        let (mut data, mut carry) = ([0; L], true);
        for (i, d) in data.iter_mut().enumerate().take(L) {
            (*d, carry) = Self::Output::sum(self.0[i], !other.0[i], carry);
        }
        LInt::<L>(data)
    }
}

impl<const L: usize> Sub<&LInt<L>> for LInt<L> {
    type Output = LInt<L>;
    fn sub(self, other: &Self) -> Self::Output {
        &self - other
    }
}

impl<const L: usize> Sub for LInt<L> {
    type Output = LInt<L>;
    fn sub(self, other: Self) -> Self::Output {
        &self - &other
    }
}

impl<const L: usize> Neg for &LInt<L> {
    type Output = LInt<L>;
    fn neg(self) -> Self::Output {
        // For the two's complement code the additive negation is the result
        // of adding 1 to the bitwise inverted argument's representation
        let (mut data, mut carry) = ([0; L], true);
        for (i, d) in data.iter_mut().enumerate().take(L) {
            (*d, carry) = (!self.0[i]).overflowing_add(carry as u64);
        }
        LInt::<L>(data)
    }
}

impl<const L: usize> Neg for LInt<L> {
    type Output = LInt<L>;
    fn neg(self) -> Self::Output {
        -&self
    }
}

impl<const L: usize> Mul for &LInt<L> {
    type Output = LInt<L>;
    fn mul(self, other: Self) -> Self::Output {
        let mut data = [0; L];
        for i in 0..L {
            let mut carry = 0;
            for k in 0..(L - i) {
                (data[i + k], carry) =
                    Self::Output::prodsum(self.0[i], other.0[k], data[i + k], carry);
            }
        }
        LInt::<L>(data)
    }
}

impl<const L: usize> Mul<&LInt<L>> for LInt<L> {
    type Output = LInt<L>;
    fn mul(self, other: &Self) -> Self::Output {
        &self * other
    }
}

impl<const L: usize> Mul for LInt<L> {
    type Output = LInt<L>;
    fn mul(self, other: Self) -> Self::Output {
        &self * &other
    }
}

impl<const L: usize> Mul<i64> for &LInt<L> {
    type Output = LInt<L>;
    fn mul(self, other: i64) -> Self::Output {
        let mut data = [0; L];
        // If the short multiplicand is non-negative, the standard multiplication
        // algorithm is performed. Otherwise, the product of the additively negated
        // multiplicands is found as follows. Since for the two's complement code
        // the additive negation is the result of adding 1 to the bitwise inverted
        // argument's representation, for any encoded integers x and y we have
        // x * y = (-x) * (-y) = (!x + 1) * (-y) = !x * (-y) + (-y),  where "!" is
        // the bitwise inversion and arithmetic operations are performed according
        // to the rules of the code. If the short multiplicand is negative, the
        // algorithm below uses this formula by substituting the short multiplicand
        // for y and becomes the modified standard multiplication algorithm, where
        // the carry variable is being initialized with the additively negated short
        // multiplicand and the chunks of the long multiplicand are bitwise inverted
        let (other, mut carry, mask) = if other < 0 {
            (-other as u64, -other as u64, u64::MAX)
        } else {
            (other as u64, 0, 0)
        };
        for (i, d) in data.iter_mut().enumerate().take(L) {
            (*d, carry) = Self::Output::prodsum(self.0[i] ^ mask, other, 0, carry);
        }
        LInt::<L>(data)
    }
}

impl<const L: usize> Mul<i64> for LInt<L> {
    type Output = LInt<L>;
    fn mul(self, other: i64) -> Self::Output {
        &self * other
    }
}

impl<const L: usize> Mul<&LInt<L>> for i64 {
    type Output = LInt<L>;
    fn mul(self, other: &LInt<L>) -> Self::Output {
        other * self
    }
}

impl<const L: usize> Mul<LInt<L>> for i64 {
    type Output = LInt<L>;
    fn mul(self, other: LInt<L>) -> Self::Output {
        other * self
    }
}

/// Returns the "approximations" of the arguments and the flag indicating whether
/// both arguments are equal to their "approximations". Both the arguments must be
/// non-negative, and at least one of them must be non-zero. For an incorrect input,
/// the behavior of the function is undefined. These "approximations" are defined
/// in the following way. Let n be the bit length of the largest argument without
/// leading zeros. For n > 64 the "approximation" of the argument, which equals v,
/// is (v div 2 ^ (n - 32)) * 2 ^ 32 + (v mod 2 ^ 32), i.e. it retains the high and
/// low bits of the n-bit representation of v. If n does not exceed 64, an argument
/// and its "approximation" are equal. These "approximations" are defined slightly
/// differently from the ones in the Pornin's method for modular inversion: instead
/// of taking the 33 high and 31 low bits of the n-bit representation of an argument,
/// the 32 high and 32 low bits are taken
fn approximate<const L: usize>(x: &LInt<L>, y: &LInt<L>) -> (u64, u64, bool) {
    debug_assert!(
        !(x.is_negative() || y.is_negative()),
        "Both the arguments must be non-negative!"
    );
    debug_assert!(
        (*x != LInt::ZERO) || (*y != LInt::ZERO),
        "At least one argument must be non-zero!"
    );
    let mut i = L - 1;
    while (x.0[i] == 0) && (y.0[i] == 0) {
        i -= 1;
    }
    if i == 0 {
        return (x.0[0], y.0[0], true);
    }
    let mut h = (x.0[i], y.0[i]);
    let z = h.0.leading_zeros().min(h.1.leading_zeros());
    h = (h.0 << z, h.1 << z);
    if z > 32 {
        h.0 |= x.0[i - 1] >> z;
        h.1 |= y.0[i - 1] >> z;
    }
    let h = (h.0 & u64::MAX << 32, h.1 & u64::MAX << 32);
    let l = (x.0[0] & u64::MAX >> 32, y.0[0] & u64::MAX >> 32);
    (h.0 | l.0, h.1 | l.1, false)
}

/// Returns the Jacobi symbol ("n" / "d") multiplied by either 1 or -1.
/// The later multiplicand is -1 iff the second-lowest bit of "t" is 1.
/// The value of "d" must be odd in accordance with the Jacobi symbol
/// definition. For even values of "d", the behavior is not defined.
/// The implementation is based on the binary Euclidean algorithm
fn jacobinary(mut n: u64, mut d: u64, mut t: u64) -> i64 {
    debug_assert!(d & 1 > 0, "The second argument must be odd!");
    while n != 0 {
        if n & 1 > 0 {
            if n < d {
                (n, d) = (d, n);
                t ^= n & d;
            }
            n = (n - d) >> 1;
            t ^= d ^ d >> 1;
        } else {
            let z = n.trailing_zeros();
            t ^= (d ^ d >> 1) & (z << 1) as u64;
            n >>= z;
        }
    }
    (d == 1) as i64 * (1 - (t & 2) as i64)
}

/// Returns the Jacobi symbol ("n" / "d") computed by means of the modification
/// of the the Pornin's method for modular inversion. The arguments are unsigned
/// big integers in the form of arrays of 64-bit chunks, the ordering of which
/// is little-endian. The value of "d" must be odd in accordance with the Jacobi
/// symbol definition. Both the arguments must be less than 2 ^ (64 * L - 31).
/// For an incorrect input, the behavior of the function is undefined. The method
/// differs from the Pornin's method for modular inversion in absence of the parts,
/// which are not necessary to compute the greatest common divisor of arguments,
/// presence of the parts used to compute the Jacobi symbol, which are based on
/// the properties of the modified Jacobi symbol (x / |y|) described by M. Hamburg,
/// and some original optimizations. Only these differences have been commented;
/// the aforesaid Pornin's method and the used ideas of M. Hamburg were given here:
/// - T. Pornin, "Optimized Binary GCD for Modular Inversion",
/// https://eprint.iacr.org/2020/972.pdf
/// - M. Hamburg, "Computing the Jacobi symbol using Bernstein-Yang",
/// https://eprint.iacr.org/2021/1271.pdf
pub fn jacobi<const L: usize>(n: &[u64], d: &[u64]) -> i64 {
    // Instead of the variable "j" taking the values from {-1, 1} and satisfying
    // at the end of the outer loop iteration the equation J = "j" * ("n" / |"d"|)
    // for the modified Jacobi symbol ("n" / |"d"|) and the sought Jacobi symbol J,
    // we store the sign bit of "j" in the second-lowest bit of "t" for optimization
    // purposes. This approach was influenced by the paper by M. Hamburg
    let (mut n, mut d, mut t) = (LInt::<L>::new(n), LInt::<L>::new(d), 0u64);
    debug_assert!(d.0[0] & 1 > 0, "The second argument must be odd!");
    debug_assert!(
        n.0[L - 1].leading_zeros().min(d.0[L - 1].leading_zeros()) >= 31,
        "Both the arguments must be less than 2 ^ (64 * L - 31)!"
    );
    loop {
        // The inner loop performs 30 iterations instead of 31 ones in the aforementioned
        // Pornin's method, and the "approximations" of "n" and "d" retain 32 of the lowest
        // bits instead of 31 in that method. These modifications allow the values of the
        // "approximation" variables to be equal modulo 8 to the corresponding "precise"
        // variables' values, which would have been computed, if the "precise" variables
        // had been updated in the inner loop along with the "approximations". This equality
        // modulo 8 is used to update the second-lowest bit of "t" in accordance with the
        // properties of the modified Jacobi symbol (x / |y|). The admissibility of these
        // modifications has been proven using the appropriately modified Pornin's theorems
        let (mut u, mut v, mut i) = ((1i64, 0i64), (0i64, 1i64), 30);
        let (mut a, mut b, precise) = approximate(&n, &d);
        // When each "approximation" variable has the same value as the corresponding "precise"
        // one, the computation is accomplished using the short-arithmetic method of the Jacobi
        // symbol calculation by means of the binary Euclidean algorithm. This approach aims at
        // avoiding the parts of the final computations, which are related to long arithmetic
        if precise {
            return jacobinary(a, b, t);
        }
        while i > 0 {
            if a & 1 > 0 {
                if a < b {
                    (a, b, u, v) = (b, a, v, u);
                    // In both the aforesaid Pornin's method and its modification "n" and "d"
                    // could not become negative simultaneously even if they were updated after
                    // each iteration of the inner loop. Also at this point they both have odd
                    // values. Therefore, the quadratic reciprocity law for the modified Jacobi
                    // symbol (x / |y|) can be used. According to it, if both x and y are odd
                    // numbers, among which there is a positive one, then for x = y = 3 (mod 4)
                    // we have (x / |y|) = -(y / |x|) and for either x or y equal 1 modulo 4
                    // the symbols (x / |y|) and (y / |x|) are equal
                    t ^= a & b;
                }
                a = (a - b) >> 1;
                u = (u.0 - v.0, u.1 - v.1);
                v = (v.0 << 1, v.1 << 1);
                // The modified Jacobi symbol (2 / |y|) is -1, iff y mod 8 is {3, 5}
                t ^= b ^ b >> 1;
                i -= 1;
            } else {
                // Performing the batch of sequential iterations, which divide "a" by 2
                let z = i.min(a.trailing_zeros());
                // The modified Jacobi symbol (2 / |y|) is -1, iff y mod 8 is {3, 5}. However,
                // we do not need its value for a batch with an even number of divisions by 2
                t ^= (b ^ b >> 1) & (z << 1) as u64;
                v = (v.0 << z, v.1 << z);
                a >>= z;
                i -= z;
            }
        }
        (n, d) = ((&n * u.0 + &d * u.1) >> 30, (&n * v.0 + &d * v.1) >> 30);

        // This fragment is present to guarantee the correct behavior of the function
        // in the case of arguments, whose greatest common divisor is no less than 2^64
        if n == LInt::ZERO {
            // In both the aforesaid Pornin's method and its modification the pair of the values
            // of "n" and "d" after the divergence point contains a positive number and a negative
            // one. Since the value of "n" is 0, the divergence point has not been reached by the
            // inner loop this time, so there is no need to check whether "d" is equal to -1
            return (d == LInt::ONE) as i64 * (1 - (t & 2) as i64);
        }

        if n.is_negative() {
            // Since in both the aforesaid Pornin's method and its modification "d" is always odd
            // and cannot become negative simultaneously with "n", the value of "d" is positive.
            // The modified Jacobi symbol (-1 / |y|) for a positive y is -1, iff y mod 4 = 3
            t ^= d.0[0];
            n = -n;
        } else if d.is_negative() {
            // The modified Jacobi symbols (x / |y|) and (x / |-y|) are equal, so "t" is not updated
            d = -d;
        }
    }
}
