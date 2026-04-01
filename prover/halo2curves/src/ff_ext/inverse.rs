use core::cmp::PartialEq;
use std::ops::{Add, Mul, Neg, Sub};

/// Big signed (B * L)-bit integer type, whose variables store
/// numbers in the two's complement code as arrays of B-bit chunks.
/// The ordering of the chunks in these arrays is little-endian.
/// The arithmetic operations for this type are wrapping ones.
#[derive(Clone)]
struct CInt<const B: usize, const L: usize>(pub [u64; L]);

impl<const B: usize, const L: usize> CInt<B, L> {
    /// Mask, in which the B lowest bits are 1 and only they
    pub const MASK: u64 = u64::MAX >> (64 - B);

    /// Representation of -1
    pub const MINUS_ONE: Self = Self([Self::MASK; L]);

    /// Representation of 0
    pub const ZERO: Self = Self([0; L]);

    /// Representation of 1
    pub const ONE: Self = {
        let mut data = [0; L];
        data[0] = 1;
        Self(data)
    };

    /// Returns the result of applying B-bit right
    /// arithmetical shift to the current number
    pub fn shift(&self) -> Self {
        let mut data = [0; L];
        if self.is_negative() {
            data[L - 1] = Self::MASK;
        }
        data[..L - 1].copy_from_slice(&self.0[1..]);
        Self(data)
    }

    /// Returns the lowest B bits of the current number
    pub fn lowest(&self) -> u64 {
        self.0[0]
    }

    /// Returns "true" iff the current number is negative
    pub fn is_negative(&self) -> bool {
        self.0[L - 1] > (Self::MASK >> 1)
    }
}

impl<const B: usize, const L: usize> PartialEq for CInt<B, L> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<const B: usize, const L: usize> Add for &CInt<B, L> {
    type Output = CInt<B, L>;
    fn add(self, other: Self) -> Self::Output {
        let (mut data, mut carry) = ([0; L], 0);
        for (i, d) in data.iter_mut().enumerate().take(L) {
            let sum = self.0[i] + other.0[i] + carry;
            *d = sum & CInt::<B, L>::MASK;
            carry = sum >> B;
        }
        CInt::<B, L>(data)
    }
}

impl<const B: usize, const L: usize> Add<&CInt<B, L>> for CInt<B, L> {
    type Output = CInt<B, L>;
    fn add(self, other: &Self) -> Self::Output {
        &self + other
    }
}

impl<const B: usize, const L: usize> Add for CInt<B, L> {
    type Output = CInt<B, L>;
    fn add(self, other: Self) -> Self::Output {
        &self + &other
    }
}

impl<const B: usize, const L: usize> Sub for &CInt<B, L> {
    type Output = CInt<B, L>;
    fn sub(self, other: Self) -> Self::Output {
        // For the two's complement code the additive negation is the result of
        // adding 1 to the bitwise inverted argument's representation. Thus, for
        // any encoded integers x and y we have x - y = x + !y + 1, where "!" is
        // the bitwise inversion and addition is done according to the rules of
        // the code. The algorithm below uses this formula and is the modified
        // addition algorithm, where the carry flag is initialized with 1 and
        // the chunks of the second argument are bitwise inverted
        let (mut data, mut carry) = ([0; L], 1);
        for (i, d) in data.iter_mut().enumerate().take(L) {
            let sum = self.0[i] + (other.0[i] ^ CInt::<B, L>::MASK) + carry;
            *d = sum & CInt::<B, L>::MASK;
            carry = sum >> B;
        }
        CInt::<B, L>(data)
    }
}

impl<const B: usize, const L: usize> Sub<&CInt<B, L>> for CInt<B, L> {
    type Output = CInt<B, L>;
    fn sub(self, other: &Self) -> Self::Output {
        &self - other
    }
}

impl<const B: usize, const L: usize> Sub for CInt<B, L> {
    type Output = CInt<B, L>;
    fn sub(self, other: Self) -> Self::Output {
        &self - &other
    }
}

impl<const B: usize, const L: usize> Neg for &CInt<B, L> {
    type Output = CInt<B, L>;
    fn neg(self) -> Self::Output {
        // For the two's complement code the additive negation is the result
        // of adding 1 to the bitwise inverted argument's representation
        let (mut data, mut carry) = ([0; L], 1);
        for (i, d) in data.iter_mut().enumerate().take(L) {
            let sum = (self.0[i] ^ CInt::<B, L>::MASK) + carry;
            *d = sum & CInt::<B, L>::MASK;
            carry = sum >> B;
        }
        CInt::<B, L>(data)
    }
}

impl<const B: usize, const L: usize> Neg for CInt<B, L> {
    type Output = CInt<B, L>;
    fn neg(self) -> Self::Output {
        -&self
    }
}

impl<const B: usize, const L: usize> Mul for &CInt<B, L> {
    type Output = CInt<B, L>;
    fn mul(self, other: Self) -> Self::Output {
        let mut data = [0; L];
        for i in 0..L {
            let mut carry = 0;
            for k in 0..(L - i) {
                let sum = (data[i + k] as u128)
                    + (carry as u128)
                    + (self.0[i] as u128) * (other.0[k] as u128);
                data[i + k] = sum as u64 & CInt::<B, L>::MASK;
                carry = (sum >> B) as u64;
            }
        }
        CInt::<B, L>(data)
    }
}

impl<const B: usize, const L: usize> Mul<&CInt<B, L>> for CInt<B, L> {
    type Output = CInt<B, L>;
    fn mul(self, other: &Self) -> Self::Output {
        &self * other
    }
}

impl<const B: usize, const L: usize> Mul for CInt<B, L> {
    type Output = CInt<B, L>;
    fn mul(self, other: Self) -> Self::Output {
        &self * &other
    }
}

impl<const B: usize, const L: usize> Mul<i64> for &CInt<B, L> {
    type Output = CInt<B, L>;
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
        // for y and turns into the modified standard multiplication algorithm,
        // where the carry flag is initialized with the additively negated short
        // multiplicand and the chunks of the long multiplicand are bitwise inverted
        let (other, mut carry, mask) = if other < 0 {
            (-other, -other as u64, CInt::<B, L>::MASK)
        } else {
            (other, 0, 0)
        };
        for (i, d) in data.iter_mut().enumerate().take(L) {
            let sum = (carry as u128) + ((self.0[i] ^ mask) as u128) * (other as u128);
            *d = sum as u64 & CInt::<B, L>::MASK;
            carry = (sum >> B) as u64;
        }
        CInt::<B, L>(data)
    }
}

impl<const B: usize, const L: usize> Mul<i64> for CInt<B, L> {
    type Output = CInt<B, L>;
    fn mul(self, other: i64) -> Self::Output {
        &self * other
    }
}

impl<const B: usize, const L: usize> Mul<&CInt<B, L>> for i64 {
    type Output = CInt<B, L>;
    fn mul(self, other: &CInt<B, L>) -> Self::Output {
        other * self
    }
}

impl<const B: usize, const L: usize> Mul<CInt<B, L>> for i64 {
    type Output = CInt<B, L>;
    fn mul(self, other: CInt<B, L>) -> Self::Output {
        other * self
    }
}

/// Type of the modular multiplicative inverter based on the Bernstein-Yang method.
/// The inverter can be created for a specified modulus M and adjusting parameter A
/// to compute the adjusted multiplicative inverses of positive integers, i.e. for
/// computing (1 / x) * A (mod M) for a positive integer x.
///
/// The adjusting parameter allows computing the multiplicative inverses in the case of
/// using the Montgomery representation for the input or the expected output. If R is
/// the Montgomery factor, the multiplicative inverses in the appropriate representation
/// can be computed provided that the value of A is chosen as follows:
/// - A = 1, if both the input and the expected output are in the standard form
/// - A = R^2 mod M, if both the input and the expected output are in the Montgomery form
/// - A = R mod M, if either the input or the expected output is in the Montgomery form,
/// but not both of them
///
/// The public methods of this type receive and return unsigned big integers as arrays of
/// 64-bit chunks, the ordering of which is little-endian. Both the modulus and the integer
/// to be inverted should not exceed 2 ^ (62 * L - 64)
///
/// For better understanding the implementation, the following resources are recommended:
/// - D. Bernstein, B.-Y. Yang, "Fast constant-time gcd computation and modular inversion",
/// https://gcd.cr.yp.to/safegcd-20190413.pdf
/// - P. Wuille, "The safegcd implementation in libsecp256k1 explained",
/// https://github.com/bitcoin-core/secp256k1/blob/master/doc/safegcd_implementation.md
pub struct BYInverter<const L: usize> {
    /// Modulus
    modulus: CInt<62, L>,

    /// Adjusting parameter
    adjuster: CInt<62, L>,

    /// Multiplicative inverse of the modulus modulo 2^62
    inverse: i64,
}

/// Type of the Bernstein-Yang transition matrix multiplied by 2^62
type Matrix = [[i64; 2]; 2];

impl<const L: usize> BYInverter<L> {
    /// Returns the Bernstein-Yang transition matrix multiplied by 2^62 and the new value
    /// of the delta variable for the 62 basic steps of the Bernstein-Yang method, which
    /// are to be performed sequentially for specified initial values of f, g and delta
    fn jump(f: &CInt<62, L>, g: &CInt<62, L>, mut delta: i64) -> (i64, Matrix) {
        let (mut steps, mut f, mut g) = (62, f.lowest() as i64, g.lowest() as i128);
        let mut t: Matrix = [[1, 0], [0, 1]];

        loop {
            let zeros = steps.min(g.trailing_zeros() as i64);
            (steps, delta, g) = (steps - zeros, delta + zeros, g >> zeros);
            t[0] = [t[0][0] << zeros, t[0][1] << zeros];

            if steps == 0 {
                break;
            }
            if delta > 0 {
                (delta, f, g) = (-delta, g as i64, -f as i128);
                (t[0], t[1]) = (t[1], [-t[0][0], -t[0][1]]);
            }

            // The formula (3 * x) xor 28 = -1 / x (mod 32) for an odd integer x
            // in the two's complement code has been derived from the formula
            // (3 * x) xor 2 = 1 / x (mod 32) attributed to Peter Montgomery
            let mask = (1 << steps.min(1 - delta).min(5)) - 1;
            let w = (g as i64).wrapping_mul(f.wrapping_mul(3) ^ 28) & mask;

            t[1] = [t[0][0] * w + t[1][0], t[0][1] * w + t[1][1]];
            g += w as i128 * f as i128;
        }

        (delta, t)
    }

    /// Returns the updated values of the variables f and g for specified initial ones and Bernstein-Yang transition
    /// matrix multiplied by 2^62. The returned vector is "matrix * (f, g)' / 2^62", where "'" is the transpose operator
    fn fg(f: CInt<62, L>, g: CInt<62, L>, t: Matrix) -> (CInt<62, L>, CInt<62, L>) {
        (
            (t[0][0] * &f + t[0][1] * &g).shift(),
            (t[1][0] * &f + t[1][1] * &g).shift(),
        )
    }

    /// Returns the updated values of the variables d and e for specified initial ones and Bernstein-Yang transition
    /// matrix multiplied by 2^62. The returned vector is congruent modulo M to "matrix * (d, e)' / 2^62 (mod M)",
    /// where M is the modulus the inverter was created for and "'" stands for the transpose operator. Both the input
    /// and output values lie in the interval (-2 * M, M)
    fn de(&self, d: CInt<62, L>, e: CInt<62, L>, t: Matrix) -> (CInt<62, L>, CInt<62, L>) {
        let mask = CInt::<62, L>::MASK as i64;
        let mut md = t[0][0] * d.is_negative() as i64 + t[0][1] * e.is_negative() as i64;
        let mut me = t[1][0] * d.is_negative() as i64 + t[1][1] * e.is_negative() as i64;

        let cd = t[0][0]
            .wrapping_mul(d.lowest() as i64)
            .wrapping_add(t[0][1].wrapping_mul(e.lowest() as i64))
            & mask;
        let ce = t[1][0]
            .wrapping_mul(d.lowest() as i64)
            .wrapping_add(t[1][1].wrapping_mul(e.lowest() as i64))
            & mask;

        md -= (self.inverse.wrapping_mul(cd).wrapping_add(md)) & mask;
        me -= (self.inverse.wrapping_mul(ce).wrapping_add(me)) & mask;

        let cd = t[0][0] * &d + t[0][1] * &e + md * &self.modulus;
        let ce = t[1][0] * &d + t[1][1] * &e + me * &self.modulus;

        (cd.shift(), ce.shift())
    }

    /// Returns either "value (mod M)" or "-value (mod M)", where M is the modulus the
    /// inverter was created for, depending on "negate", which determines the presence
    /// of "-" in the used formula. The input integer lies in the interval (-2 * M, M)
    fn norm(&self, mut value: CInt<62, L>, negate: bool) -> CInt<62, L> {
        if value.is_negative() {
            value = value + &self.modulus;
        }

        if negate {
            value = -value;
        }

        if value.is_negative() {
            value = value + &self.modulus;
        }

        value
    }

    /// Returns a big unsigned integer as an array of O-bit chunks, which is equal modulo
    /// 2 ^ (O * S) to the input big unsigned integer stored as an array of I-bit chunks.
    /// The ordering of the chunks in these arrays is little-endian
    const fn convert<const I: usize, const O: usize, const S: usize>(input: &[u64]) -> [u64; S] {
        // This function is defined because the method "min" of the usize type is not constant
        const fn min(a: usize, b: usize) -> usize {
            if a > b {
                b
            } else {
                a
            }
        }

        let (total, mut output, mut bits) = (min(input.len() * I, S * O), [0; S], 0);

        while bits < total {
            let (i, o) = (bits % I, bits % O);
            output[bits / O] |= (input[bits / I] >> i) << o;
            bits += min(I - i, O - o);
        }

        let mask = u64::MAX >> (64 - O);
        let mut filled = total / O + if total % O > 0 { 1 } else { 0 };

        while filled > 0 {
            filled -= 1;
            output[filled] &= mask;
        }

        output
    }

    /// Returns the multiplicative inverse of the argument modulo 2^62. The implementation is based
    /// on the Hurchalla's method for computing the multiplicative inverse modulo a power of two.
    /// For better understanding the implementation, the following paper is recommended:
    /// J. Hurchalla, "An Improved Integer Multiplicative Inverse (modulo 2^w)",
    /// https://arxiv.org/pdf/2204.04342.pdf
    const fn inv(value: u64) -> i64 {
        let x = value.wrapping_mul(3) ^ 2;
        let y = 1u64.wrapping_sub(x.wrapping_mul(value));
        let (x, y) = (x.wrapping_mul(y.wrapping_add(1)), y.wrapping_mul(y));
        let (x, y) = (x.wrapping_mul(y.wrapping_add(1)), y.wrapping_mul(y));
        let (x, y) = (x.wrapping_mul(y.wrapping_add(1)), y.wrapping_mul(y));
        (x.wrapping_mul(y.wrapping_add(1)) & CInt::<62, L>::MASK) as i64
    }

    /// Creates the inverter for specified modulus and adjusting parameter
    pub const fn new(modulus: &[u64], adjuster: &[u64]) -> Self {
        Self {
            modulus: CInt::<62, L>(Self::convert::<64, 62, L>(modulus)),
            adjuster: CInt::<62, L>(Self::convert::<64, 62, L>(adjuster)),
            inverse: Self::inv(modulus[0]),
        }
    }

    /// Returns either the adjusted modular multiplicative inverse for the argument or None
    /// depending on invertibility of the argument, i.e. its coprimality with the modulus
    pub fn invert<const S: usize>(&self, value: &[u64]) -> Option<[u64; S]> {
        let (mut d, mut e) = (CInt::ZERO, self.adjuster.clone());
        let mut g = CInt::<62, L>(Self::convert::<64, 62, L>(value));
        let (mut delta, mut f) = (1, self.modulus.clone());
        let mut matrix;
        while g != CInt::ZERO {
            (delta, matrix) = Self::jump(&f, &g, delta);
            (f, g) = Self::fg(f, g, matrix);
            (d, e) = self.de(d, e, matrix);
        }
        // At this point the absolute value of "f" equals the greatest common divisor
        // of the integer to be inverted and the modulus the inverter was created for.
        // Thus, if "f" is neither 1 nor -1, then the sought inverse does not exist
        let antiunit = f == CInt::MINUS_ONE;
        if (f != CInt::ONE) && !antiunit {
            return None;
        }
        Some(Self::convert::<62, 64, S>(&self.norm(d, antiunit).0))
    }
}
