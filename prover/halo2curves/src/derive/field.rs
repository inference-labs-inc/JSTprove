#[macro_export]
macro_rules! impl_from_u64_u32 {
    ($field:ident, $r2:ident) => {
        impl From<u64> for $field {
            fn from(val: u64) -> $field {
                $field([val, 0, 0, 0]) * $r2
            }
        }

        impl From<u32> for $field {
            fn from(val: u32) -> $field {
                $field([val as u64, 0, 0, 0]) * $r2
            }
        }
    };
}

#[macro_export]
macro_rules! field_common {
    (
        $field:ident,
        $modulus:ident,
        $inv:ident,
        $modulus_str:ident,
        $two_inv:ident,
        $root_of_unity_inv:ident,
        $delta:ident,
        $zeta:ident,
        $r:ident,
        $r2:ident,
        $r3:ident
    ) => {
        /// Bernstein-Yang modular multiplicative inverter created for the modulus equal to
        /// the characteristic of the field to invert positive integers in the Montgomery form.
        const BYINVERTOR: $crate::ff_ext::inverse::BYInverter<6> =
            $crate::ff_ext::inverse::BYInverter::<6>::new(&$modulus.0, &$r2.0);

        impl $field {
            /// Returns zero, the additive identity.
            #[inline]
            pub const fn zero() -> $field {
                $field([0, 0, 0, 0])
            }

            /// Returns one, the multiplicative identity.
            #[inline]
            pub const fn one() -> $field {
                $r
            }

            /// Returns the multiplicative inverse of the
            /// element. If it is zero, the method fails.
            #[inline(always)]
            pub fn invert(&self) -> CtOption<Self> {
                if let Some(inverse) = BYINVERTOR.invert(&self.0) {
                    CtOption::new(Self(inverse), Choice::from(1))
                } else {
                    CtOption::new(Self::zero(), Choice::from(0))
                }
            }

            // Returns the Jacobi symbol, where the numerator and denominator
            // are the element and the characteristic of the field, respectively.
            // The Jacobi symbol is applicable to odd moduli
            // while the Legendre symbol is applicable to prime moduli.
            // They are equivalent for prime moduli.
            #[inline(always)]
            pub fn jacobi(&self) -> i64 {
                $crate::ff_ext::jacobi::jacobi::<5>(&self.0, &$modulus.0)
            }

            const fn montgomery_form(val: [u64; 4], r: $field) -> $field {
                // Converts a 4 64-bit limb value into its congruent field representation.
                // If `val` represents a 256 bit value then `r` should be R^2,
                // if `val` represents the 256 MSB of a 512 bit value, then `r` should be R^3.

                let (r0, carry) = mac(0, val[0], r.0[0], 0);
                let (r1, carry) = mac(0, val[0], r.0[1], carry);
                let (r2, carry) = mac(0, val[0], r.0[2], carry);
                let (r3, r4) = mac(0, val[0], r.0[3], carry);

                let (r1, carry) = mac(r1, val[1], r.0[0], 0);
                let (r2, carry) = mac(r2, val[1], r.0[1], carry);
                let (r3, carry) = mac(r3, val[1], r.0[2], carry);
                let (r4, r5) = mac(r4, val[1], r.0[3], carry);

                let (r2, carry) = mac(r2, val[2], r.0[0], 0);
                let (r3, carry) = mac(r3, val[2], r.0[1], carry);
                let (r4, carry) = mac(r4, val[2], r.0[2], carry);
                let (r5, r6) = mac(r5, val[2], r.0[3], carry);

                let (r3, carry) = mac(r3, val[3], r.0[0], 0);
                let (r4, carry) = mac(r4, val[3], r.0[1], carry);
                let (r5, carry) = mac(r5, val[3], r.0[2], carry);
                let (r6, r7) = mac(r6, val[3], r.0[3], carry);

                // Montgomery reduction
                let k = r0.wrapping_mul($inv);
                let (_, carry) = mac(r0, k, $modulus.0[0], 0);
                let (r1, carry) = mac(r1, k, $modulus.0[1], carry);
                let (r2, carry) = mac(r2, k, $modulus.0[2], carry);
                let (r3, carry) = mac(r3, k, $modulus.0[3], carry);
                let (r4, carry2) = adc(r4, 0, carry);

                let k = r1.wrapping_mul($inv);
                let (_, carry) = mac(r1, k, $modulus.0[0], 0);
                let (r2, carry) = mac(r2, k, $modulus.0[1], carry);
                let (r3, carry) = mac(r3, k, $modulus.0[2], carry);
                let (r4, carry) = mac(r4, k, $modulus.0[3], carry);
                let (r5, carry2) = adc(r5, carry2, carry);

                let k = r2.wrapping_mul($inv);
                let (_, carry) = mac(r2, k, $modulus.0[0], 0);
                let (r3, carry) = mac(r3, k, $modulus.0[1], carry);
                let (r4, carry) = mac(r4, k, $modulus.0[2], carry);
                let (r5, carry) = mac(r5, k, $modulus.0[3], carry);
                let (r6, carry2) = adc(r6, carry2, carry);

                let k = r3.wrapping_mul($inv);
                let (_, carry) = mac(r3, k, $modulus.0[0], 0);
                let (r4, carry) = mac(r4, k, $modulus.0[1], carry);
                let (r5, carry) = mac(r5, k, $modulus.0[2], carry);
                let (r6, carry) = mac(r6, k, $modulus.0[3], carry);
                let (r7, carry2) = adc(r7, carry2, carry);

                // Result may be within MODULUS of the correct value
                let (d0, borrow) = sbb(r4, $modulus.0[0], 0);
                let (d1, borrow) = sbb(r5, $modulus.0[1], borrow);
                let (d2, borrow) = sbb(r6, $modulus.0[2], borrow);
                let (d3, borrow) = sbb(r7, $modulus.0[3], borrow);
                let (_, borrow) = sbb(carry2, 0, borrow);
                let (d0, carry) = adc(d0, $modulus.0[0] & borrow, 0);
                let (d1, carry) = adc(d1, $modulus.0[1] & borrow, carry);
                let (d2, carry) = adc(d2, $modulus.0[2] & borrow, carry);
                let (d3, _) = adc(d3, $modulus.0[3] & borrow, carry);

                $field([d0, d1, d2, d3])
            }

            fn from_u512(limbs: [u64; 8]) -> $field {
                // We reduce an arbitrary 512-bit number by decomposing it into two 256-bit digits
                // with the higher bits multiplied by 2^256. Thus, we perform two reductions
                //
                // 1. the lower bits are multiplied by R^2, as normal
                // 2. the upper bits are multiplied by R^2 * 2^256 = R^3
                //
                // and computing their sum in the field. It remains to see that arbitrary 256-bit
                // numbers can be placed into Montgomery form safely using the reduction. The
                // reduction works so long as the product is less than R=2^256 multiplied by
                // the modulus. This holds because for any `c` smaller than the modulus, we have
                // that (2^256 - 1)*c is an acceptable product for the reduction. Therefore, the
                // reduction always works so long as `c` is in the field; in this case it is either the
                // constant `R2` or `R3`.

                let lower_256 = [limbs[0], limbs[1], limbs[2], limbs[3]];
                let upper_256 = [limbs[4], limbs[5], limbs[6], limbs[7]];

                Self::montgomery_form(lower_256, $r2) + Self::montgomery_form(upper_256, $r3)
            }

            /// Converts from an integer represented in little endian
            /// into its (congruent) `$field` representation.
            pub const fn from_raw(val: [u64; 4]) -> Self {
                Self::montgomery_form(val, $r2)
            }

            /// Attempts to convert a little-endian byte representation of
            /// a scalar into a `Fr`, failing if the input is not canonical.
            pub fn from_bytes(bytes: &[u8; 32]) -> CtOption<$field> {
                <Self as ff::PrimeField>::from_repr(*bytes)
            }

            /// Converts an element of `Fr` into a byte representation in
            /// little-endian byte order.
            pub fn to_bytes(&self) -> [u8; 32] {
                <Self as ff::PrimeField>::to_repr(self)
            }

            /// Lexicographic comparison of Montgomery forms.
            #[inline(always)]
            const fn is_less_than(x: &[u64; 4], y: &[u64; 4]) -> bool {
                let (_, borrow) = sbb(x[0], y[0], 0);
                let (_, borrow) = sbb(x[1], y[1], borrow);
                let (_, borrow) = sbb(x[2], y[2], borrow);
                let (_, borrow) = sbb(x[3], y[3], borrow);
                borrow >> 63 == 1
            }
        }

        impl fmt::Debug for $field {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let tmp = self.to_repr();
                write!(f, "0x")?;
                for &b in tmp.iter().rev() {
                    write!(f, "{:02x}", b)?;
                }
                Ok(())
            }
        }

        impl Default for $field {
            #[inline]
            fn default() -> Self {
                Self::zero()
            }
        }

        impl From<bool> for $field {
            fn from(bit: bool) -> $field {
                if bit {
                    $field::one()
                } else {
                    $field::zero()
                }
            }
        }

        impl ConstantTimeEq for $field {
            fn ct_eq(&self, other: &Self) -> Choice {
                self.0[0].ct_eq(&other.0[0])
                    & self.0[1].ct_eq(&other.0[1])
                    & self.0[2].ct_eq(&other.0[2])
                    & self.0[3].ct_eq(&other.0[3])
            }
        }

        impl core::cmp::Ord for $field {
            fn cmp(&self, other: &Self) -> core::cmp::Ordering {
                let left = self.to_repr();
                let right = other.to_repr();
                left.iter()
                    .zip(right.iter())
                    .rev()
                    .find_map(|(left_byte, right_byte)| match left_byte.cmp(right_byte) {
                        core::cmp::Ordering::Equal => None,
                        res => Some(res),
                    })
                    .unwrap_or(core::cmp::Ordering::Equal)
            }
        }

        impl core::cmp::PartialOrd for $field {
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        impl ConditionallySelectable for $field {
            fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
                $field([
                    u64::conditional_select(&a.0[0], &b.0[0], choice),
                    u64::conditional_select(&a.0[1], &b.0[1], choice),
                    u64::conditional_select(&a.0[2], &b.0[2], choice),
                    u64::conditional_select(&a.0[3], &b.0[3], choice),
                ])
            }
        }

        impl<'a> Neg for &'a $field {
            type Output = $field;

            #[inline]
            fn neg(self) -> $field {
                self.neg()
            }
        }

        impl Neg for $field {
            type Output = $field;

            #[inline]
            fn neg(self) -> $field {
                -&self
            }
        }

        impl<'a, 'b> Sub<&'b $field> for &'a $field {
            type Output = $field;

            #[inline]
            fn sub(self, rhs: &'b $field) -> $field {
                self.sub(rhs)
            }
        }

        impl<'a, 'b> Add<&'b $field> for &'a $field {
            type Output = $field;

            #[inline]
            fn add(self, rhs: &'b $field) -> $field {
                self.add(rhs)
            }
        }

        impl<'a, 'b> Mul<&'b $field> for &'a $field {
            type Output = $field;

            #[inline]
            fn mul(self, rhs: &'b $field) -> $field {
                self.mul(rhs)
            }
        }

        impl From<[u64; 4]> for $field {
            fn from(digits: [u64; 4]) -> Self {
                Self::from_raw(digits)
            }
        }

        impl From<$field> for [u8; 32] {
            fn from(value: $field) -> [u8; 32] {
                value.to_repr()
            }
        }

        impl<'a> From<&'a $field> for [u8; 32] {
            fn from(value: &'a $field) -> [u8; 32] {
                value.to_repr()
            }
        }

        impl $crate::serde::SerdeObject for $field {
            fn from_raw_bytes_unchecked(bytes: &[u8]) -> Self {
                debug_assert_eq!(bytes.len(), 32);
                let inner =
                    [0, 8, 16, 24].map(|i| u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap()));
                Self(inner)
            }
            fn from_raw_bytes(bytes: &[u8]) -> Option<Self> {
                if bytes.len() != 32 {
                    return None;
                }
                let elt = Self::from_raw_bytes_unchecked(bytes);
                Self::is_less_than(&elt.0, &$modulus.0).then(|| elt)
            }
            fn to_raw_bytes(&self) -> Vec<u8> {
                let mut res = Vec::with_capacity(32);
                for limb in self.0.iter() {
                    res.extend_from_slice(&limb.to_le_bytes());
                }
                res
            }
            fn read_raw_unchecked<R: std::io::Read>(reader: &mut R) -> Self {
                let inner = [(); 4].map(|_| {
                    let mut buf = [0; 8];
                    reader.read_exact(&mut buf).unwrap();
                    u64::from_le_bytes(buf)
                });
                Self(inner)
            }
            fn read_raw<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
                let mut inner = [0u64; 4];
                for limb in inner.iter_mut() {
                    let mut buf = [0; 8];
                    reader.read_exact(&mut buf)?;
                    *limb = u64::from_le_bytes(buf);
                }
                let elt = Self(inner);
                Self::is_less_than(&elt.0, &$modulus.0)
                    .then(|| elt)
                    .ok_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "input number is not less than field modulus",
                        )
                    })
            }
            fn write_raw<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
                for limb in self.0.iter() {
                    writer.write_all(&limb.to_le_bytes())?;
                }
                Ok(())
            }
        }
    };
}

#[macro_export]
macro_rules! field_arithmetic {
    ($field:ident, $modulus:ident, $inv:ident, $field_type:ident) => {
        field_specific!($field, $modulus, $inv, $field_type);

        impl $field {
            /// Doubles this field element.
            #[inline]
            pub const fn double(&self) -> $field {
                self.add(self)
            }

            /// Squares this element.
            #[inline]
            pub const fn square(&self) -> $field {
                let (r1, carry) = mac(0, self.0[0], self.0[1], 0);
                let (r2, carry) = mac(0, self.0[0], self.0[2], carry);
                let (r3, r4) = mac(0, self.0[0], self.0[3], carry);

                let (r3, carry) = mac(r3, self.0[1], self.0[2], 0);
                let (r4, r5) = mac(r4, self.0[1], self.0[3], carry);

                let (r5, r6) = mac(r5, self.0[2], self.0[3], 0);

                let r7 = r6 >> 63;
                let r6 = (r6 << 1) | (r5 >> 63);
                let r5 = (r5 << 1) | (r4 >> 63);
                let r4 = (r4 << 1) | (r3 >> 63);
                let r3 = (r3 << 1) | (r2 >> 63);
                let r2 = (r2 << 1) | (r1 >> 63);
                let r1 = r1 << 1;

                let (r0, carry) = mac(0, self.0[0], self.0[0], 0);
                let (r1, carry) = adc(0, r1, carry);
                let (r2, carry) = mac(r2, self.0[1], self.0[1], carry);
                let (r3, carry) = adc(0, r3, carry);
                let (r4, carry) = mac(r4, self.0[2], self.0[2], carry);
                let (r5, carry) = adc(0, r5, carry);
                let (r6, carry) = mac(r6, self.0[3], self.0[3], carry);
                let (r7, _) = adc(0, r7, carry);

                $field::montgomery_reduce(&[r0, r1, r2, r3, r4, r5, r6, r7])
            }

            /// Multiplies `rhs` by `self`, returning the result.
            #[inline(always)]
            #[unroll::unroll_for_loops]
            #[allow(unused_assignments)]
            pub const fn mul(&self, rhs: &Self) -> Self {
                // Fast Coarsely Integrated Operand Scanning (CIOS) as described
                // in Algorithm 2 of EdMSM: https://eprint.iacr.org/2022/1400.pdf
                //
                // Cannot use the fast version (algorithm 2) if
                // modulus_high_word >= (WORD_SIZE - 1) / 2 - 1 = (2^64 - 1)/2 - 1

                if $modulus.0[3] < (u64::MAX / 2) {
                    const N: usize = 4;
                    let mut t: [u64; N] = [0u64; N];
                    let mut c_2: u64;
                    for i in 0..4 {
                        let mut c: u64 = 0u64;
                        for j in 0..4 {
                            (t[j], c) = mac(t[j], self.0[j], rhs.0[i], c);
                        }
                        c_2 = c;

                        let m = t[0].wrapping_mul(INV);
                        (_, c) = macx(t[0], m, $modulus.0[0]);

                        for j in 1..4 {
                            (t[j - 1], c) = mac(t[j], m, $modulus.0[j], c);
                        }
                        (t[N - 1], _) = adc(c_2, c, 0);
                    }

                    if bigint_geq(&t, &$modulus.0) {
                        let mut borrow = 0;
                        (t[0], borrow) = sbb(t[0], $modulus.0[0], borrow);
                        (t[1], borrow) = sbb(t[1], $modulus.0[1], borrow);
                        (t[2], borrow) = sbb(t[2], $modulus.0[2], borrow);
                        (t[3], borrow) = sbb(t[3], $modulus.0[3], borrow);
                    }
                    $field(t)
                } else {
                    // Schoolbook multiplication

                    let (r0, carry) = mac(0, self.0[0], rhs.0[0], 0);
                    let (r1, carry) = mac(0, self.0[0], rhs.0[1], carry);
                    let (r2, carry) = mac(0, self.0[0], rhs.0[2], carry);
                    let (r3, r4) = mac(0, self.0[0], rhs.0[3], carry);

                    let (r1, carry) = mac(r1, self.0[1], rhs.0[0], 0);
                    let (r2, carry) = mac(r2, self.0[1], rhs.0[1], carry);
                    let (r3, carry) = mac(r3, self.0[1], rhs.0[2], carry);
                    let (r4, r5) = mac(r4, self.0[1], rhs.0[3], carry);

                    let (r2, carry) = mac(r2, self.0[2], rhs.0[0], 0);
                    let (r3, carry) = mac(r3, self.0[2], rhs.0[1], carry);
                    let (r4, carry) = mac(r4, self.0[2], rhs.0[2], carry);
                    let (r5, r6) = mac(r5, self.0[2], rhs.0[3], carry);

                    let (r3, carry) = mac(r3, self.0[3], rhs.0[0], 0);
                    let (r4, carry) = mac(r4, self.0[3], rhs.0[1], carry);
                    let (r5, carry) = mac(r5, self.0[3], rhs.0[2], carry);
                    let (r6, r7) = mac(r6, self.0[3], rhs.0[3], carry);

                    $field::montgomery_reduce(&[r0, r1, r2, r3, r4, r5, r6, r7])
                }
            }

            /// Subtracts `rhs` from `self`, returning the result.
            #[inline]
            pub const fn sub(&self, rhs: &Self) -> Self {
                let (d0, borrow) = sbb(self.0[0], rhs.0[0], 0);
                let (d1, borrow) = sbb(self.0[1], rhs.0[1], borrow);
                let (d2, borrow) = sbb(self.0[2], rhs.0[2], borrow);
                let (d3, borrow) = sbb(self.0[3], rhs.0[3], borrow);

                // If underflow occurred on the final limb, borrow = 0xfff...fff, otherwise
                // borrow = 0x000...000. Thus, we use it as a mask to conditionally add the modulus.
                let (d0, carry) = adc(d0, $modulus.0[0] & borrow, 0);
                let (d1, carry) = adc(d1, $modulus.0[1] & borrow, carry);
                let (d2, carry) = adc(d2, $modulus.0[2] & borrow, carry);
                let (d3, _) = adc(d3, $modulus.0[3] & borrow, carry);

                $field([d0, d1, d2, d3])
            }

            /// Negates `self`.
            #[inline]
            pub const fn neg(&self) -> Self {
                // Subtract `self` from `MODULUS` to negate. Ignore the final
                // borrow because it cannot underflow; self is guaranteed to
                // be in the field.
                let (d0, borrow) = sbb($modulus.0[0], self.0[0], 0);
                let (d1, borrow) = sbb($modulus.0[1], self.0[1], borrow);
                let (d2, borrow) = sbb($modulus.0[2], self.0[2], borrow);
                let (d3, _) = sbb($modulus.0[3], self.0[3], borrow);

                // `tmp` could be `MODULUS` if `self` was zero. Create a mask that is
                // zero if `self` was zero, and `u64::max_value()` if self was nonzero.
                let mask =
                    (((self.0[0] | self.0[1] | self.0[2] | self.0[3]) == 0) as u64).wrapping_sub(1);

                $field([d0 & mask, d1 & mask, d2 & mask, d3 & mask])
            }

            /// Montgomery reduce where last 4 registers are 0
            #[inline(always)]
            pub(crate) const fn montgomery_reduce_short(r: &[u64; 4]) -> $field {
                // The Montgomery reduction here is based on Algorithm 14.32 in
                // Handbook of Applied Cryptography
                // <http://cacr.uwaterloo.ca/hac/about/chap14.pdf>.

                let k = r[0].wrapping_mul($inv);
                let (_, r0) = macx(r[0], k, $modulus.0[0]);
                let (r1, r0) = mac(r[1], k, $modulus.0[1], r0);
                let (r2, r0) = mac(r[2], k, $modulus.0[2], r0);
                let (r3, r0) = mac(r[3], k, $modulus.0[3], r0);

                let k = r1.wrapping_mul($inv);
                let (_, r1) = macx(r1, k, $modulus.0[0]);
                let (r2, r1) = mac(r2, k, $modulus.0[1], r1);
                let (r3, r1) = mac(r3, k, $modulus.0[2], r1);
                let (r0, r1) = mac(r0, k, $modulus.0[3], r1);

                let k = r2.wrapping_mul($inv);
                let (_, r2) = macx(r2, k, $modulus.0[0]);
                let (r3, r2) = mac(r3, k, $modulus.0[1], r2);
                let (r0, r2) = mac(r0, k, $modulus.0[2], r2);
                let (r1, r2) = mac(r1, k, $modulus.0[3], r2);

                let k = r3.wrapping_mul($inv);
                let (_, r3) = macx(r3, k, $modulus.0[0]);
                let (r0, r3) = mac(r0, k, $modulus.0[1], r3);
                let (r1, r3) = mac(r1, k, $modulus.0[2], r3);
                let (r2, r3) = mac(r2, k, $modulus.0[3], r3);

                // Result may be within MODULUS of the correct value
                (&$field([r0, r1, r2, r3])).sub(&$modulus)
            }
        }

        impl From<$field> for [u64; 4] {
            fn from(elt: $field) -> [u64; 4] {
                // Turn into canonical form by computing
                // (a.R) / R = a
                $field::montgomery_reduce_short(&elt.0).0
            }
        }
    };
}

#[macro_export]
macro_rules! field_specific {
    ($field:ident, $modulus:ident, $inv:ident, sparse) => {
        impl $field {
            /// Adds `rhs` to `self`, returning the result.
            #[inline]
            pub const fn add(&self, rhs: &Self) -> Self {
                let (d0, carry) = adc(self.0[0], rhs.0[0], 0);
                let (d1, carry) = adc(self.0[1], rhs.0[1], carry);
                let (d2, carry) = adc(self.0[2], rhs.0[2], carry);
                let (d3, _) = adc(self.0[3], rhs.0[3], carry);

                // Attempt to subtract the modulus, to ensure the value
                // is smaller than the modulus.
                (&$field([d0, d1, d2, d3])).sub(&$modulus)
            }

            #[inline(always)]
            pub(crate) const fn montgomery_reduce(r: &[u64; 8]) -> $field {
                // The Montgomery reduction here is based on Algorithm 14.32 in
                // Handbook of Applied Cryptography
                // <http://cacr.uwaterloo.ca/hac/about/chap14.pdf>.

                let k = r[0].wrapping_mul($inv);
                let (_, carry) = mac(r[0], k, $modulus.0[0], 0);
                let (r1, carry) = mac(r[1], k, $modulus.0[1], carry);
                let (r2, carry) = mac(r[2], k, $modulus.0[2], carry);
                let (r3, carry) = mac(r[3], k, $modulus.0[3], carry);
                let (r4, carry2) = adc(r[4], 0, carry);

                let k = r1.wrapping_mul($inv);
                let (_, carry) = mac(r1, k, $modulus.0[0], 0);
                let (r2, carry) = mac(r2, k, $modulus.0[1], carry);
                let (r3, carry) = mac(r3, k, $modulus.0[2], carry);
                let (r4, carry) = mac(r4, k, $modulus.0[3], carry);
                let (r5, carry2) = adc(r[5], carry2, carry);

                let k = r2.wrapping_mul($inv);
                let (_, carry) = mac(r2, k, $modulus.0[0], 0);
                let (r3, carry) = mac(r3, k, $modulus.0[1], carry);
                let (r4, carry) = mac(r4, k, $modulus.0[2], carry);
                let (r5, carry) = mac(r5, k, $modulus.0[3], carry);
                let (r6, carry2) = adc(r[6], carry2, carry);

                let k = r3.wrapping_mul($inv);
                let (_, carry) = mac(r3, k, $modulus.0[0], 0);
                let (r4, carry) = mac(r4, k, $modulus.0[1], carry);
                let (r5, carry) = mac(r5, k, $modulus.0[2], carry);
                let (r6, carry) = mac(r6, k, $modulus.0[3], carry);
                let (r7, _) = adc(r[7], carry2, carry);

                // Result may be within MODULUS of the correct value
                (&$field([r4, r5, r6, r7])).sub(&$modulus)
            }
        }
    };
    ($field:ident, $modulus:ident, $inv:ident, dense) => {
        impl $field {
            /// Adds `rhs` to `self`, returning the result.
            #[inline]
            pub const fn add(&self, rhs: &Self) -> Self {
                let (d0, carry) = adc(self.0[0], rhs.0[0], 0);
                let (d1, carry) = adc(self.0[1], rhs.0[1], carry);
                let (d2, carry) = adc(self.0[2], rhs.0[2], carry);
                let (d3, carry) = adc(self.0[3], rhs.0[3], carry);

                // Attempt to subtract the modulus, to ensure the value
                // is smaller than the modulus.
                let (d0, borrow) = sbb(d0, $modulus.0[0], 0);
                let (d1, borrow) = sbb(d1, $modulus.0[1], borrow);
                let (d2, borrow) = sbb(d2, $modulus.0[2], borrow);
                let (d3, borrow) = sbb(d3, $modulus.0[3], borrow);
                let (_, borrow) = sbb(carry, 0, borrow);

                let (d0, carry) = adc(d0, $modulus.0[0] & borrow, 0);
                let (d1, carry) = adc(d1, $modulus.0[1] & borrow, carry);
                let (d2, carry) = adc(d2, $modulus.0[2] & borrow, carry);
                let (d3, _) = adc(d3, $modulus.0[3] & borrow, carry);

                $field([d0, d1, d2, d3])
            }

            #[inline(always)]
            pub(crate) const fn montgomery_reduce(r: &[u64; 8]) -> Self {
                // The Montgomery reduction here is based on Algorithm 14.32 in
                // Handbook of Applied Cryptography
                // <http://cacr.uwaterloo.ca/hac/about/chap14.pdf>.

                let k = r[0].wrapping_mul($inv);
                let (_, carry) = mac(r[0], k, $modulus.0[0], 0);
                let (r1, carry) = mac(r[1], k, $modulus.0[1], carry);
                let (r2, carry) = mac(r[2], k, $modulus.0[2], carry);
                let (r3, carry) = mac(r[3], k, $modulus.0[3], carry);
                let (r4, carry2) = adc(r[4], 0, carry);

                let k = r1.wrapping_mul($inv);
                let (_, carry) = mac(r1, k, $modulus.0[0], 0);
                let (r2, carry) = mac(r2, k, $modulus.0[1], carry);
                let (r3, carry) = mac(r3, k, $modulus.0[2], carry);
                let (r4, carry) = mac(r4, k, $modulus.0[3], carry);
                let (r5, carry2) = adc(r[5], carry2, carry);

                let k = r2.wrapping_mul($inv);
                let (_, carry) = mac(r2, k, $modulus.0[0], 0);
                let (r3, carry) = mac(r3, k, $modulus.0[1], carry);
                let (r4, carry) = mac(r4, k, $modulus.0[2], carry);
                let (r5, carry) = mac(r5, k, $modulus.0[3], carry);
                let (r6, carry2) = adc(r[6], carry2, carry);

                let k = r3.wrapping_mul($inv);
                let (_, carry) = mac(r3, k, $modulus.0[0], 0);
                let (r4, carry) = mac(r4, k, $modulus.0[1], carry);
                let (r5, carry) = mac(r5, k, $modulus.0[2], carry);
                let (r6, carry) = mac(r6, k, $modulus.0[3], carry);
                let (r7, carry2) = adc(r[7], carry2, carry);

                // Result may be within MODULUS of the correct value
                let (d0, borrow) = sbb(r4, $modulus.0[0], 0);
                let (d1, borrow) = sbb(r5, $modulus.0[1], borrow);
                let (d2, borrow) = sbb(r6, $modulus.0[2], borrow);
                let (d3, borrow) = sbb(r7, $modulus.0[3], borrow);
                let (_, borrow) = sbb(carry2, 0, borrow);

                let (d0, carry) = adc(d0, $modulus.0[0] & borrow, 0);
                let (d1, carry) = adc(d1, $modulus.0[1] & borrow, carry);
                let (d2, carry) = adc(d2, $modulus.0[2] & borrow, carry);
                let (d3, _) = adc(d3, $modulus.0[3] & borrow, carry);

                $field([d0, d1, d2, d3])
            }
        }
    };
}

#[macro_export]
macro_rules! field_bits {
    // For #[cfg(target_pointer_width = "64")]
    ($field:ident, $modulus:ident) => {
        #[cfg(feature = "bits")]
        #[cfg_attr(docsrs, doc(cfg(feature = "bits")))]
        impl ::ff::PrimeFieldBits for $field {
            type ReprBits = [u64; 4];

            fn to_le_bits(&self) -> ::ff::FieldBits<Self::ReprBits> {
                let bytes = self.to_repr();

                let limbs = [
                    u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
                    u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
                    u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
                    u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
                ];

                ::ff::FieldBits::new(limbs)
            }

            fn char_le_bits() -> ::ff::FieldBits<Self::ReprBits> {
                ::ff::FieldBits::new($modulus.0)
            }
        }
    };
    // For #[cfg(not(target_pointer_width = "64"))]
    ($field:ident, $modulus:ident, $modulus_limbs_32:ident) => {
        #[cfg(feature = "bits")]
        #[cfg_attr(docsrs, doc(cfg(feature = "bits")))]
        impl ::ff::PrimeFieldBits for $field {
            type ReprBits = [u32; 8];

            fn to_le_bits(&self) -> ::ff::FieldBits<Self::ReprBits> {
                let bytes = self.to_repr();

                let limbs = [
                    u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
                    u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
                    u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
                    u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
                    u32::from_le_bytes(bytes[16..20].try_into().unwrap()),
                    u32::from_le_bytes(bytes[20..24].try_into().unwrap()),
                    u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
                    u32::from_le_bytes(bytes[28..32].try_into().unwrap()),
                ];

                ::ff::FieldBits::new(limbs)
            }

            fn char_le_bits() -> ::ff::FieldBits<Self::ReprBits> {
                ::ff::FieldBits::new($modulus_limbs_32)
            }
        }
    };
}

/// A macro to help define serialization and deserialization for prime field implementations
/// that use 32-byte representations. This assumes the concerned type implements PrimeField
/// (for from_repr, to_repr).
#[macro_export]
macro_rules! serialize_deserialize_32_byte_primefield {
    ($type:ty) => {
        impl ::serde::Serialize for $type {
            fn serialize<S: ::serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let bytes = &self.to_repr();
                if serializer.is_human_readable() {
                    hex::serde::serialize(bytes, serializer)
                } else {
                    bytes.serialize(serializer)
                }
            }
        }

        use ::serde::de::Error as _;
        impl<'de> ::serde::Deserialize<'de> for $type {
            fn deserialize<D: ::serde::Deserializer<'de>>(
                deserializer: D,
            ) -> Result<Self, D::Error> {
                let bytes = if deserializer.is_human_readable() {
                    ::hex::serde::deserialize(deserializer)?
                } else {
                    <[u8; 32]>::deserialize(deserializer)?
                };
                Option::from(Self::from_repr(bytes)).ok_or_else(|| {
                    D::Error::custom("deserialized bytes don't encode a valid field element")
                })
            }
        }
    };
}
