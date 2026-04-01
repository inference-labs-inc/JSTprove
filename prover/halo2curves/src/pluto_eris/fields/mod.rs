pub mod fp;
pub mod fp12;
pub mod fp2;
pub mod fp6;
pub mod fq;

#[macro_export]
macro_rules! impl_from_u64_u32_7_limbs {
    ($field:ident, $r2:ident) => {
        impl From<u64> for $field {
            fn from(val: u64) -> $field {
                $field([val, 0, 0, 0, 0, 0, 0]) * $r2
            }
        }
    };

    ($field:ident, $r2:ident) => {
        impl From<u32> for $field {
            fn from(val: u32) -> $field {
                $field([val as u64, 0, 0, 0, 0, 0, 0]) * $r2
            }
        }
    };
}

#[macro_export]
macro_rules! field_common_7_limbs {
    (
        $field:ident,
        $field_repr:ident,
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
        impl $field {
            /// Returns zero, the additive identity.
            #[inline]
            pub const fn zero() -> $field {
                $field([0, 0, 0, 0, 0, 0, 0])
            }

            /// Returns one, the multiplicative identity.
            #[inline]
            pub const fn one() -> $field {
                $r
            }

            // Returns the Jacobi symbol, where the numerator and denominator
            // are the element and the characteristic of the field, respectively.
            // The Jacobi symbol is applicable to odd moduli
            // while the Legendre symbol is applicable to prime moduli.
            // They are equivalent for prime moduli.
            #[inline(always)]
            pub fn jacobi(&self) -> i64 {
                $crate::ff_ext::jacobi::jacobi::<8>(&self.0, &$modulus.0)
            }

            fn from_u512(limbs: [u64; 8]) -> $field {
                let d0 = $field([
                    limbs[0], limbs[1], limbs[2], limbs[3], limbs[4], limbs[5], limbs[6],
                ]);
                let d1 = $field([limbs[7], 0, 0, 0, 0, 0, 0]);

                d0 * $r2 + d1 * $r3
            }

            /// Converts from an integer represented in little endian
            /// into its (congruent) `$field` representation.
            pub const fn from_raw(val: [u64; 7]) -> Self {
                #[cfg(feature = "asm")]
                {
                    let (r0, carry) = mac(0, val[0], $r2.0[0], 0);
                    let (r1, carry) = mac(0, val[0], $r2.0[1], carry);
                    let (r2, carry) = mac(0, val[0], $r2.0[2], carry);
                    let (r3, carry) = mac(0, val[0], $r2.0[3], carry);
                    let (r4, carry) = mac(0, val[0], $r2.0[4], carry);
                    let (r5, carry) = mac(0, val[0], $r2.0[5], carry);
                    let (r6, r7) = mac(0, val[0], $r2.0[6], carry);

                    let (r1, carry) = mac(r1, val[1], $r2.0[0], 0);
                    let (r2, carry) = mac(r2, val[1], $r2.0[1], carry);
                    let (r3, carry) = mac(r3, val[1], $r2.0[2], carry);
                    let (r4, carry) = mac(r4, val[1], $r2.0[3], carry);
                    let (r5, carry) = mac(r5, val[1], $r2.0[4], carry);
                    let (r6, carry) = mac(r6, val[1], $r2.0[5], carry);
                    let (r7, r8) = mac(r7, val[1], $r2.0[6], carry);

                    let (r2, carry) = mac(r2, val[2], $r2.0[0], 0);
                    let (r3, carry) = mac(r3, val[2], $r2.0[1], carry);
                    let (r4, carry) = mac(r4, val[2], $r2.0[2], carry);
                    let (r5, carry) = mac(r5, val[2], $r2.0[3], carry);
                    let (r6, carry) = mac(r6, val[2], $r2.0[4], carry);
                    let (r7, carry) = mac(r7, val[2], $r2.0[5], carry);
                    let (r8, r9) = mac(r8, val[2], $r2.0[6], carry);

                    let (r3, carry) = mac(r3, val[3], $r2.0[0], 0);
                    let (r4, carry) = mac(r4, val[3], $r2.0[1], carry);
                    let (r5, carry) = mac(r5, val[3], $r2.0[2], carry);
                    let (r6, carry) = mac(r6, val[3], $r2.0[3], carry);
                    let (r7, carry) = mac(r7, val[3], $r2.0[4], carry);
                    let (r8, carry) = mac(r8, val[3], $r2.0[5], carry);
                    let (r9, r10) = mac(r9, val[3], $r2.0[6], carry);

                    let (r4, carry) = mac(r4, val[4], $r2.0[0], 0);
                    let (r5, carry) = mac(r5, val[4], $r2.0[1], carry);
                    let (r6, carry) = mac(r6, val[4], $r2.0[2], carry);
                    let (r7, carry) = mac(r7, val[4], $r2.0[3], carry);
                    let (r8, carry) = mac(r8, val[4], $r2.0[4], carry);
                    let (r9, carry) = mac(r9, val[4], $r2.0[5], carry);
                    let (r10, r11) = mac(r10, val[4], $r2.0[6], carry);

                    let (r5, carry) = mac(r5, val[5], $r2.0[0], 0);
                    let (r6, carry) = mac(r6, val[5], $r2.0[1], carry);
                    let (r7, carry) = mac(r7, val[5], $r2.0[2], carry);
                    let (r8, carry) = mac(r8, val[5], $r2.0[3], carry);
                    let (r9, carry) = mac(r9, val[5], $r2.0[4], carry);
                    let (r10, carry) = mac(r10, val[5], $r2.0[5], carry);
                    let (r11, r12) = mac(r11, val[5], $r2.0[6], carry);

                    let (r6, carry) = mac(r6, val[6], $r2.0[0], 0);
                    let (r7, carry) = mac(r7, val[6], $r2.0[1], carry);
                    let (r8, carry) = mac(r8, val[6], $r2.0[2], carry);
                    let (r9, carry) = mac(r9, val[6], $r2.0[3], carry);
                    let (r10, carry) = mac(r10, val[6], $r2.0[4], carry);
                    let (r11, carry) = mac(r11, val[6], $r2.0[5], carry);
                    let (r12, r13) = mac(r12, val[6], $r2.0[6], carry);

                    // Montgomery reduction
                    let k = r0.wrapping_mul($inv);
                    let (_, carry) = mac(r0, k, $modulus.0[0], 0);
                    let (r1, carry) = mac(r1, k, $modulus.0[1], carry);
                    let (r2, carry) = mac(r2, k, $modulus.0[2], carry);
                    let (r3, carry) = mac(r3, k, $modulus.0[3], carry);
                    let (r4, carry) = mac(r4, k, $modulus.0[4], carry);
                    let (r5, carry) = mac(r5, k, $modulus.0[5], carry);
                    let (r6, carry) = mac(r6, k, $modulus.0[6], carry);
                    let (r7, carry2) = adc(r7, 0, carry);

                    let k = r1.wrapping_mul($inv);
                    let (_, carry) = mac(r1, k, $modulus.0[0], 0);
                    let (r2, carry) = mac(r2, k, $modulus.0[1], carry);
                    let (r3, carry) = mac(r3, k, $modulus.0[2], carry);
                    let (r4, carry) = mac(r4, k, $modulus.0[3], carry);
                    let (r5, carry) = mac(r5, k, $modulus.0[4], carry);
                    let (r6, carry) = mac(r6, k, $modulus.0[5], carry);
                    let (r7, carry) = mac(r7, k, $modulus.0[6], carry);
                    let (r8, carry2) = adc(r8, carry2, carry);

                    let k = r2.wrapping_mul($inv);
                    let (_, carry) = mac(r2, k, $modulus.0[0], 0);
                    let (r3, carry) = mac(r3, k, $modulus.0[1], carry);
                    let (r4, carry) = mac(r4, k, $modulus.0[2], carry);
                    let (r5, carry) = mac(r5, k, $modulus.0[3], carry);
                    let (r6, carry) = mac(r6, k, $modulus.0[4], carry);
                    let (r7, carry) = mac(r7, k, $modulus.0[5], carry);
                    let (r8, carry) = mac(r8, k, $modulus.0[6], carry);
                    let (r9, carry2) = adc(r9, carry2, carry);

                    let k = r3.wrapping_mul($inv);
                    let (_, carry) = mac(r3, k, $modulus.0[0], 0);
                    let (r4, carry) = mac(r4, k, $modulus.0[1], carry);
                    let (r5, carry) = mac(r5, k, $modulus.0[2], carry);
                    let (r6, carry) = mac(r6, k, $modulus.0[3], carry);
                    let (r7, carry) = mac(r7, k, $modulus.0[4], carry);
                    let (r8, carry) = mac(r8, k, $modulus.0[5], carry);
                    let (r9, carry) = mac(r9, k, $modulus.0[6], carry);
                    let (r10, carry2) = adc(r10, carry2, carry);

                    let k = r4.wrapping_mul($inv);
                    let (_, carry) = mac(r4, k, $modulus.0[0], 0);
                    let (r5, carry) = mac(r5, k, $modulus.0[1], carry);
                    let (r6, carry) = mac(r6, k, $modulus.0[2], carry);
                    let (r7, carry) = mac(r7, k, $modulus.0[3], carry);
                    let (r8, carry) = mac(r8, k, $modulus.0[4], carry);
                    let (r9, carry) = mac(r9, k, $modulus.0[5], carry);
                    let (r10, carry) = mac(r10, k, $modulus.0[6], carry);
                    let (r11, carry2) = adc(r11, carry2, carry);

                    let k = r5.wrapping_mul($inv);
                    let (_, carry) = mac(r5, k, $modulus.0[0], 0);
                    let (r6, carry) = mac(r6, k, $modulus.0[1], carry);
                    let (r7, carry) = mac(r7, k, $modulus.0[2], carry);
                    let (r8, carry) = mac(r8, k, $modulus.0[3], carry);
                    let (r9, carry) = mac(r9, k, $modulus.0[4], carry);
                    let (r10, carry) = mac(r10, k, $modulus.0[5], carry);
                    let (r11, carry) = mac(r11, k, $modulus.0[6], carry);
                    let (r12, carry2) = adc(r12, carry2, carry);

                    let k = r6.wrapping_mul($inv);
                    let (_, carry) = mac(r6, k, $modulus.0[0], 0);
                    let (r7, carry) = mac(r7, k, $modulus.0[1], carry);
                    let (r8, carry) = mac(r8, k, $modulus.0[2], carry);
                    let (r9, carry) = mac(r9, k, $modulus.0[3], carry);
                    let (r10, carry) = mac(r10, k, $modulus.0[4], carry);
                    let (r11, carry) = mac(r11, k, $modulus.0[5], carry);
                    let (r12, carry) = mac(r12, k, $modulus.0[6], carry);
                    let (r13, carry2) = adc(r13, carry2, carry);

                    // Result may be within MODULUS of the correct value
                    let (d0, borrow) = sbb(r7, $modulus.0[0], 0);
                    let (d1, borrow) = sbb(r8, $modulus.0[1], borrow);
                    let (d2, borrow) = sbb(r9, $modulus.0[2], borrow);
                    let (d3, borrow) = sbb(r10, $modulus.0[3], borrow);
                    let (d4, borrow) = sbb(r11, $modulus.0[4], borrow);
                    let (d5, borrow) = sbb(r12, $modulus.0[5], borrow);
                    let (d6, borrow) = sbb(r13, $modulus.0[6], borrow);
                    let (_, borrow) = sbb(carry2, 0, borrow);
                    let (d0, carry) = adc(d0, $modulus.0[0] & borrow, 0);
                    let (d1, carry) = adc(d1, $modulus.0[1] & borrow, carry);
                    let (d2, carry) = adc(d2, $modulus.0[2] & borrow, carry);
                    let (d3, carry) = adc(d3, $modulus.0[3] & borrow, carry);
                    let (d4, carry) = adc(d4, $modulus.0[4] & borrow, carry);
                    let (d5, carry) = adc(d5, $modulus.0[5] & borrow, carry);
                    let (d6, _) = adc(d6, $modulus.0[6] & borrow, carry);

                    $field([d0, d1, d2, d3, d4, d5, d6])
                }
                #[cfg(not(feature = "asm"))]
                {
                    (&$field(val)).mul(&$r2)
                }
            }

            /// Attempts to convert a little-endian byte representation of
            /// a scalar into a `Fr`, failing if the input is not canonical.
            pub fn from_bytes(bytes: &[u8; 56]) -> CtOption<$field> {
                <Self as ff::PrimeField>::from_repr($field_repr { repr: *bytes })
            }

            /// Converts an element of `Fr` into a byte representation in
            /// little-endian byte order.
            pub fn to_bytes(&self) -> [u8; 56] {
                <Self as ff::PrimeField>::to_repr(self).repr
            }

            /// Lexicographic comparison of Montgomery forms.
            #[inline(always)]
            const fn is_less_than(x: &[u64; 7], y: &[u64; 7]) -> bool {
                let (_, borrow) = sbb(x[0], y[0], 0);
                let (_, borrow) = sbb(x[1], y[1], borrow);
                let (_, borrow) = sbb(x[2], y[2], borrow);
                let (_, borrow) = sbb(x[3], y[3], borrow);
                let (_, borrow) = sbb(x[4], y[4], borrow);
                let (_, borrow) = sbb(x[5], y[5], borrow);
                let (_, borrow) = sbb(x[6], y[6], borrow);
                borrow >> 63 == 1
            }
        }

        impl fmt::Debug for $field {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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
                    & self.0[4].ct_eq(&other.0[4])
                    & self.0[5].ct_eq(&other.0[5])
                    & self.0[6].ct_eq(&other.0[6])
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
                    u64::conditional_select(&a.0[4], &b.0[4], choice),
                    u64::conditional_select(&a.0[5], &b.0[5], choice),
                    u64::conditional_select(&a.0[6], &b.0[6], choice),
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

        impl From<$field> for [u8; 56] {
            fn from(value: $field) -> [u8; 56] {
                value.to_repr().repr
            }
        }

        impl<'a> From<&'a $field> for [u8; 56] {
            fn from(value: &'a $field) -> [u8; 56] {
                value.to_repr().repr
            }
        }

        impl $crate::serde::SerdeObject for $field {
            fn from_raw_bytes_unchecked(bytes: &[u8]) -> Self {
                debug_assert_eq!(bytes.len(), 56);
                let inner = [0, 8, 16, 24, 32, 40, 48]
                    .map(|i| u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap()));
                Self(inner)
            }
            fn from_raw_bytes(bytes: &[u8]) -> Option<Self> {
                if bytes.len() != 56 {
                    return None;
                }
                let elt = Self::from_raw_bytes_unchecked(bytes);
                Self::is_less_than(&elt.0, &$modulus.0).then(|| elt)
            }
            fn to_raw_bytes(&self) -> Vec<u8> {
                let mut res = Vec::with_capacity(56);
                for limb in self.0.iter() {
                    res.extend_from_slice(&limb.to_le_bytes());
                }
                res
            }
            fn read_raw_unchecked<R: std::io::Read>(reader: &mut R) -> Self {
                let inner = [(); 7].map(|_| {
                    let mut buf = [0; 8];
                    reader.read_exact(&mut buf).unwrap();
                    u64::from_le_bytes(buf)
                });
                Self(inner)
            }
            fn read_raw<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
                let mut inner = [0u64; 7];
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
macro_rules! field_arithmetic_7_limbs {
    ($field:ident, $modulus:ident, $inv:ident, $field_type:ident) => {
        $crate::field_specific_7_limbs!($field, $modulus, $inv, $field_type);
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
                let (r3, carry) = mac(0, self.0[0], self.0[3], carry);
                let (r4, carry) = mac(0, self.0[0], self.0[4], carry);
                let (r5, carry) = mac(0, self.0[0], self.0[5], carry);
                let (r6, r7) = mac(0, self.0[0], self.0[6], carry);

                let (r3, carry) = mac(r3, self.0[1], self.0[2], 0);
                let (r4, carry) = mac(r4, self.0[1], self.0[3], carry);
                let (r5, carry) = mac(r5, self.0[1], self.0[4], carry);
                let (r6, carry) = mac(r6, self.0[1], self.0[5], carry);
                let (r7, r8) = mac(r7, self.0[1], self.0[6], carry);

                let (r5, carry) = mac(r5, self.0[2], self.0[3], 0);
                let (r6, carry) = mac(r6, self.0[2], self.0[4], carry);
                let (r7, carry) = mac(r7, self.0[2], self.0[5], carry);
                let (r8, r9) = mac(r8, self.0[2], self.0[6], carry);

                let (r7, carry) = mac(r7, self.0[3], self.0[4], 0);
                let (r8, carry) = mac(r8, self.0[3], self.0[5], carry);
                let (r9, r10) = mac(r9, self.0[3], self.0[6], carry);

                let (r9, carry) = mac(r9, self.0[4], self.0[5], 0);
                let (r10, r11) = mac(r10, self.0[4], self.0[6], carry);

                let (r11, r12) = mac(r11, self.0[5], self.0[6], 0);

                let r13 = r12 >> 63;
                let r12 = (r12 << 1) | (r11 >> 63);
                let r11 = (r11 << 1) | (r10 >> 63);
                let r10 = (r10 << 1) | (r9 >> 63);
                let r9 = (r9 << 1) | (r8 >> 63);
                let r8 = (r8 << 1) | (r7 >> 63);
                let r7 = (r7 << 1) | (r6 >> 63);
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
                let (r7, carry) = adc(0, r7, carry);
                let (r8, carry) = mac(r8, self.0[4], self.0[4], carry);
                let (r9, carry) = adc(0, r9, carry);
                let (r10, carry) = mac(r10, self.0[5], self.0[5], carry);
                let (r11, carry) = adc(0, r11, carry);
                let (r12, carry) = mac(r12, self.0[6], self.0[6], carry);
                let (r13, _) = adc(0, r13, carry);

                $field::montgomery_reduce(&[
                    r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13,
                ])
            }

            /// Multiplies `rhs` by `self`, returning the result.
            #[inline]
            pub const fn mul(&self, rhs: &Self) -> $field {
                // Schoolbook multiplication
                let (r0, carry) = mac(0, self.0[0], rhs.0[0], 0);
                let (r1, carry) = mac(0, self.0[0], rhs.0[1], carry);
                let (r2, carry) = mac(0, self.0[0], rhs.0[2], carry);
                let (r3, carry) = mac(0, self.0[0], rhs.0[3], carry);
                let (r4, carry) = mac(0, self.0[0], rhs.0[4], carry);
                let (r5, carry) = mac(0, self.0[0], rhs.0[5], carry);
                let (r6, r7) = mac(0, self.0[0], rhs.0[6], carry);

                let (r1, carry) = mac(r1, self.0[1], rhs.0[0], 0);
                let (r2, carry) = mac(r2, self.0[1], rhs.0[1], carry);
                let (r3, carry) = mac(r3, self.0[1], rhs.0[2], carry);
                let (r4, carry) = mac(r4, self.0[1], rhs.0[3], carry);
                let (r5, carry) = mac(r5, self.0[1], rhs.0[4], carry);
                let (r6, carry) = mac(r6, self.0[1], rhs.0[5], carry);
                let (r7, r8) = mac(r7, self.0[1], rhs.0[6], carry);

                let (r2, carry) = mac(r2, self.0[2], rhs.0[0], 0);
                let (r3, carry) = mac(r3, self.0[2], rhs.0[1], carry);
                let (r4, carry) = mac(r4, self.0[2], rhs.0[2], carry);
                let (r5, carry) = mac(r5, self.0[2], rhs.0[3], carry);
                let (r6, carry) = mac(r6, self.0[2], rhs.0[4], carry);
                let (r7, carry) = mac(r7, self.0[2], rhs.0[5], carry);
                let (r8, r9) = mac(r8, self.0[2], rhs.0[6], carry);

                let (r3, carry) = mac(r3, self.0[3], rhs.0[0], 0);
                let (r4, carry) = mac(r4, self.0[3], rhs.0[1], carry);
                let (r5, carry) = mac(r5, self.0[3], rhs.0[2], carry);
                let (r6, carry) = mac(r6, self.0[3], rhs.0[3], carry);
                let (r7, carry) = mac(r7, self.0[3], rhs.0[4], carry);
                let (r8, carry) = mac(r8, self.0[3], rhs.0[5], carry);
                let (r9, r10) = mac(r9, self.0[3], rhs.0[6], carry);

                let (r4, carry) = mac(r4, self.0[4], rhs.0[0], 0);
                let (r5, carry) = mac(r5, self.0[4], rhs.0[1], carry);
                let (r6, carry) = mac(r6, self.0[4], rhs.0[2], carry);
                let (r7, carry) = mac(r7, self.0[4], rhs.0[3], carry);
                let (r8, carry) = mac(r8, self.0[4], rhs.0[4], carry);
                let (r9, carry) = mac(r9, self.0[4], rhs.0[5], carry);
                let (r10, r11) = mac(r10, self.0[4], rhs.0[6], carry);

                let (r5, carry) = mac(r5, self.0[5], rhs.0[0], 0);
                let (r6, carry) = mac(r6, self.0[5], rhs.0[1], carry);
                let (r7, carry) = mac(r7, self.0[5], rhs.0[2], carry);
                let (r8, carry) = mac(r8, self.0[5], rhs.0[3], carry);
                let (r9, carry) = mac(r9, self.0[5], rhs.0[4], carry);
                let (r10, carry) = mac(r10, self.0[5], rhs.0[5], carry);
                let (r11, r12) = mac(r11, self.0[5], rhs.0[6], carry);

                let (r6, carry) = mac(r6, self.0[6], rhs.0[0], 0);
                let (r7, carry) = mac(r7, self.0[6], rhs.0[1], carry);
                let (r8, carry) = mac(r8, self.0[6], rhs.0[2], carry);
                let (r9, carry) = mac(r9, self.0[6], rhs.0[3], carry);
                let (r10, carry) = mac(r10, self.0[6], rhs.0[4], carry);
                let (r11, carry) = mac(r11, self.0[6], rhs.0[5], carry);
                let (r12, r13) = mac(r12, self.0[6], rhs.0[6], carry);

                $field::montgomery_reduce(&[
                    r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13,
                ])
            }

            /// Subtracts `rhs` from `self`, returning the result.
            #[inline]
            pub const fn sub(&self, rhs: &Self) -> Self {
                let (d0, borrow) = sbb(self.0[0], rhs.0[0], 0);
                let (d1, borrow) = sbb(self.0[1], rhs.0[1], borrow);
                let (d2, borrow) = sbb(self.0[2], rhs.0[2], borrow);
                let (d3, borrow) = sbb(self.0[3], rhs.0[3], borrow);
                let (d4, borrow) = sbb(self.0[4], rhs.0[4], borrow);
                let (d5, borrow) = sbb(self.0[5], rhs.0[5], borrow);
                let (d6, borrow) = sbb(self.0[6], rhs.0[6], borrow);

                // If underflow occurred on the final limb, borrow = 0xfff...fff, otherwise
                // borrow = 0x000...000. Thus, we use it as a mask to conditionally add the modulus.
                let (d0, carry) = adc(d0, $modulus.0[0] & borrow, 0);
                let (d1, carry) = adc(d1, $modulus.0[1] & borrow, carry);
                let (d2, carry) = adc(d2, $modulus.0[2] & borrow, carry);
                let (d3, carry) = adc(d3, $modulus.0[3] & borrow, carry);
                let (d4, carry) = adc(d4, $modulus.0[4] & borrow, carry);
                let (d5, carry) = adc(d5, $modulus.0[5] & borrow, carry);
                let (d6, _) = adc(d6, $modulus.0[6] & borrow, carry);

                $field([d0, d1, d2, d3, d4, d5, d6])
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
                let (d3, borrow) = sbb($modulus.0[3], self.0[3], borrow);
                let (d4, borrow) = sbb($modulus.0[4], self.0[4], borrow);
                let (d5, borrow) = sbb($modulus.0[5], self.0[5], borrow);
                let (d6, _) = sbb($modulus.0[6], self.0[6], borrow);

                // `tmp` could be `MODULUS` if `self` was zero. Create a mask that is
                // zero if `self` was zero, and `u64::max_value()` if self was nonzero.
                let mask = (((self.0[0]
                    | self.0[1]
                    | self.0[2]
                    | self.0[3]
                    | self.0[4]
                    | self.0[5]
                    | self.0[6])
                    == 0) as u64)
                    .wrapping_sub(1);

                $field([
                    d0 & mask,
                    d1 & mask,
                    d2 & mask,
                    d3 & mask,
                    d4 & mask,
                    d5 & mask,
                    d6 & mask,
                ])
            }
        }
    };
}

#[macro_export]
macro_rules! field_specific_7_limbs {
    ($field:ident, $modulus:ident, $inv:ident, sparse) => {
        impl $field {
            /// Adds `rhs` to `self`, returning the result.
            #[inline]
            pub const fn add(&self, rhs: &Self) -> Self {
                let (d0, carry) = adc(self.0[0], rhs.0[0], 0);
                let (d1, carry) = adc(self.0[1], rhs.0[1], carry);
                let (d2, carry) = adc(self.0[2], rhs.0[2], carry);
                let (d3, carry) = adc(self.0[3], rhs.0[3], carry);
                let (d4, carry) = adc(self.0[4], rhs.0[4], carry);
                let (d5, carry) = adc(self.0[5], rhs.0[5], carry);
                let (d6, _) = adc(self.0[6], rhs.0[6], carry);

                // Attempt to subtract the modulus, to ensure the value
                // is smaller than the modulus.
                (&$field([d0, d1, d2, d3, d4, d5, d6])).sub(&$modulus)
            }

            #[inline(always)]
            pub(crate) const fn montgomery_reduce(r: &[u64; 14]) -> $field {
                // The Montgomery reduction here is based on Algorithm 14.32 in
                // Handbook of Applied Cryptography
                // <http://cacr.uwaterloo.ca/hac/about/chap14.pdf>.

                let k = r[0].wrapping_mul($inv);
                let (_, carry) = mac(r[0], k, $modulus.0[0], 0);
                let (r1, carry) = mac(r[1], k, $modulus.0[1], carry);
                let (r2, carry) = mac(r[2], k, $modulus.0[2], carry);
                let (r3, carry) = mac(r[3], k, $modulus.0[3], carry);
                let (r4, carry) = mac(r[4], k, $modulus.0[4], carry);
                let (r5, carry) = mac(r[5], k, $modulus.0[5], carry);
                let (r6, carry) = mac(r[6], k, $modulus.0[6], carry);
                let (r7, carry2) = adc(r[7], 0, carry);

                let k = r1.wrapping_mul($inv);
                let (_, carry) = mac(r1, k, $modulus.0[0], 0);
                let (r2, carry) = mac(r2, k, $modulus.0[1], carry);
                let (r3, carry) = mac(r3, k, $modulus.0[2], carry);
                let (r4, carry) = mac(r4, k, $modulus.0[3], carry);
                let (r5, carry) = mac(r5, k, $modulus.0[4], carry);
                let (r6, carry) = mac(r6, k, $modulus.0[5], carry);
                let (r7, carry) = mac(r7, k, $modulus.0[6], carry);
                let (r8, carry2) = adc(r[8], carry2, carry);

                let k = r2.wrapping_mul($inv);
                let (_, carry) = mac(r2, k, $modulus.0[0], 0);
                let (r3, carry) = mac(r3, k, $modulus.0[1], carry);
                let (r4, carry) = mac(r4, k, $modulus.0[2], carry);
                let (r5, carry) = mac(r5, k, $modulus.0[3], carry);
                let (r6, carry) = mac(r6, k, $modulus.0[4], carry);
                let (r7, carry) = mac(r7, k, $modulus.0[5], carry);
                let (r8, carry) = mac(r8, k, $modulus.0[6], carry);
                let (r9, carry2) = adc(r[9], carry2, carry);

                let k = r3.wrapping_mul($inv);
                let (_, carry) = mac(r3, k, $modulus.0[0], 0);
                let (r4, carry) = mac(r4, k, $modulus.0[1], carry);
                let (r5, carry) = mac(r5, k, $modulus.0[2], carry);
                let (r6, carry) = mac(r6, k, $modulus.0[3], carry);
                let (r7, carry) = mac(r7, k, $modulus.0[4], carry);
                let (r8, carry) = mac(r8, k, $modulus.0[5], carry);
                let (r9, carry) = mac(r9, k, $modulus.0[6], carry);
                let (r10, carry2) = adc(r[10], carry2, carry);

                let k = r4.wrapping_mul($inv);
                let (_, carry) = mac(r4, k, $modulus.0[0], 0);
                let (r5, carry) = mac(r5, k, $modulus.0[1], carry);
                let (r6, carry) = mac(r6, k, $modulus.0[2], carry);
                let (r7, carry) = mac(r7, k, $modulus.0[3], carry);
                let (r8, carry) = mac(r8, k, $modulus.0[4], carry);
                let (r9, carry) = mac(r9, k, $modulus.0[5], carry);
                let (r10, carry) = mac(r10, k, $modulus.0[6], carry);
                let (r11, carry2) = adc(r[11], carry2, carry);

                let k = r5.wrapping_mul($inv);
                let (_, carry) = mac(r5, k, $modulus.0[0], 0);
                let (r6, carry) = mac(r6, k, $modulus.0[1], carry);
                let (r7, carry) = mac(r7, k, $modulus.0[2], carry);
                let (r8, carry) = mac(r8, k, $modulus.0[3], carry);
                let (r9, carry) = mac(r9, k, $modulus.0[4], carry);
                let (r10, carry) = mac(r10, k, $modulus.0[5], carry);
                let (r11, carry) = mac(r11, k, $modulus.0[6], carry);
                let (r12, carry2) = adc(r[12], carry2, carry);

                let k = r6.wrapping_mul($inv);
                let (_, carry) = mac(r6, k, $modulus.0[0], 0);
                let (r7, carry) = mac(r7, k, $modulus.0[1], carry);
                let (r8, carry) = mac(r8, k, $modulus.0[2], carry);
                let (r9, carry) = mac(r9, k, $modulus.0[3], carry);
                let (r10, carry) = mac(r10, k, $modulus.0[4], carry);
                let (r11, carry) = mac(r11, k, $modulus.0[5], carry);
                let (r12, carry) = mac(r12, k, $modulus.0[6], carry);
                let (r13, _) = adc(r[13], carry2, carry);
                // Result may be within MODULUS of the correct value
                (&$field([r7, r8, r9, r10, r11, r12, r13])).sub(&$modulus)
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
                let (d4, carry) = adc(self.0[4], rhs.0[4], carry);
                let (d5, carry) = adc(self.0[5], rhs.0[5], carry);
                let (d6, carry) = adc(self.0[6], rhs.0[6], carry);

                // Attempt to subtract the modulus, to ensure the value
                // is smaller than the modulus.
                let (d0, borrow) = sbb(d0, $modulus.0[0], 0);
                let (d1, borrow) = sbb(d1, $modulus.0[1], borrow);
                let (d2, borrow) = sbb(d2, $modulus.0[2], borrow);
                let (d3, borrow) = sbb(d3, $modulus.0[3], borrow);
                let (d4, borrow) = sbb(d4, $modulus.0[4], borrow);
                let (d5, borrow) = sbb(d5, $modulus.0[5], borrow);
                let (_, borrow) = sbb(carry, 0, borrow);

                let (d0, carry) = adc(d0, $modulus.0[0] & borrow, 0);
                let (d1, carry) = adc(d1, $modulus.0[1] & borrow, carry);
                let (d2, carry) = adc(d2, $modulus.0[2] & borrow, carry);
                let (d3, carry) = adc(d3, $modulus.0[3] & borrow, carry);
                let (d4, carry) = adc(d4, $modulus.0[4] & borrow, carry);
                let (d5, carry) = adc(d5, $modulus.0[5] & borrow, carry);
                let (d6, _) = adc(d6, $modulus.0[6] & borrow, carry);

                $field([d0, d1, d2, d3, d4, d5, d6])
            }
        }
    };
}

#[macro_export]
macro_rules! field_bits_7_limbs {
    // For #[cfg(target_pointer_width = "64")]
    ($field:ident, $modulus:ident) => {
        #[cfg(feature = "bits")]
        #[cfg_attr(docsrs, doc(cfg(feature = "bits")))]
        impl ::ff::PrimeFieldBits for $field {
            type ReprBits = [u64; 7];

            fn to_le_bits(&self) -> ::ff::FieldBits<Self::ReprBits> {
                let bytes = self.to_repr().repr;

                let limbs = [
                    u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
                    u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
                    u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
                    u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
                    u64::from_le_bytes(bytes[32..40].try_into().unwrap()),
                    u64::from_le_bytes(bytes[40..48].try_into().unwrap()),
                    u64::from_le_bytes(bytes[48..56].try_into().unwrap()),
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
            type ReprBits = [u32; 14];

            fn to_le_bits(&self) -> ::ff::FieldBits<Self::ReprBits> {
                let bytes = self.to_repr().repr;

                let limbs = [
                    u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
                    u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
                    u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
                    u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
                    u32::from_le_bytes(bytes[16..20].try_into().unwrap()),
                    u32::from_le_bytes(bytes[20..24].try_into().unwrap()),
                    u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
                    u32::from_le_bytes(bytes[28..32].try_into().unwrap()),
                    u32::from_le_bytes(bytes[32..36].try_into().unwrap()),
                    u32::from_le_bytes(bytes[36..40].try_into().unwrap()),
                    u32::from_le_bytes(bytes[40..44].try_into().unwrap()),
                    u32::from_le_bytes(bytes[44..48].try_into().unwrap()),
                    u32::from_le_bytes(bytes[48..52].try_into().unwrap()),
                    u32::from_le_bytes(bytes[52..56].try_into().unwrap()),
                ];

                ::ff::FieldBits::new(limbs)
            }

            fn char_le_bits() -> ::ff::FieldBits<Self::ReprBits> {
                ::ff::FieldBits::new($modulus_limbs_32)
            }
        }
    };
}
