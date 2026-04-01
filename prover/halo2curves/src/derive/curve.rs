#[macro_export]
macro_rules! endo {
    ($name:ident, $field:ident, $params:expr) => {
        impl CurveEndo for $name {
            fn decompose_scalar(k: &$field) -> (u128, bool, u128, bool) {
                let to_limbs = |e: &$field| {
                    let repr = e.to_repr();
                    let repr = repr.as_ref();
                    let tmp0 = u64::from_le_bytes(repr[0..8].try_into().unwrap());
                    let tmp1 = u64::from_le_bytes(repr[8..16].try_into().unwrap());
                    let tmp2 = u64::from_le_bytes(repr[16..24].try_into().unwrap());
                    let tmp3 = u64::from_le_bytes(repr[24..32].try_into().unwrap());
                    [tmp0, tmp1, tmp2, tmp3]
                };

                let get_lower_128 = |e: &$field| {
                    let e = to_limbs(e);
                    u128::from(e[0]) | (u128::from(e[1]) << 64)
                };

                let is_neg = |e: &$field| {
                    let e = to_limbs(e);
                    let (_, borrow) = sbb(0xffffffffffffffff, e[0], 0);
                    let (_, borrow) = sbb(0xffffffffffffffff, e[1], borrow);
                    let (_, borrow) = sbb(0xffffffffffffffff, e[2], borrow);
                    let (_, borrow) = sbb(0x00, e[3], borrow);
                    borrow & 1 != 0
                };

                let input = to_limbs(&k);
                let c1 = mul_512($params.gamma2, input);
                let c2 = mul_512($params.gamma1, input);
                let c1 = [c1[4], c1[5], c1[6], c1[7]];
                let c2 = [c2[4], c2[5], c2[6], c2[7]];
                let q1 = mul_512(c1, $params.b1);
                let q2 = mul_512(c2, $params.b2);
                let q1 = $field::from_raw([q1[0], q1[1], q1[2], q1[3]]);
                let q2 = $field::from_raw([q2[0], q2[1], q2[2], q2[3]]);
                let k2 = q2 - q1;
                let k1 = k + k2 * $field::ZETA;
                let k1_neg = is_neg(&k1);
                let k2_neg = is_neg(&k2);
                let k1 = if k1_neg { -k1 } else { k1 };
                let k2 = if k2_neg { -k2 } else { k2 };

                (get_lower_128(&k1), k1_neg, get_lower_128(&k2), k2_neg)
            }
        }
    };
}

// Sign mask for 0, 1 and 2 spare bits.
pub(crate) const SIGN_MASK: u8 = 0b1000_0000;
pub(crate) const SIGN_SHIFT: u8 = 7;
// Identity mask for 0 and 2 spare bits (1 spare bit does not use it).
pub(crate) const IDENTITY_MASK: u8 = 0b0100_0000;
pub(crate) const IDENTITY_SHIFT: u8 = 6;

#[macro_export]
macro_rules! new_curve_impl {
    (($($privacy:tt)*),
    $name:ident,
    $name_affine:ident,
    $base:ident,
    $scalar:ident,
    $generator:expr,
    $constant_a:expr,
    $constant_b:expr,
    $curve_id:literal,
    $hash_to_curve:expr,
    ) => {


        // **Compressed formats**:
        // The encoding of the x-coordinate can be Little Endian or Big Endian (inherited from the
        // field encoding).
        // The bit flags appear in the MSB of the encoded x-coordinate in the 1 and 2 Spare bits
        // case, and in an extra byte after the encoded x-coordinate in the 0 Spare bits case.
        // `BS` is the base size: the number of bytes required to encode a coordinate.
        //
        // According to the number of spare bits.
        // 1 Spare bit:
        //
        //     |                  | sign      | x-coordinate |
        //     | Byte pos. (LE)   | BS-1          ..       0 |
        //     | Byte pos. (BE)   | 0          ..       BS-1 |
        //     | Bit pos.         | 7         |              |
        //     | ---------------- | --------  | ------------ |
        //     | Identity         | 0         | 0            |
        //     | Non-identity $P$ | $sgn0(P)$ | $P.x$        |
        //
        // ---
        // 2 Spare bits:
        //     |                  | sign      | ident    | x-coordinate |
        //     | Byte pos. (LE)   | BS-1                     ..       0 |
        //     | Byte pos. (BE)   | 0                     ..       BS-1 |
        //     | Bit pos.         | 7         | 6        |              |
        //     | ---------------- | --------  | -------- | --------     |
        //     | Identity         | 0         | 1        | 0            |
        //     | Non-identity $P$ | $sgn0(P)$ | 0        | $P.x$        |
        //
        // ---
        // 0 Spare bits:
        //     Add an extra byte after the compressed x-coordinate to hold the flags. Then follow
        //     the 2 spare bit flag format.
        //
        //     |                  | sign      | ident    | 000000 | x-coordinate |
        //     | Byte pos. (LE)   | BS                            | BS-1  ..   0 |
        //     | Byte pos. (BE)   | BS                            | 0  ..   BS-1 |
        //     | Bit pos.         | 7         | 6        | 5..0   |              |
        //     | ---------------- | --------- | -------- | ------ | ------------ |
        //     | Identity         | 0         | 1        | 000000 | 0            |
        //     | Non-identity $P$ | $sgn0(P)$ | 0        | 000000 | $P.x$        |
        //
        macro_rules! impl_compressed {
            ($spare_bits: expr) => {
                paste::paste! {

                // The compressed size is the size of the x-coordinate (one base field element)
                // when there is at least 1 spare bit. When there is no spare bits (secp256k1)
                // the size is increased by 1 byte.
                #[allow(non_upper_case_globals)]
                const [< $name _COMPRESSED_SIZE >]: usize =
                    if $spare_bits == 0 {
                        $base::size() + 1
                    } else {
                        $base::size()
                    };

                #[derive(Copy, Clone, PartialEq, Eq)]
                #[cfg_attr(feature = "derive_serde", derive(Serialize, Deserialize))]
                pub struct [<$name Compressed >](
                    #[cfg_attr(feature = "derive_serde", serde(with = "serde_arrays"))]
                    [u8; [< $name _COMPRESSED_SIZE >]]
                );

                impl std::fmt::Debug for [< $name Compressed >] {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        self.0[..].fmt(f)
                    }
                }

                impl Default for [< $name Compressed >] {
                    fn default() -> Self {
                        [< $name Compressed >]([0; [< $name _COMPRESSED_SIZE >]])
                    }
                }

                impl AsRef<[u8]> for [< $name Compressed >] {
                    fn as_ref(&self) -> &[u8] {
                        &self.0
                    }
                }

                impl AsMut<[u8]> for [< $name Compressed >] {
                    fn as_mut(&mut self) -> &mut [u8] {
                        &mut self.0
                    }
                }

                impl GroupEncoding for $name {
                    type Repr = [< $name Compressed >];

                    fn from_bytes(bytes: &Self::Repr) -> CtOption<Self> {
                        $name_affine::from_bytes(bytes).map(Self::from)
                    }

                    fn from_bytes_unchecked(bytes: &Self::Repr) -> CtOption<Self> {
                        $name_affine::from_bytes_unchecked(bytes).map(Self::from)
                    }

                    fn to_bytes(&self) -> Self::Repr {
                        $name_affine::from(self).to_bytes()
                    }
                }


                // The flags are placed in the last byte (the most significant byte).
                #[allow(non_upper_case_globals)]
                const [< $name _FLAG_BYTE_INDEX>]: usize= [< $name _COMPRESSED_SIZE >]-1 ;

                #[allow(non_upper_case_globals)]
                const [< $name _FLAG_BITS >]: u8 =
                if $spare_bits == 1 {
                    0b1000_0000
                } else if $spare_bits == 2 {
                    0b1100_0000
                } else {
                    //$spare_bits == 0
                    0b1111_1111
                };

                impl group::GroupEncoding for $name_affine {
                    type Repr = [< $name Compressed >];


                    fn from_bytes(bytes: &Self::Repr) -> CtOption<Self> {
                        let mut tmp = bytes.0;

                        let flag_byte = tmp[[< $name _FLAG_BYTE_INDEX>]];
                        // Get identity and sign flags.
                        let identity_flag = if $spare_bits == 0  || $spare_bits == 2 {
                            Choice::from((flag_byte & IDENTITY_MASK) >> IDENTITY_SHIFT )
                        } else {
                            Choice::from(0u8)
                        };

                        let sign_flag = Choice::from( (flag_byte  & SIGN_MASK) >> SIGN_SHIFT );

                        let extra_bits = if $spare_bits == 0 {
                            // In the case of 0 spare bits, an extra byte is added to hold the flags.
                            // In this byte, the 6 least significant bits must be set to 0.
                            Choice::from(u8::from((flag_byte & 0b0011_1111) !=0) )
                        } else {
                            // There are no extra bits in the rest of cases.
                            Choice::from(0u8)
                        };

                        // Clear flag bits
                        tmp[[< $name _FLAG_BYTE_INDEX>]] &= ![< $name _FLAG_BITS >];

                        // Get x-coordinate
                        let mut xbytes = [0u8; $base::size()];
                        xbytes.copy_from_slice(&tmp[..$base::size()]);



                        $base::from_bytes(&xbytes).and_then(|x| {

                            // Decide if the point is the identity and the validity of the encoding.
                            let (is_valid, is_identity) =
                            if $spare_bits   == 0  || $spare_bits == 2 {
                                // Correct encoding follows one of the following:
                                // 1. Identity:
                                //  identity_flag = 1, sign = 0, x = 0, extra_bits = 0
                                // 2. Non-identity:
                                //  identity_flag = 0, , extra_bits = 0
                                //
                                // is_valid = !identity_flag \/ (!sign /\ x.is_zero()) /\ !extra_bits
                                ( (!identity_flag | (!sign_flag & x.is_zero())) & !extra_bits, identity_flag)
                            } else {
                                // Correct encoding follows one of the following:
                                // 1. Identity:
                                //  sign = 0, x = 0
                                // 2. Non-identity:
                                //  x!=0
                                //
                                // is_valid = !(x.is_zero() /\ sign_flag)
                                ( !(x.is_zero() & sign_flag), x.is_zero())
                            };


                            CtOption::new(
                                Self::identity(),
                                is_identity)
                            .or_else(|| {
                                // Computes corresponding y coordinate.
                                $name_affine::y2(x).sqrt().and_then(|y| {
                                    // Get sign of obtained solution. sign = y % 2.
                                    let sign = Choice::from(y.to_bytes()[0] & 1);
                                    // Adjust sign if necessary.
                                    let y = $base::conditional_select(&y, &-y, sign_flag ^ sign);
                                    CtOption::new(
                                        $name_affine {
                                            x,
                                            y,
                                        },
                                        is_valid,
                                    )
                                })
                            })
                        })
                    }

                    fn from_bytes_unchecked(bytes: &Self::Repr) -> CtOption<Self> {
                        // In compressed form we cannot skip the curve check.
                        Self::from_bytes(bytes)
                    }

                    fn to_bytes(&self) -> Self::Repr {
                        let mut res = [0; [< $name _COMPRESSED_SIZE >]];

                        let x_bytes = $base::conditional_select(&self.x, &$base::zero(), self.is_identity()).to_bytes();
                        res[..$base::size()].copy_from_slice(&x_bytes);

                        // Set identity flag if necessary.
                        res[ [< $name _FLAG_BYTE_INDEX>]] |= u8::conditional_select(&0u8, &IDENTITY_MASK, self.is_identity());

                        // Set sign flag if point is not identity, and has negative sign.
                        res[ [< $name _FLAG_BYTE_INDEX>]] |= u8::conditional_select(&0u8, &SIGN_MASK, !self.is_identity() & Choice::from(self.y.to_bytes()[0] & 1));
                        [< $name Compressed >](res)
                    }
                }


                }
            };
        }


        // **Uncompressed format**
        // The encoding of the x-coordinate and y-coordinate can be Little Endian or Big Endian
        // (inherited from the field encoding).
        //
        // There are no flag bits, the spare bits must be 0.
        // `BS` is the base size: the number of bytes required to encode a coordinate.
        //
        // According to the number of spare bits:
        // 1 Spare bit:
        //
        //     |                  | 0 | x-coordinate | 0 | y-coordinate |
        //     | Byte pos. (LE)   | BS-1   ..      0 | 2*BS-1   ..   BS |
        //     | Byte pos. (BE)   | 0   ..      BS-1 | BS   ..   2*BS-1 |
        //     | Bit pos.         | 7 |              | 7 |              |
        //     | ---------------- | - | ------------ | - | ------------ |
        //     | Identity         | 0 | 0            | 0 | 0            |
        //     | Non-identity $P$ | 0 | $P.x$        | 0 | $P.y$        |
        //
        // ----
        // 2 Spare bits:
        //
        //     |                  | 0 | 0 | x-coordinate | 0 | 0 | y-coordinate |
        //     | Byte pos. (LE)   | BS-1      ..       0 | 2*BS-1       ..   BS |
        //     | Byte pos. (BE)   | 0      ..       BS-1 | BS       ..   2*BS-1 |
        //     | Bit pos.         | 7 | 6 |              | 7 | 6 |              |
        //     | ---------------- | - | - | ------------ | - | - | ------------ |
        //     | Identity         | 0 | 0 | 0            | 0 | 0 | 0            |
        //     | Non-identity $P$ | 0 | 0 | $P.x$        | 0 | 0 | $P.y$        |
        //
        // ----
        // 0 Spare bits:
        //
        //     |                  | x-coordinate | y-coordinate |
        //     | Byte pos. (LE)   | BS-1  ..   0 | 2*BS-1 .. BS |
        //     | Byte pos. (BE)   | 0  ..   BS-1 | BS .. 2*BS-1 |
        //     | ---------------- | ------------ | ------------ |
        //     | Identity         | 0            | 0            |
        //     | Non-identity $P$ | $P.x$        | $P.y$        |
        //

        macro_rules! impl_uncompressed {
            ($spare_bits: expr) => {
                paste::paste! {

                #[derive(Copy, Clone)]
                pub struct [< $name Uncompressed >]([u8; 2*$base::size()]);
                    impl std::fmt::Debug for [< $name Uncompressed >] {
                        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                            self.0[..].fmt(f)
                        }
                    }

                    impl Default for [< $name Uncompressed >] {
                        fn default() -> Self {
                            [< $name Uncompressed >]([0; 2*$base::size() ])
                        }
                    }

                    impl AsRef<[u8]> for [< $name Uncompressed >] {
                        fn as_ref(&self) -> &[u8] {
                            &self.0
                        }
                    }

                    impl AsMut<[u8]> for [< $name Uncompressed >] {
                        fn as_mut(&mut self) -> &mut [u8] {
                            &mut self.0
                        }
                    }

                    impl ConstantTimeEq for [< $name Uncompressed >] {
                        fn ct_eq(&self, other: &Self) -> Choice {
                            self.0.ct_eq(&other.0)
                        }
                    }

                    impl Eq for [< $name Uncompressed >] {}

                    impl PartialEq for [< $name Uncompressed >] {
                        #[inline]
                        fn eq(&self, other: &Self) -> bool {
                            bool::from(self.ct_eq(other))
                        }
                    }

                    impl group::UncompressedEncoding for $name_affine{
                        type Uncompressed = [< $name Uncompressed >];

                        fn from_uncompressed(bytes: &Self::Uncompressed) -> CtOption<Self> {
                            Self::from_uncompressed_unchecked(bytes).and_then(|p| CtOption::new(p, p.is_on_curve()))
                        }

                        fn from_uncompressed_unchecked(bytes: &Self::Uncompressed) -> CtOption<Self> {
                            let mut bytes = bytes.0;

                            let flag_idx_x = $base::size() -1;
                            let flag_idx_y = 2* $base::size() -1;

                            // In the uncompressed format, the spare bits in both coordinates must be 0.
                            let mut any_flag_set = Choice::from(0u8);

                            // Get sign flag to check they are set to 0.
                            if $spare_bits == 2 || $spare_bits == 1 {
                                any_flag_set |=  Choice::from( (bytes[ flag_idx_x ] & SIGN_MASK) >> SIGN_SHIFT  |
                             (bytes[ flag_idx_y ] & SIGN_MASK) >> SIGN_SHIFT )
                            }

                            // Get identity flag to check they are set to 0.
                            if $spare_bits == 2 {
                                any_flag_set |= Choice::from( (( bytes[ flag_idx_x ] & IDENTITY_MASK) >> IDENTITY_SHIFT) | (( bytes[ flag_idx_y ] & IDENTITY_MASK) >> IDENTITY_SHIFT) );
                            }

                            // Clear spare bits.
                            if $spare_bits == 2 || $spare_bits == 1 {
                                bytes[flag_idx_x] &= ![< $name _FLAG_BITS >];
                                bytes[flag_idx_y] &= ![< $name _FLAG_BITS >];
                            }


                            // Get x, y coordinates.
                            let mut repr = [0u8; $base::size()];
                            let x = {
                                repr.copy_from_slice(&bytes[0..$base::size()]);
                                $base::from_bytes(&repr)
                            };

                            let y = {
                                repr.copy_from_slice(&bytes[$base::size()..2*$base::size()]);
                                $base::from_bytes(&repr)
                            };


                            x.and_then(|x| {
                                y.and_then(|y| {
                                    let zero_coords = x.is_zero() & y.is_zero();

                                    // Check  identity condition and encoding validity:
                                    // The point is the identity if both coordinates are zero.
                                    // The encoding is valid if both coordinates represent valid field elements and
                                    // the spare bits are all zero.
                                    let (is_valid, is_identity) =
                                        ( !any_flag_set, zero_coords);


                                    let p = $name_affine::conditional_select(
                                        &$name_affine{
                                            x,
                                            y,
                                        },
                                        &$name_affine::identity(),
                                        is_identity,
                                    );

                                    // eprintln!("Is the point valid? {:?}", is_valid);

                                    CtOption::new(
                                        p,
                                        is_valid
                                    )
                                })
                            })
                        }

                        fn to_uncompressed(&self) -> Self::Uncompressed {
                            let mut res = [0; 2*$base::size()];

                            res[0..$base::size()].copy_from_slice(
                                &$base::conditional_select(&self.x, &$base::zero(), self.is_identity()).to_bytes()[..],
                            );
                            res[$base::size().. 2*$base::size()].copy_from_slice(
                                &$base::conditional_select(&self.y, &$base::zero(), self.is_identity()).to_bytes()[..],
                            );

                            [< $name Uncompressed >](res)
                        }
                    }
                }
            };
        }

        /// A macro to help define point serialization using the [`group::GroupEncoding`] trait
        /// This assumes both point types ($name, $nameaffine) implement [`group::GroupEncoding`].
        #[cfg(feature = "derive_serde")]
        macro_rules! serialize_deserialize_to_from_bytes {
            () => {
                impl ::serde::Serialize for $name {
                    fn serialize<S: ::serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                        let bytes = &self.to_bytes();
                        if serializer.is_human_readable() {
                            ::hex::serde::serialize(&bytes.0, serializer)
                        } else {
                            ::serde_arrays::serialize(&bytes.0, serializer)
                        }
                    }
                }

                paste::paste! {
                    impl<'de> ::serde::Deserialize<'de> for $name {
                        fn deserialize<D: ::serde::Deserializer<'de>>(
                            deserializer: D,
                        ) -> Result<Self, D::Error> {
                            use ::serde::de::Error as _;
                            let bytes = if deserializer.is_human_readable() {
                                ::hex::serde::deserialize(deserializer)?
                            } else {
                                ::serde_arrays::deserialize::<_, u8, [< $name _COMPRESSED_SIZE >]>(deserializer)?
                            };
                            Option::from(Self::from_bytes(&[< $name Compressed >](bytes))).ok_or_else(|| {
                                D::Error::custom("deserialized bytes don't encode a valid field element")
                            })
                        }
                    }
                }

                impl ::serde::Serialize for $name_affine {
                    fn serialize<S: ::serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                        let bytes = &self.to_bytes();
                        if serializer.is_human_readable() {
                            ::hex::serde::serialize(&bytes.0, serializer)
                        } else {
                            ::serde_arrays::serialize(&bytes.0, serializer)
                        }
                    }
                }

                paste::paste! {
                    impl<'de> ::serde::Deserialize<'de> for $name_affine {
                        fn deserialize<D: ::serde::Deserializer<'de>>(
                            deserializer: D,
                        ) -> Result<Self, D::Error> {
                            use ::serde::de::Error as _;
                            let bytes = if deserializer.is_human_readable() {
                                ::hex::serde::deserialize(deserializer)?
                            } else {
                                ::serde_arrays::deserialize::<_, u8, [< $name _COMPRESSED_SIZE >]>(deserializer)?
                            };
                            Option::from(Self::from_bytes(&[< $name Compressed >](bytes))).ok_or_else(|| {
                                D::Error::custom("deserialized bytes don't encode a valid field element")
                            })
                        }
                    }
                }
            };
        }

        #[derive(Copy, Clone, Debug)]
        $($privacy)* struct $name {
            pub x: $base,
            pub y: $base,
            pub z: $base,
        }

        #[derive(Copy, Clone, PartialEq)]
        $($privacy)* struct $name_affine {
            pub x: $base,
            pub y: $base,
        }

        #[cfg(feature = "derive_serde")]
        serialize_deserialize_to_from_bytes!();

        // Base's num_bits is the number of bits for the base prime field,
        // so the computation of spare bits is correct for extensions as well.
        impl_compressed!((($base::NUM_BITS-1) / 8 +1) * 8 - $base::NUM_BITS);
        impl_uncompressed!((($base::NUM_BITS-1) / 8 +1) * 8 - $base::NUM_BITS);



        impl $name {
            pub fn generator() -> Self {
                let generator = $name_affine::generator();
                Self {
                    x: generator.x,
                    y: generator.y,
                    z: $base::one(),
                }
            }

            #[inline]
            fn curve_constant_3b() -> $base {
                lazy_static::lazy_static! {
                    static ref CONST_3B: $base = $constant_b + $constant_b + $constant_b;
                }
                *CONST_3B
            }

            fn mul_by_3b(input: &$base) -> $base {
                if $name::CURVE_ID == "bn256_g1"{
                    input.double().double().double() + input
                } else {
                    input * $name::curve_constant_3b()
                }
            }
        }

        impl $name_affine {
            pub fn generator() -> Self {
                Self {
                    x: $generator.0,
                    y: $generator.1,
                }
            }

            #[inline(always)]
            fn y2(x: $base) -> $base {
                if $constant_a == $base::ZERO {
                    let x3 = x.square() * x;
                    (x3 + $constant_b)
                } else {
                    let x2 = x.square();
                    ((x2 + $constant_a) * x + $constant_b)
                }
            }

            pub fn random(mut rng: impl RngCore) -> Self {
                loop {
                    let x = $base::random(&mut rng);
                    let ysign = (rng.next_u32() % 2) as u8;

                    let y2 = $name_affine::y2(x);
                    if let Some(y) = Option::<$base>::from(y2.sqrt()) {
                        let sign = y.to_bytes()[0] & 1;
                        let y = if ysign ^ sign == 0 { y } else { -y };

                        let p = $name_affine {
                            x,
                            y,
                        };


                        use $crate::group::cofactor::CofactorGroup;
                        let p = p.to_curve();
                        return p.clear_cofactor().to_affine()
                    }
                }
            }
        }



        // Jacobian implementations

        impl<'a> From<&'a $name_affine> for $name {
            fn from(p: &'a $name_affine) -> $name {
                p.to_curve()
            }
        }

        impl From<$name_affine> for $name {
            fn from(p: $name_affine) -> $name {
                p.to_curve()
            }
        }

        impl Default for $name {
            fn default() -> $name {
                $name::identity()
            }
        }

        impl subtle::ConstantTimeEq for $name {
            fn ct_eq(&self, other: &Self) -> Choice {
                // Is (x, y, z) equal to (x', y, z') when converted to affine?
                // => (x/z , y/z) equal to (x'/z' , y'/z')
                // => (xz' == x'z) & (yz' == y'z)

                let x1 = self.x * other.z;
                let y1 = self.y * other.z;

                let x2 = other.x * self.z;
                let y2 = other.y * self.z;

                let self_is_zero = self.is_identity();
                let other_is_zero = other.is_identity();

                (self_is_zero & other_is_zero) // Both point at infinity
                            | ((!self_is_zero) & (!other_is_zero) & x1.ct_eq(&x2) & y1.ct_eq(&y2))
                // Neither point at infinity, coordinates are the same
            }

        }

        impl subtle::ConditionallySelectable for $name {
            fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
                $name {
                    x: $base::conditional_select(&a.x, &b.x, choice),
                    y: $base::conditional_select(&a.y, &b.y, choice),
                    z: $base::conditional_select(&a.z, &b.z, choice),
                }
            }
        }

        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.ct_eq(other).into()
            }
        }

        impl cmp::Eq for $name {}

        impl CurveExt for $name {

            type ScalarExt = $scalar;
            type Base = $base;
            type AffineExt = $name_affine;

            const CURVE_ID: &'static str = $curve_id;

            fn endo(&self) -> Self {
                Self {
                    x: self.x * Self::Base::ZETA,
                    y: self.y,
                    z: self.z,
                }
            }

            fn jacobian_coordinates(&self) -> ($base, $base, $base) {
                // Homogeneous to Jacobian
                let x = self.x * self.z;
                let y = self.y * self.z.square();
                (x, y, self.z)
            }


            #[allow(clippy::redundant_closure_call)]
            fn hash_to_curve<'a>(domain_prefix: &'a str) -> Box<dyn Fn(&[u8]) -> Self + 'a> {
                $hash_to_curve(domain_prefix)
            }

            fn is_on_curve(&self) -> Choice {
                if $constant_a == $base::ZERO {
                    // Check (Y/Z)^2 = (X/Z)^3 + b
                    // <=>    Z Y^2 - X^3 = Z^3 b

                    (self.z * self.y.square() - self.x.square() * self.x)
                        .ct_eq(&(self.z.square() * self.z * $constant_b))
                        | self.z.is_zero()
                } else {
                    // Check (Y/Z)^2 = (X/Z)^3 + a(X/Z) + b
                    // <=>    Z Y^2 - X^3 - a(X Z^2) = Z^3 b

                    let z2 = self.z.square();
                    (self.z * self.y.square() - (self.x.square() + $constant_a * z2) * self.x)
                        .ct_eq(&(z2 * self.z * $constant_b))
                        | self.z.is_zero()
                }
            }

            fn b() -> Self::Base {
                $constant_b
            }

            fn a() -> Self::Base {
                $constant_a
            }

            fn new_jacobian(x: Self::Base, y: Self::Base, z: Self::Base) -> CtOption<Self> {
                // Jacobian to homogeneous
                let z_inv = z.invert().unwrap_or($base::zero());
                let p_x = x * z_inv;
                let p_y = y * z_inv.square();
                let p = $name {
                    x:p_x,
                    y:$base::conditional_select(&p_y, &$base::one(), z.is_zero()),
                    z
                };
                CtOption::new(p, p.is_on_curve())
            }
        }

        impl group::Curve for $name {

            type AffineRepr = $name_affine;

            fn batch_normalize(p: &[Self], q: &mut [Self::AffineRepr]) {
                assert_eq!(p.len(), q.len());

                let mut acc = $base::one();
                for (p, q) in p.iter().zip(q.iter_mut()) {
                    // We use the `x` field of $name_affine to store the product
                    // of previous z-coordinates seen.
                    q.x = acc;

                    // We will end up skipping all identities in p
                    acc = $base::conditional_select(&(acc * p.z), &acc, p.is_identity());
                }

                // This is the inverse, as all z-coordinates are nonzero and the ones
                // that are not are skipped.
                acc = acc.invert().unwrap();

                for (p, q) in p.iter().rev().zip(q.iter_mut().rev()) {
                    let skip = p.is_identity();

                    // Compute tmp = 1/z
                    let tmp = q.x * acc;

                    // Cancel out z-coordinate in denominator of `acc`
                    acc = $base::conditional_select(&(acc * p.z), &acc, skip);

                    q.x = p.x * tmp;
                    q.y = p.y * tmp;

                    *q = $name_affine::conditional_select(&q, &$name_affine::identity(), skip);
                }
            }

            fn to_affine(&self) -> Self::AffineRepr {
                let zinv = self.z.invert().unwrap_or($base::zero());
                let x = self.x * zinv;
                let y = self.y * zinv;
                let tmp = $name_affine {
                    x,
                    y,
                };
                $name_affine::conditional_select(&tmp, &$name_affine::identity(), zinv.is_zero())
            }
        }

        impl group::Group for $name {
            type Scalar = $scalar;

            fn random(mut rng: impl RngCore) -> Self {
                $name_affine::random(&mut rng).to_curve()
            }

            fn double(&self) -> Self {
                if $constant_a == $base::ZERO {
                    // Algorithm 9, https://eprint.iacr.org/2015/1060.pdf
                    let t0 = self.y.square();
                    let z3 = t0 + t0;
                    let z3 = z3 + z3;
                    let z3 = z3 + z3;
                    let t1 = self.y * self.z;
                    let t2 = self.z.square();
                    let t2 = $name::mul_by_3b(&t2);
                    let x3 = t2 * z3;
                    let y3 = t0 + t2;
                    let z3 = t1 * z3;
                    let t1 = t2 + t2;
                    let t2 = t1 + t2;
                    let t0 = t0 - t2;
                    let y3 = t0 * y3;
                    let y3 = x3 + y3;
                    let t1 = self.x * self.y;
                    let x3 = t0 * t1;
                    let x3 = x3 + x3;

                    let tmp = $name {
                        x: x3,
                        y: y3,
                        z: z3,
                    };

                    $name::conditional_select(&tmp, &$name::identity(), self.is_identity())
                } else {
                    // Algorithm 3, https://eprint.iacr.org/2015/1060.pdf
                    let t0 = self.x.square();
                    let t1 = self.y.square();
                    let t2 = self.z.square();
                    let t3 = self.x * self.y;
                    let t3 = t3 + t3;
                    let z3 = self.x * self.z;
                    let z3 = z3 + z3;
                    let x3 = $constant_a * z3;
                    let y3 = $name::mul_by_3b(&t2);
                    let y3 = x3 + y3;
                    let x3 = t1 - y3;
                    let y3 = t1 + y3;
                    let y3 = x3 * y3;
                    let x3 = t3 * x3;
                    let z3 = $name::mul_by_3b(&z3);
                    let t2 = $constant_a * t2;
                    let t3 = t0 - t2;
                    let t3 = $constant_a * t3;
                    let t3 = t3 + z3;
                    let z3 = t0 + t0;
                    let t0 = z3 + t0;
                    let t0 = t0 + t2;
                    let t0 = t0 * t3;
                    let y3 = y3 + t0;
                    let t2 = self.y * self.z;
                    let t2 = t2 + t2;
                    let t0 = t2 * t3;
                    let x3 = x3 - t0;
                    let z3 = t2 * t1;
                    let z3 = z3 + z3;
                    let z3 = z3 + z3;

                    let tmp = $name {
                        x: x3,
                        y: y3,
                        z: z3,
                    };

                    $name::conditional_select(&tmp, &$name::identity(), self.is_identity())
                }
            }

            fn generator() -> Self {
                $name::generator()
            }

            fn identity() -> Self {
                Self {
                    x: $base::zero(),
                    y: $base::one(),
                    z: $base::zero(),
                }
            }

            fn is_identity(&self) -> Choice {
                self.z.is_zero()
            }
        }

        impl $crate::serde::SerdeObject for $name {
            fn from_raw_bytes_unchecked(bytes: &[u8]) -> Self {
                debug_assert_eq!(bytes.len(), 3 * $base::size());
                let [x, y, z] = [0, 1, 2]
                    .map(|i| $base::from_raw_bytes_unchecked(&bytes[i * $base::size()..(i + 1) * $base::size()]));
                Self { x, y, z }
            }
            fn from_raw_bytes(bytes: &[u8]) -> Option<Self> {
                if bytes.len() != 3 * $base::size() {
                    return None;
                }
                let [x, y, z] =
                    [0, 1, 2].map(|i| $base::from_raw_bytes(&bytes[i * $base::size()..(i + 1) * $base::size()]));
                x.zip(y).zip(z).and_then(|((x, y), z)| {
                    let res = Self { x, y, z };
                    // Check that the point is on the curve.
                    bool::from(res.is_on_curve()).then(|| res)
                })
            }
            fn to_raw_bytes(&self) -> Vec<u8> {
                let mut res = Vec::with_capacity(3 * $base::size());
                Self::write_raw(self, &mut res).unwrap();
                res
            }
            fn read_raw_unchecked<R: std::io::Read>(reader: &mut R) -> Self {
                let [x, y, z] = [(); 3].map(|_| $base::read_raw_unchecked(reader));
                Self { x, y, z }
            }
            fn read_raw<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
                let x = $base::read_raw(reader)?;
                let y = $base::read_raw(reader)?;
                let z = $base::read_raw(reader)?;
                Ok(Self { x, y, z })
            }
            fn write_raw<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
                self.x.write_raw(writer)?;
                self.y.write_raw(writer)?;
                self.z.write_raw(writer)
            }
        }

        impl group::prime::PrimeGroup for $name {}

        impl group::prime::PrimeCurve for $name {
            type Affine = $name_affine;
        }

        impl group::cofactor::CofactorCurve for $name {
            type Affine = $name_affine;
        }

        // Affine implementations

        impl std::fmt::Debug for $name_affine {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
                if self.is_identity().into() {
                    write!(f, "Infinity")
                } else {
                    write!(f, "({:?}, {:?})", self.x, self.y)
                }
            }
        }

        impl<'a> From<&'a $name> for $name_affine {
            fn from(p: &'a $name) -> $name_affine {
                p.to_affine()
            }
        }

        impl From<$name> for $name_affine {
            fn from(p: $name) -> $name_affine {
                p.to_affine()
            }
        }

        impl Default for $name_affine {
            fn default() -> $name_affine {
                $name_affine::identity()
            }
        }

        impl subtle::ConstantTimeEq for $name_affine {
            fn ct_eq(&self, other: &Self) -> Choice {
                let z1 = self.is_identity();
                let z2 = other.is_identity();

                (z1 & z2) | ((!z1) & (!z2) & (self.x.ct_eq(&other.x)) & (self.y.ct_eq(&other.y)))
            }
        }

        impl subtle::ConditionallySelectable for $name_affine {
            fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
                $name_affine {
                    x: $base::conditional_select(&a.x, &b.x, choice),
                    y: $base::conditional_select(&a.y, &b.y, choice),
                }
            }
        }

        impl cmp::Eq for $name_affine {}


        impl $crate::serde::SerdeObject for $name_affine {
            fn from_raw_bytes_unchecked(bytes: &[u8]) -> Self {
                debug_assert_eq!(bytes.len(), 2 * $base::size());
                let [x, y] =
                    [0, $base::size()].map(|i| $base::from_raw_bytes_unchecked(&bytes[i..i + $base::size()]));
                Self { x, y }
            }
            fn from_raw_bytes(bytes: &[u8]) -> Option<Self> {
                if bytes.len() != 2 * $base::size() {
                    return None;
                }
                let [x, y] = [0, $base::size()].map(|i| $base::from_raw_bytes(&bytes[i..i + $base::size()]));
                x.zip(y).and_then(|(x, y)| {
                    let res = Self { x, y };
                    // Check that the point is on the curve.
                    bool::from(res.is_on_curve()).then(|| res)
                })
            }
            fn to_raw_bytes(&self) -> Vec<u8> {
                let mut res = Vec::with_capacity(2 * $base::size());
                Self::write_raw(self, &mut res).unwrap();
                res
            }
            fn read_raw_unchecked<R: std::io::Read>(reader: &mut R) -> Self {
                let [x, y] = [(); 2].map(|_| $base::read_raw_unchecked(reader));
                Self { x, y }
            }
            fn read_raw<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
                let x = $base::read_raw(reader)?;
                let y = $base::read_raw(reader)?;
                Ok(Self { x, y })
            }
            fn write_raw<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
                self.x.write_raw(writer)?;
                self.y.write_raw(writer)
            }
        }

        impl group::prime::PrimeCurveAffine for $name_affine {
            type Curve = $name;
            type Scalar = $scalar;


            fn generator() -> Self {
                $name_affine::generator()
            }

            fn identity() -> Self {
                Self {
                    x: $base::zero(),
                    y: $base::zero(),
                }
            }

            fn is_identity(&self) -> Choice {
                self.x.is_zero() & self.y.is_zero()
            }

            fn to_curve(&self) -> Self::Curve {
                let tmp = $name {
                    x: self.x,
                    y: self.y,
                    z: $base::one(),
                };
                $name::conditional_select(&tmp, &$name::identity(), self.is_identity())
            }
        }

        impl group::cofactor::CofactorCurveAffine for $name_affine {
            type Curve = $name;
            type Scalar = $scalar;

            fn identity() -> Self {
                <Self as group::prime::PrimeCurveAffine>::identity()
            }

            fn generator() -> Self {
                <Self as group::prime::PrimeCurveAffine>::generator()
            }

            fn is_identity(&self) -> Choice {
                <Self as group::prime::PrimeCurveAffine>::is_identity(self)
            }

            fn to_curve(&self) -> Self::Curve {
                <Self as group::prime::PrimeCurveAffine>::to_curve(self)
            }
        }


        impl CurveAffine for $name_affine {
            type ScalarExt = $scalar;
            type Base = $base;
            type CurveExt = $name;

            fn is_on_curve(&self) -> Choice {
                if $constant_a == $base::ZERO {
                    // y^2 - x^3 ?= b
                    (self.y.square() - self.x.square() * self.x).ct_eq(&$constant_b)
                        | self.is_identity()
                } else {
                    // y^2 - x^3 - ax ?= b
                    (self.y.square() - (self.x.square() + $constant_a) * self.x).ct_eq(&$constant_b)
                        | self.is_identity()
                }
            }

            fn coordinates(&self) -> CtOption<Coordinates<Self>> {
                Coordinates::from_xy( self.x, self.y )
            }

            fn from_xy(x: Self::Base, y: Self::Base) -> CtOption<Self> {
                let p = $name_affine {
                    x, y
                };
                CtOption::new(p, p.is_on_curve())
            }

            fn a() -> Self::Base {
                $constant_a
            }

            fn b() -> Self::Base {
                $constant_b
            }
        }


        impl_binops_additive!($name, $name);
        impl_binops_additive!($name, $name_affine);
        impl_binops_additive_specify_output!($name_affine, $name_affine, $name);
        impl_binops_additive_specify_output!($name_affine, $name, $name);
        impl_binops_multiplicative!($name, $scalar);
        impl_binops_multiplicative_mixed!($name_affine, $scalar, $name);

        impl<'a> Neg for &'a $name {
            type Output = $name;

            fn neg(self) -> $name {
                $name {
                    x: self.x,
                    y: -self.y,
                    z: self.z,
                }
            }
        }

        impl Neg for $name {
            type Output = $name;

            fn neg(self) -> $name {
                -&self
            }
        }

        impl<T> Sum<T> for $name
        where
            T: core::borrow::Borrow<$name>,
        {
            fn sum<I>(iter: I) -> Self
            where
                I: Iterator<Item = T>,
            {
                iter.fold(Self::identity(), |acc, item| acc + item.borrow())
            }
        }

        impl<'a, 'b> Add<&'a $name> for &'b $name {
            type Output = $name;

            fn add(self, rhs: &'a $name) -> $name {
                if $constant_a == $base::ZERO {
                    // Algorithm 7, https://eprint.iacr.org/2015/1060.pdf
                    let t0 = self.x * rhs.x;
                    let t1 = self.y * rhs.y;
                    let t2 = self.z * rhs.z;
                    let t3 = self.x + self.y;
                    let t4 = rhs.x + rhs.y;
                    let t3 = t3 * t4;
                    let t4 = t0 + t1;
                    let t3 = t3 - t4;
                    let t4 = self.y + self.z;
                    let x3 = rhs.y + rhs.z;
                    let t4 = t4 * x3;
                    let x3 = t1 + t2;
                    let t4 = t4 - x3;
                    let x3 = self.x + self.z;
                    let y3 = rhs.x + rhs.z;
                    let x3 = x3 * y3;
                    let y3 = t0 + t2;
                    let y3 = x3 - y3;
                    let x3 = t0 + t0;
                    let t0 = x3 + t0;
                    let t2 = $name::mul_by_3b(&t2);
                    let z3 = t1 + t2;
                    let t1 = t1 - t2;
                    let y3 = $name::mul_by_3b(&y3);
                    let x3 = t4 * y3;
                    let t2 = t3 * t1;
                    let x3 = t2 - x3;
                    let y3 = y3 * t0;
                    let t1 = t1 * z3;
                    let y3 = t1 + y3;
                    let t0 = t0 * t3;
                    let z3 = z3 * t4;
                    let z3 = z3 + t0;

                    $name {
                        x: x3,
                        y: y3,
                        z: z3,
                    }
                } else {
                    // Algorithm 1, https://eprint.iacr.org/2015/1060.pdf
                    let t0 = self.x * rhs.x;
                    let t1 = self.y * rhs.y;
                    let t2 = self.z * rhs.z;
                    let t3 = self.x + self.y;
                    let t4 = rhs.x + rhs.y;
                    let t3 = t3 * t4;
                    let t4 = t0 + t1;
                    let t3 = t3 - t4;
                    let t4 = self.x + self.z;
                    let t5 = rhs.x + rhs.z;
                    let t4 = t4 * t5;
                    let t5 = t0 + t2;
                    let t4 = t4 - t5;
                    let t5 = self.y + self.z;
                    let x3 = rhs.y + rhs.z;
                    let t5 = t5 * x3;
                    let x3 = t1 + t2;
                    let t5 = t5 - x3;
                    let z3 = $constant_a * t4;
                    let x3 = $name::mul_by_3b(&t2);
                    let z3 = x3 + z3;
                    let x3 = t1 - z3;
                    let z3 = t1 + z3;
                    let y3 = x3 * z3;
                    let t1 = t0 + t0;
                    let t1 = t1 + t0;
                    let t2 = $constant_a * t2;
                    let t4 = $name::mul_by_3b(&t4);
                    let t1 = t1 + t2;
                    let t2 = t0 - t2;
                    let t2 = $constant_a * t2;
                    let t4 = t4 + t2;
                    let t0 = t1 * t4;
                    let y3 = y3 + t0;
                    let t0 = t5 * t4;
                    let x3 = t3 * x3;
                    let x3 = x3 - t0;
                    let t0 = t3 * t1;
                    let z3 = t5 * z3;
                    let z3 = z3 + t0;

                    $name {
                        x: x3,
                        y: y3,
                        z: z3,
                    }
                }
            }
        }

        impl<'a, 'b> Add<&'a $name_affine> for &'b $name {
            type Output = $name;

            // Mixed addition
            fn add(self, rhs: &'a $name_affine) -> $name {
                if $constant_a == $base::ZERO {
                    // Algorithm 8, https://eprint.iacr.org/2015/1060.pdf
                    let t0 = self.x * rhs.x;
                    let t1 = self.y * rhs.y;
                    let t3 = rhs.x + rhs.y;
                    let t4 = self.x + self.y;
                    let t3 = t3 * t4;
                    let t4 = t0 + t1;
                    let t3 = t3 - t4;
                    let t4 = rhs.y * self.z;
                    let t4 = t4 + self.y;
                    let y3 = rhs.x * self.z;
                    let y3 = y3 + self.x;
                    let x3 = t0 + t0;
                    let t0 = x3 + t0;
                    let t2 = $name::mul_by_3b(&self.z);
                    let z3 = t1 + t2;
                    let t1 = t1 - t2;
                    let y3 = $name::mul_by_3b(&y3);
                    let x3 = t4 * y3;
                    let t2 = t3 * t1;
                    let x3 = t2 - x3;
                    let y3 = y3 * t0;
                    let t1 = t1 * z3;
                    let y3 = t1 + y3;
                    let t0 = t0 * t3;
                    let z3 = z3 * t4;
                    let z3 = z3 + t0;

                    let tmp = $name{
                        x: x3,
                        y: y3,
                        z: z3,
                    };

                    $name::conditional_select(&tmp, self, rhs.is_identity())
                } else {
                    // Algorithm 2, https://eprint.iacr.org/2015/1060.pdf
                    let t0 = self.x * rhs.x;
                    let t1 = self.y * rhs.y;
                    let t3 = rhs.x + rhs.y;
                    let t4 = self.x + self.y;
                    let t3 = t3 * t4;
                    let t4 = t0 + t1;
                    let t3 = t3 - t4;
                    let t4 = rhs.x * self.z;
                    let t4 = t4 + self.x;
                    let t5 = rhs.y * self.z;
                    let t5 = t5 + self.y;
                    let z3 = $constant_a * t4;
                    let x3 = $name::mul_by_3b(&self.z);
                    let z3 = x3 + z3;
                    let x3 = t1 - z3;
                    let z3 = t1 + z3;
                    let y3 = x3 * z3;
                    let t1 = t0 + t0;
                    let t1 = t1 + t0;
                    let t2 = $constant_a * self.z;
                    let t4 = $name::mul_by_3b(&t4);
                    let t1 = t1 + t2;
                    let t2 = t0 - t2;
                    let t2 = $constant_a * t2;
                    let t4 = t4 + t2;
                    let t0 = t1 * t4;
                    let y3 = y3 + t0;
                    let t0 = t5 * t4;
                    let x3 = t3 * x3;
                    let x3 = x3 - t0;
                    let t0 = t3 * t1;
                    let z3 = t5 * z3;
                    let z3 = z3 + t0;

                    let tmp = $name{
                        x: x3,
                        y: y3,
                        z: z3,
                    };

                    $name::conditional_select(&tmp, self, rhs.is_identity())
                }
            }
        }

        impl<'a, 'b> Sub<&'a $name> for &'b $name {
            type Output = $name;

            fn sub(self, other: &'a $name) -> $name {
                self + (-other)
            }
        }

        impl<'a, 'b> Sub<&'a $name_affine> for &'b $name {
            type Output = $name;

            fn sub(self, other: &'a $name_affine) -> $name {
                self + (-other)
            }
        }



        #[allow(clippy::suspicious_arithmetic_impl)]
        impl<'a, 'b> Mul<&'b $scalar> for &'a $name {
            type Output = $name;

            // This is a simple double-and-add implementation of point
            // multiplication, moving from most significant to least
            // significant bit of the scalar.

            fn mul(self, other: &'b $scalar) -> Self::Output {
                let mut acc = $name::identity();
                for bit in other
                    .to_repr()
                    .iter()
                    .rev()
                    .flat_map(|byte| (0..8).rev().map(move |i| Choice::from((byte >> i) & 1u8)))
                {
                    acc = acc.double();
                    acc = $name::conditional_select(&acc, &(acc + self), bit);
                }

                acc
            }
        }

        impl<'a> Neg for &'a $name_affine {
            type Output = $name_affine;

            fn neg(self) -> $name_affine {
                $name_affine {
                    x: self.x,
                    y: -self.y,
                }
            }
        }

        impl Neg for $name_affine {
            type Output = $name_affine;

            fn neg(self) -> $name_affine {
                -&self
            }
        }

        impl<'a, 'b> Add<&'a $name> for &'b $name_affine {
            type Output = $name;

            fn add(self, rhs: &'a $name) -> $name {
                rhs + self
            }
        }

        impl<'a, 'b> Add<&'a $name_affine> for &'b $name_affine {
            type Output = $name;

            fn add(self, rhs: &'a $name_affine) -> $name {
                rhs.to_curve() + self.to_curve()
            }
        }

        impl<'a, 'b> Sub<&'a $name_affine> for &'b $name_affine {
            type Output = $name;

            fn sub(self, other: &'a $name_affine) -> $name {
                self + (-other)
            }
        }

        impl<'a, 'b> Sub<&'a $name> for &'b $name_affine {
            type Output = $name;

            fn sub(self, other: &'a $name) -> $name {
                self + (-other)
            }
        }

        #[allow(clippy::suspicious_arithmetic_impl)]
        impl<'a, 'b> Mul<&'b $scalar> for &'a $name_affine {
            type Output = $name;

            fn mul(self, other: &'b $scalar) -> Self::Output {
                let mut acc = $name::identity();

                // This is a simple double-and-add implementation of point
                // multiplication, moving from most significant to least
                // significant bit of the scalar.

                for bit in other
                    .to_repr()
                    .iter()
                    .rev()
                    .flat_map(|byte| (0..8).rev().map(move |i| Choice::from((byte >> i) & 1u8)))
                {
                    acc = acc.double();
                    acc = $name::conditional_select(&acc, &(acc + self), bit);
                }

                acc
            }
        }
    };
}
