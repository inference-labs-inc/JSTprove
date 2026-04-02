// Copyright 2024-2025 Irreducible Inc.

use std::{
    fmt::{Debug, Display, Formatter},
    iter::{Product, Sum},
    marker::PhantomData,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use binius_utils::{
    DeserializeBytes, SerializationError, SerializeBytes,
    bytes::{Buf, BufMut},
};
use bytemuck::{Pod, Zeroable};

use super::{
    PackedExtension, PackedSubfield,
    arithmetic_traits::InvertOrZero,
    binary_field::{BinaryField, BinaryField1b, binary_field, impl_field_extension},
    binary_field_arithmetic::TowerFieldArithmetic,
    mul_by_binary_field_1b,
};
use crate::{
    ExtensionField, Field, TowerField, binary_field_arithmetic::impl_arithmetic_using_packed,
    linear_transformation::Transformation, underlier::U1,
};

// These fields represent a tower based on AES GF(2^8) field (GF(256)/x^8+x^4+x^3+x+1)
// that is isomorphically included into binary tower, i.e.:
//  - AESTowerField16b is GF(2^16) / (x^2 + x * x_2 + 1) where `x_2` is 0x10 from
// BinaryField8b isomorphically projected to AESTowerField8b.
//  - AESTowerField32b is GF(2^32) / (x^2 + x * x_3 + 1), where `x_3` is 0x1000 from
//    AESTowerField16b.
//  ...
binary_field!(pub AESTowerField8b(u8), 0xD0);

unsafe impl Pod for AESTowerField8b {}

impl_field_extension!(BinaryField1b(U1) < @3 => AESTowerField8b(u8));

mul_by_binary_field_1b!(AESTowerField8b);

impl_arithmetic_using_packed!(AESTowerField8b);

impl TowerField for AESTowerField8b {
    fn min_tower_level(self) -> usize {
        match self {
            Self::ZERO | Self::ONE => 0,
            _ => 3,
        }
    }

    fn mul_primitive(self, iota: usize) -> Self {
        match iota {
            0..=1 => self * ISOMORPHIC_ALPHAS[iota],
            2 => self.multiply_alpha(),
            _ => panic!("iota {iota} must be less than tower level 3"),
        }
    }
}

/// A 3- step transformation :
/// 1. Cast to base b-bit packed field
/// 2. Apply linear transformation between aes and binary b8 tower fields
/// 3. Cast back to the target field
pub struct SubfieldTransformer<IF, OF, T> {
    inner_transform: T,
    _ip_pd: PhantomData<IF>,
    _op_pd: PhantomData<OF>,
}

impl<IF, OF, IEP, OEP, T> Transformation<IEP, OEP> for SubfieldTransformer<IF, OF, T>
where
    IF: Field,
    OF: Field,
    IEP: PackedExtension<IF>,
    OEP: PackedExtension<OF>,
    T: Transformation<PackedSubfield<IEP, IF>, PackedSubfield<OEP, OF>>,
{
    fn transform(&self, input: &IEP) -> OEP {
        OEP::cast_ext(self.inner_transform.transform(IEP::cast_base_ref(input)))
    }
}

/// Values isomorphic to 0x02, 0x04 and 0x10 in BinaryField8b
const ISOMORPHIC_ALPHAS: [AESTowerField8b; 3] = [
    AESTowerField8b(0xBC),
    AESTowerField8b(0xB0),
    AESTowerField8b(0xD3),
];

impl SerializeBytes for AESTowerField8b {
    fn serialize(&self, write_buf: impl BufMut) -> Result<(), SerializationError> {
        self.0.serialize(write_buf)
    }
}

impl DeserializeBytes for AESTowerField8b {
    fn deserialize(read_buf: impl Buf) -> Result<Self, SerializationError>
    where
        Self: Sized,
    {
        Ok(Self(DeserializeBytes::deserialize(read_buf)?))
    }
}

#[cfg(test)]
mod tests {
    use binius_utils::{SerializeBytes, bytes::BytesMut};
    use proptest::{arbitrary::any, proptest};
    use rand::prelude::*;

    use super::*;
    use crate::{
        Random, binary_field::tests::is_binary_field_valid_generator, underlier::WithUnderlier,
    };

    fn check_square(f: impl Field) {
        assert_eq!(f.square(), f * f);
    }

    proptest! {
        #[test]
        fn test_square_8(a in any::<u8>()) {
            check_square(AESTowerField8b::from(a))
        }
    }

    fn check_invert(f: impl Field) {
        let inversed = f.invert();
        if f.is_zero() {
            assert!(inversed.is_none());
        } else {
            assert_eq!(inversed.unwrap() * f, Field::ONE);
        }
    }

    proptest! {
        #[test]
        fn test_invert_8(a in any::<u8>()) {
            check_invert(AESTowerField8b::from(a))
        }
    }

    fn check_mul_by_one<F: Field>(f: F) {
        assert_eq!(F::ONE * f, f);
        assert_eq!(f * F::ONE, f);
    }

    fn check_commutative<F: Field>(f_1: F, f_2: F) {
        assert_eq!(f_1 * f_2, f_2 * f_1);
    }

    fn check_associativity_and_lineraity<F: Field>(f_1: F, f_2: F, f_3: F) {
        assert_eq!(f_1 * (f_2 * f_3), (f_1 * f_2) * f_3);
        assert_eq!(f_1 * (f_2 + f_3), f_1 * f_2 + f_1 * f_3);
    }

    fn check_mul<F: Field>(f_1: F, f_2: F, f_3: F) {
        check_mul_by_one(f_1);
        check_mul_by_one(f_2);
        check_mul_by_one(f_3);

        check_commutative(f_1, f_2);
        check_commutative(f_1, f_3);
        check_commutative(f_2, f_3);

        check_associativity_and_lineraity(f_1, f_2, f_3);
        check_associativity_and_lineraity(f_1, f_3, f_2);
        check_associativity_and_lineraity(f_2, f_1, f_3);
        check_associativity_and_lineraity(f_2, f_3, f_1);
        check_associativity_and_lineraity(f_3, f_1, f_2);
        check_associativity_and_lineraity(f_3, f_2, f_1);
    }

    proptest! {
        #[test]
        fn test_mul_8(a in any::<u8>(), b in any::<u8>(), c in any::<u8>()) {
            check_mul(AESTowerField8b::from(a), AESTowerField8b::from(b), AESTowerField8b::from(c))
        }
    }

    #[test]
    fn test_multiplicative_generators() {
        assert!(is_binary_field_valid_generator::<AESTowerField8b>());
    }

    fn test_mul_primitive_valid<F: TowerField + WithUnderlier<Underlier: From<u8>>>(
        val: F,
        iota: usize,
    ) {
        let result = val.mul_primitive(iota);
        let expected =
            val * F::from_underlier(F::Underlier::from(ISOMORPHIC_ALPHAS[iota].to_underlier()));
        assert_eq!(result, expected);
    }

    proptest! {
        #[test]
        fn test_mul_primitive_8b(val in 0u8.., iota in 0usize..3) {
            test_mul_primitive_valid::<AESTowerField8b>(val.into(), iota)
        }
    }

    #[test]
    #[should_panic(expected = "must be less than tower level")]
    fn test_mul_primitive_out_of_range() {
        let val = AESTowerField8b::from(1u8);
        val.mul_primitive(3);
    }

    #[test]
    fn test_serialization() {
        let mut buffer = BytesMut::new();
        let mut rng = StdRng::seed_from_u64(0);
        let aes8 = AESTowerField8b::random(&mut rng);

        SerializeBytes::serialize(&aes8, &mut buffer).unwrap();

        let mut read_buffer = buffer.freeze();

        assert_eq!(
            AESTowerField8b::deserialize(&mut read_buffer).unwrap(),
            aes8
        );
    }
}
