// Copyright 2025 Irreducible Inc.
//! Test utilities for PackedUnderlier implementations.

#[cfg(test)]
use proptest::prelude::*;

#[cfg(test)]
use crate::underlier::{PackedUnderlier, Underlier};

#[derive(Debug)]
pub enum GetSetOp<U> {
    Get { index: usize },
    Set { index: usize, val: U },
}

/// Test that a sequence of get/set calls on a default underlier
pub fn test_packed_underlier_get_set_behaves_like_vec<U, Inner>(ops: Vec<GetSetOp<Inner>>)
where
    U: PackedUnderlier<Inner>,
    Inner: Underlier + std::fmt::Debug,
{
    let mut subject = U::zero();
    let mut reference = vec![Inner::zero(); 1 << U::LOG_WIDTH];
    for op in ops {
        match op {
            GetSetOp::Get { index } => {
                assert!(Inner::is_equal(subject.get(index), reference[index]));
            }
            GetSetOp::Set { index, val } => {
                subject = subject.set(index, val);
                reference[index] = val;
            }
        }
    }
}

/// Strategy for generating GetSetOp<T> with valid indices for a given width
pub fn arb_get_set_op<T>(width: usize) -> impl Strategy<Value = GetSetOp<T>>
where
    T: Arbitrary + 'static,
{
    prop_oneof![
        // 50% Get operations
        (0..width).prop_map(|index| GetSetOp::Get { index }),
        // 50% Set operations
        (0..width, any::<T>()).prop_map(|(index, val)| GetSetOp::Set { index, val })
    ]
}

/// Generic multiplication test helpers for binary field operations
#[cfg(test)]
#[allow(dead_code)]
pub mod multiplication_tests {
    use super::*;

    /// Test that multiplication is commutative: a * b = b * a
    pub fn test_mul_commutative<T, F>(a: T, b: T, mul_fn: F, field_name: &str)
    where
        T: Underlier,
        F: Fn(T, T) -> T,
    {
        let ab = mul_fn(a, b);
        let ba = mul_fn(b, a); // spellchecker:disable-line
        assert!(
            T::is_equal(ab, ba), // spellchecker:disable-line
            "{field_name} multiplication is not commutative"
        );
    }

    /// Test that multiplication is associative: (a * b) * c = a * (b * c)
    pub fn test_mul_associative<T, F>(a: T, b: T, c: T, mul_fn: F, field_name: &str)
    where
        T: Underlier,
        F: Fn(T, T) -> T,
    {
        let ab_c = mul_fn(mul_fn(a, b), c);
        let a_bc = mul_fn(a, mul_fn(b, c));
        assert!(
            T::is_equal(ab_c, a_bc),
            "{field_name} multiplication is not associative"
        );
    }

    /// Test the distributive law: a * (b + c) = (a * b) + (a * c)
    /// where + is XOR (binary field addition)
    pub fn test_mul_distributive<T, F>(a: T, b: T, c: T, mul_fn: F, field_name: &str)
    where
        T: Underlier,
        F: Fn(T, T) -> T,
    {
        let b_plus_c = T::xor(b, c);
        let a_times_b_plus_c = mul_fn(a, b_plus_c);

        let ab = mul_fn(a, b);
        let ac = mul_fn(a, c);
        let ab_plus_ac = T::xor(ab, ac);

        assert!(
            T::is_equal(a_times_b_plus_c, ab_plus_ac),
            "{field_name} multiplication does not satisfy the distributive law"
        );
    }

    /// Test that the given identity is the multiplicative identity: a * 1 = a
    pub fn test_mul_identity<T, F, U>(a: T, identity: U, mul_fn: F, field_name: &str)
    where
        T: Underlier + PackedUnderlier<U>,
        U: Underlier,
        F: Fn(T, T) -> T,
    {
        let one = T::broadcast(identity);
        let a_times_one = mul_fn(a, one);
        assert!(
            T::is_equal(a_times_one, a),
            "The provided identity is not the multiplicative identity in {field_name}"
        );
    }
}
