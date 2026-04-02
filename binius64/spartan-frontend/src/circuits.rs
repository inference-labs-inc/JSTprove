// Copyright 2025 Irreducible Inc.

use std::{iter, iter::successors};

use binius_field::{BinaryField128bGhash as B128, Field, arithmetic_traits::InvertOrZero};

use crate::circuit_builder::CircuitBuilder;

pub fn extrapolate_line<Builder: CircuitBuilder>(
    builder: &mut Builder,
    y0: Builder::Wire,
    y1: Builder::Wire,
    z: Builder::Wire,
) -> Builder::Wire {
    // y(z) = y0 + (y1 - y0) * z
    // In binary fields, subtraction is addition (XOR)
    let diff = builder.add(y1, y0);
    let scaled = builder.mul(diff, z);
    builder.add(y0, scaled)
}

pub fn evaluate_univariate<Builder: CircuitBuilder>(
    builder: &mut Builder,
    coeffs: &[Builder::Wire],
    z: Builder::Wire,
) -> Builder::Wire {
    // Use Horner's method: p(z) = a0 + z(a1 + z(a2 + z(...)))
    // Start from highest degree coefficient and work backwards
    if coeffs.is_empty() {
        return builder.constant(B128::ZERO);
    }

    coeffs[..coeffs.len() - 1]
        .iter()
        .rev()
        .fold(coeffs[coeffs.len() - 1], |acc, &coeff| {
            let temp = builder.mul(acc, z);
            builder.add(temp, coeff)
        })
}

pub fn evaluate_multilinear<Builder: CircuitBuilder>(
    builder: &mut Builder,
    coeffs: &[Builder::Wire],
    coords: &[Builder::Wire],
) -> Vec<Builder::Wire> {
    // coords has length n, coeffs has length 2^n
    // Evaluation algorithm: fold over each coordinate in reverse order
    // For each coordinate, interpolate between pairs: lo + coord * (hi - lo)
    coords
        .iter()
        .rev()
        .fold(coeffs.to_vec(), |current, &coord| {
            let (evals_0, evals_1) = current.split_at(current.len() / 2);
            iter::zip(evals_0, evals_1)
                .map(|(&eval_0, &eval_1)| extrapolate_line(builder, eval_0, eval_1, coord))
                .collect()
        })
}

pub fn powers<Builder: CircuitBuilder>(
    builder: &mut Builder,
    x: Builder::Wire,
    n: usize,
) -> Vec<Builder::Wire> {
    // return a vector of n wires containing the values of x^i for i in [1, n]
    successors(Some(x), |&prev| Some(builder.mul(prev, x)))
        .take(n)
        .collect()
}

pub fn square<Builder: CircuitBuilder>(builder: &mut Builder, x: Builder::Wire) -> Builder::Wire {
    builder.mul(x, x)
}

pub fn invert<Builder: CircuitBuilder>(builder: &mut Builder, x: Builder::Wire) -> Builder::Wire {
    let [inv] = builder.hint([x], |vals| {
        let [x_val] = vals;
        [x_val.invert_or_zero()]
    });

    let one = builder.constant(B128::ONE);
    let prod = builder.mul(x, inv);
    builder.assert_eq(prod, one);

    inv
}

pub fn invert_or_zero<Builder: CircuitBuilder>(
    builder: &mut Builder,
    x: Builder::Wire,
) -> Builder::Wire {
    let [inv] = builder.hint([x], |vals| {
        let [x_val] = vals;
        [x_val.invert_or_zero()]
    });

    let one = builder.constant(B128::ONE);
    let prod = builder.mul(x, inv);
    let prod_sub_one = builder.sub(prod, one);

    let prod_sub_one_or_x = builder.mul(prod_sub_one, x);
    builder.assert_zero(prod_sub_one_or_x);

    let prod_sub_one_or_inv = builder.mul(prod_sub_one, inv);
    builder.assert_zero(prod_sub_one_or_inv);

    inv
}

pub fn assert_is_bit<Builder: CircuitBuilder>(builder: &mut Builder, val: Builder::Wire) {
    let val_sq = square(builder, val);
    builder.assert_eq(val_sq, val);
}

#[cfg(test)]
mod tests {
    use std::{array, iter};

    use binius_field::{BinaryField128bGhash as B128, Field, Random, arithmetic_traits::Square};
    use binius_math::{
        line::extrapolate_line_packed,
        multilinear,
        test_utils::{random_field_buffer, random_scalars},
        univariate,
    };
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;
    use crate::{
        circuit_builder::{ConstraintBuilder, WitnessError, WitnessGenerator},
        compiler::compile,
    };

    trait TestCircuit<const N_INOUT: usize> {
        fn build<Builder: CircuitBuilder>(builder: &mut Builder, inout: [Builder::Wire; N_INOUT]);
    }

    fn test_helper<C: TestCircuit<N_INOUT>, const N_INOUT: usize>(
        inout_vals: [B128; N_INOUT],
    ) -> Result<(), WitnessError> {
        let mut constraint_builder = ConstraintBuilder::new();
        let inout_wires = array::from_fn(|_| constraint_builder.alloc_inout());
        C::build(&mut constraint_builder, inout_wires);
        let (cs, layout) = compile(constraint_builder);

        let mut witness_gen = WitnessGenerator::new(&layout);
        let inout_assigned =
            array::from_fn(|i| witness_gen.write_inout(inout_wires[i], inout_vals[i]));
        C::build(&mut witness_gen, inout_assigned);
        let witness = witness_gen.build()?;

        cs.validate(&witness);
        Ok(())
    }

    #[test]
    fn test_square() {
        struct SquareCircuit;

        impl TestCircuit<2> for SquareCircuit {
            fn build<Builder: CircuitBuilder>(builder: &mut Builder, inout: [Builder::Wire; 2]) {
                let [x, expected] = inout;
                let result = square(builder, x);
                builder.assert_eq(result, expected);
            }
        }

        let mut rng = StdRng::seed_from_u64(0);
        let x_val = B128::random(&mut rng);
        let expected = Square::square(x_val);

        test_helper::<SquareCircuit, 2>([x_val, expected]).unwrap();
    }

    #[test]
    fn test_assert_is_bit_zero() {
        struct BitCheckCircuit;

        impl TestCircuit<1> for BitCheckCircuit {
            fn build<Builder: CircuitBuilder>(builder: &mut Builder, inout: [Builder::Wire; 1]) {
                assert_is_bit(builder, inout[0]);
            }
        }

        // Test that 0 is a valid bit
        test_helper::<BitCheckCircuit, 1>([B128::ZERO]).unwrap();
    }

    #[test]
    fn test_assert_is_bit_one() {
        struct BitCheckCircuit;

        impl TestCircuit<1> for BitCheckCircuit {
            fn build<Builder: CircuitBuilder>(builder: &mut Builder, inout: [Builder::Wire; 1]) {
                assert_is_bit(builder, inout[0]);
            }
        }

        // Test that 1 is a valid bit
        test_helper::<BitCheckCircuit, 1>([B128::ONE]).unwrap();
    }

    #[test]
    fn test_extrapolate_line() {
        struct ExtrapolateLineCircuit;

        impl TestCircuit<4> for ExtrapolateLineCircuit {
            fn build<Builder: CircuitBuilder>(builder: &mut Builder, inout: [Builder::Wire; 4]) {
                let [y0, y1, z, expected] = inout;
                let result = extrapolate_line(builder, y0, y1, z);
                builder.assert_eq(result, expected);
            }
        }

        let mut rng = StdRng::seed_from_u64(0);
        let y0_val = B128::random(&mut rng);
        let y1_val = B128::random(&mut rng);
        let z_val = B128::random(&mut rng);
        let expected = extrapolate_line_packed(y0_val, y1_val, z_val);

        test_helper::<ExtrapolateLineCircuit, 4>([y0_val, y1_val, z_val, expected]).unwrap();
    }

    #[test]
    fn test_evaluate_univariate() {
        struct UnivariateCircuit;

        impl TestCircuit<5> for UnivariateCircuit {
            fn build<Builder: CircuitBuilder>(builder: &mut Builder, inout: [Builder::Wire; 5]) {
                let (coeffs, rest) = inout.split_at(3);
                let [z, expected] = rest else { unreachable!() };
                let result = evaluate_univariate(builder, coeffs, *z);
                builder.assert_eq(result, *expected);
            }
        }

        let mut rng = StdRng::seed_from_u64(0);
        let coeffs_vals = [
            B128::random(&mut rng),
            B128::random(&mut rng),
            B128::random(&mut rng),
        ];
        let z_val = B128::random(&mut rng);
        let expected = univariate::evaluate_univariate(&coeffs_vals, z_val);

        test_helper::<UnivariateCircuit, 5>([
            coeffs_vals[0],
            coeffs_vals[1],
            coeffs_vals[2],
            z_val,
            expected,
        ])
        .unwrap();
    }

    #[test]
    fn test_evaluate_multilinear() {
        struct MultilinearCircuit;

        impl TestCircuit<7> for MultilinearCircuit {
            fn build<Builder: CircuitBuilder>(builder: &mut Builder, inout: [Builder::Wire; 7]) {
                let (coeffs, rest) = inout.split_at(4);
                let (coords, expected_slice) = rest.split_at(2);
                let expected = expected_slice[0];
                let result = evaluate_multilinear(builder, coeffs, coords);
                assert_eq!(result.len(), 1);
                builder.assert_eq(result[0], expected);
            }
        }

        let mut rng = StdRng::seed_from_u64(0);
        let coeffs_vals = random_field_buffer(&mut rng, 2);
        let coords_vals = random_scalars(&mut rng, 2);
        let expected = multilinear::evaluate::evaluate(&coeffs_vals, &coords_vals);

        test_helper::<MultilinearCircuit, 7>([
            coeffs_vals[0],
            coeffs_vals[1],
            coeffs_vals[2],
            coeffs_vals[3],
            coords_vals[0],
            coords_vals[1],
            expected,
        ])
        .unwrap();
    }

    #[test]
    fn test_powers() {
        struct PowersCircuit;

        impl<const N_INOUT: usize> TestCircuit<N_INOUT> for PowersCircuit {
            fn build<Builder: CircuitBuilder>(
                builder: &mut Builder,
                inout: [Builder::Wire; N_INOUT],
            ) {
                let x = inout[0];
                let expected = &inout[1..];
                let result = powers(builder, x, N_INOUT - 1);
                assert_eq!(result.len(), expected.len());
                for (r, e) in iter::zip(&result, expected) {
                    builder.assert_eq(*r, *e);
                }
            }
        }

        let mut rng = StdRng::seed_from_u64(0);
        let x_val = B128::random(&mut rng);
        let expected_vals = binius_field::util::powers(x_val)
            .skip(1)
            .take(4)
            .collect::<Vec<_>>();

        test_helper::<PowersCircuit, 5>([
            x_val,
            expected_vals[0],
            expected_vals[1],
            expected_vals[2],
            expected_vals[3],
        ])
        .unwrap();
    }

    #[test]
    fn test_invert() {
        struct InvertCircuit;

        impl TestCircuit<2> for InvertCircuit {
            fn build<Builder: CircuitBuilder>(builder: &mut Builder, inout: [Builder::Wire; 2]) {
                let [x, expected] = inout;
                let result = invert(builder, x);
                builder.assert_eq(result, expected);
            }
        }

        let mut rng = StdRng::seed_from_u64(0);
        let x_val = B128::random(&mut rng);
        let expected = x_val.invert_or_zero();

        test_helper::<InvertCircuit, 2>([x_val, expected]).unwrap();
        assert!(test_helper::<InvertCircuit, 2>([B128::ZERO, B128::ZERO]).is_err());
    }

    #[test]
    fn test_invert_or_zero_nonzero() {
        struct InvertOrZeroCircuit;

        impl TestCircuit<2> for InvertOrZeroCircuit {
            fn build<Builder: CircuitBuilder>(builder: &mut Builder, inout: [Builder::Wire; 2]) {
                let [x, expected] = inout;
                let result = invert_or_zero(builder, x);
                builder.assert_eq(result, expected);
            }
        }

        let mut rng = StdRng::seed_from_u64(0);
        let x_val = B128::random(&mut rng);
        let expected = x_val.invert_or_zero();

        test_helper::<InvertOrZeroCircuit, 2>([x_val, expected]).unwrap();
    }

    #[test]
    fn test_invert_or_zero_zero() {
        struct InvertOrZeroCircuit;

        impl TestCircuit<2> for InvertOrZeroCircuit {
            fn build<Builder: CircuitBuilder>(builder: &mut Builder, inout: [Builder::Wire; 2]) {
                let [x, expected] = inout;
                let result = invert_or_zero(builder, x);
                builder.assert_eq(result, expected);
            }
        }

        test_helper::<InvertOrZeroCircuit, 2>([B128::ZERO, B128::ZERO]).unwrap();
    }
}
