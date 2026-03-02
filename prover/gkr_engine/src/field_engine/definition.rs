use std::cmp;
use std::fmt::Debug;
use std::ops::{Add, Mul};

use arith::{ExtensionField, Field, SimdField};
use polynomials::MultiLinearPoly;

use crate::{ExpanderSingleVarChallenge, MPIEngine};

#[derive(Debug, Clone, PartialEq, Default)]
pub enum FieldType {
    #[default]
    M31x16, // M31Ext3x16
    M31x1,        // M31Ext3x1
    BN254,        // BN254 scalar field
    GF2Ext128,    // GF2Ext128x8
    Goldilocksx8, // GoldilocksExt2x8
    Goldilocksx1, // GoldilocksExt2x1
    BabyBearx16,  // BabyBearExt3x16
}

pub trait FieldEngine: Default + Debug + Clone + Send + Sync + PartialEq + 'static {
    /// Enum type for Self::Field
    const FIELD_TYPE: FieldType;

    /// Sentinel value for the field type; this is the order of the field
    const SENTINEL: [u8; 32];

    /// Field type for the circuit, e.g., M31
    type CircuitField: Field + Send + Sync;

    /// Field type for the challenge, e.g., M31Ext3
    type ChallengeField: ExtensionField<BaseField = Self::CircuitField>
        + Send
        + Sync
        + Mul<Self::CircuitField, Output = Self::ChallengeField>
        + Mul<Self::SimdCircuitField, Output = Self::Field>;

    /// Main field type for the scheme, e.g., M31Ext3x16
    type Field: ExtensionField<BaseField = Self::SimdCircuitField>
        + Mul<Self::CircuitField, Output = Self::Field>
        + Add<Self::CircuitField, Output = Self::Field>
        + Mul<Self::SimdCircuitField, Output = Self::Field>
        + Add<Self::SimdCircuitField, Output = Self::Field>
        + Mul<Self::ChallengeField, Output = Self::Field>
        + From<Self::SimdCircuitField>
        + SimdField<Scalar = Self::ChallengeField>
        + Send
        + Sync;

    /// Simd field for circuit, e.g., M31x16
    type SimdCircuitField: SimdField<Scalar = Self::CircuitField>
        + Send
        + Sync
        + Mul<Self::CircuitField, Output = Self::SimdCircuitField>
        + From<Self::CircuitField>;

    /// The pack size for the simd circuit field, e.g., 16 for M31x16
    fn get_field_pack_size() -> usize {
        Self::SimdCircuitField::PACK_SIZE
    }

    /// Some dedicated mle implementations for FieldEngine
    /// Take into consideration the simd challenge and the mpi challenge
    ///
    /// This is more efficient than the generic implementation by avoiding
    /// unnecessary conversions between field types
    #[inline]
    fn eval_circuit_vals_at_challenge(
        evals: &[Self::SimdCircuitField],
        x: &[Self::ChallengeField],
        scratch: &mut [Self::Field],
    ) -> Self::Field {
        assert_eq!(1 << x.len(), evals.len());
        assert!(scratch.len() >= evals.len());

        if x.is_empty() {
            evals[0].into()
        } else {
            for i in 0..(evals.len() >> 1) {
                scratch[i] = x[0] * (evals[i * 2 + 1] - evals[i * 2]) + evals[i * 2];
            }

            let mut cur_eval_size = evals.len() >> 2;
            for r in x.iter().skip(1) {
                for i in 0..cur_eval_size {
                    scratch[i] = scratch[i * 2] + (scratch[i * 2 + 1] - scratch[i * 2]).scale(r);
                }
                cur_eval_size >>= 1;
            }
            scratch[0]
        }
    }

    /// This assumes each mpi core hold their own evals, and collectively
    /// compute the global evaluation.
    /// Mostly used by the prover run with `mpiexec`
    #[inline]
    fn collectively_eval_circuit_vals_at_expander_challenge(
        local_evals: &[Self::SimdCircuitField],
        challenge: &ExpanderSingleVarChallenge<Self>,

        // x: &[Self::ChallengeField],
        // x_simd: &[Self::ChallengeField],
        // x_mpi: &[Self::ChallengeField],
        scratch_field: &mut [Self::Field],
        scratch_challenge_field: &mut [Self::ChallengeField],
        mpi_config: &impl MPIEngine,
    ) -> Self::ChallengeField {
        assert!(
            scratch_challenge_field.len()
                >= 1 << cmp::max(challenge.r_simd.len(), challenge.r_mpi.len())
        );

        let local_simd =
            Self::eval_circuit_vals_at_challenge(local_evals, &challenge.rz, scratch_field);
        let local_simd_unpacked = local_simd.unpack();
        let local_v = MultiLinearPoly::evaluate_with_buffer(
            &local_simd_unpacked,
            &challenge.r_simd,
            scratch_challenge_field,
        );

        if mpi_config.is_root() {
            let mut claimed_v_gathering_buffer =
                vec![Self::ChallengeField::zero(); mpi_config.world_size()];
            mpi_config.gather_vec(&[local_v], &mut claimed_v_gathering_buffer);
            MultiLinearPoly::evaluate_with_buffer(
                &claimed_v_gathering_buffer,
                &challenge.r_mpi,
                scratch_challenge_field,
            )
        } else {
            mpi_config.gather_vec(&[local_v], &mut vec![]);
            Self::ChallengeField::zero()
        }
    }

    /// This assumes only a single core holds all the evals, and evaluate it locally
    /// mostly used by the verifier
    #[inline]
    fn single_core_eval_circuit_vals_at_expander_challenge(
        global_vals: &[Self::SimdCircuitField],
        challenge: &ExpanderSingleVarChallenge<Self>,
    ) -> Self::ChallengeField {
        let local_poly_size = global_vals.len() >> challenge.r_mpi.len();
        assert_eq!(local_poly_size, 1 << challenge.rz.len());

        let mut scratch_field = vec![Self::Field::default(); local_poly_size];
        let mut scratch_challenge_field =
            vec![
                Self::ChallengeField::default();
                1 << cmp::max(challenge.r_simd.len(), challenge.r_mpi.len())
            ];
        let local_evals = global_vals
            .chunks(local_poly_size)
            .map(|local_vals| {
                let local_simd = Self::eval_circuit_vals_at_challenge(
                    local_vals,
                    &challenge.rz,
                    &mut scratch_field,
                );
                let local_simd_unpacked = local_simd.unpack();
                MultiLinearPoly::evaluate_with_buffer(
                    &local_simd_unpacked,
                    &challenge.r_simd,
                    &mut scratch_challenge_field,
                )
            })
            .collect::<Vec<Self::ChallengeField>>();

        let mut scratch = vec![Self::ChallengeField::default(); local_evals.len()];
        MultiLinearPoly::evaluate_with_buffer(&local_evals, &challenge.r_mpi, &mut scratch)
    }
}
