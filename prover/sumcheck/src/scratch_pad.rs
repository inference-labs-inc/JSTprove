//! Scratch pad for prover and verifier to store intermediate values during the sumcheck protocol.

use std::cmp::max;

use arith::{ExtensionField, Field};
use circuit::Circuit;
use gkr_engine::{FieldEngine, FieldType};

#[derive(Clone, Debug, Default)]
pub struct ProverScratchPad<F: FieldEngine> {
    pub v_evals: Vec<F::Field>,
    pub hg_evals_5: Vec<F::ChallengeField>,
    pub hg_evals_1: Vec<F::ChallengeField>,
    pub hg_evals: Vec<F::Field>,
    pub simd_var_v_evals: Vec<F::ChallengeField>,
    pub simd_var_hg_evals: Vec<F::ChallengeField>,
    pub mpi_var_v_evals: Vec<F::ChallengeField>,
    pub mpi_var_hg_evals: Vec<F::ChallengeField>,

    pub eq_evals_at_rx: Vec<F::ChallengeField>,
    pub eq_evals_at_rz0: Vec<F::ChallengeField>,
    pub eq_evals_at_r_simd0: Vec<F::ChallengeField>,
    pub eq_evals_at_r_mpi0: Vec<F::ChallengeField>,
    pub eq_evals_first_half: Vec<F::ChallengeField>,
    pub eq_evals_second_half: Vec<F::ChallengeField>,

    pub gate_exists_5: Vec<bool>,
    pub gate_exists_1: Vec<bool>,

    pub phase2_coef: F::ChallengeField,
}

impl<F: FieldEngine> ProverScratchPad<F> {
    pub fn new(max_num_input_var: usize, max_num_output_var: usize, mpi_world_size: usize) -> Self {
        let max_input_size = 1 << max_num_input_var;
        let max_output_size = 1 << max_num_output_var;
        let max_io_size = max(max_input_size, max_output_size);
        ProverScratchPad {
            v_evals: vec![F::Field::default(); max_input_size],
            hg_evals_5: vec![F::ChallengeField::default(); max_input_size],
            hg_evals_1: vec![F::ChallengeField::default(); max_input_size],
            hg_evals: vec![F::Field::default(); max_input_size],
            simd_var_v_evals: vec![F::ChallengeField::default(); F::get_field_pack_size()],
            simd_var_hg_evals: vec![F::ChallengeField::default(); F::get_field_pack_size()],
            mpi_var_v_evals: vec![F::ChallengeField::default(); mpi_world_size],
            mpi_var_hg_evals: vec![F::ChallengeField::default(); mpi_world_size],

            eq_evals_at_rx: vec![F::ChallengeField::default(); max_input_size],
            eq_evals_at_rz0: vec![F::ChallengeField::default(); max_output_size],
            eq_evals_at_r_simd0: vec![F::ChallengeField::default(); F::get_field_pack_size()],
            eq_evals_at_r_mpi0: vec![F::ChallengeField::default(); mpi_world_size],
            eq_evals_first_half: vec![
                F::ChallengeField::default();
                max(
                    max(max_io_size, F::get_field_pack_size()),
                    mpi_world_size
                )
            ],
            eq_evals_second_half: vec![
                F::ChallengeField::default();
                max(
                    max(max_io_size, F::get_field_pack_size()),
                    mpi_world_size
                )
            ],

            gate_exists_5: vec![false; max_input_size],
            gate_exists_1: vec![false; max_input_size],
            phase2_coef: F::ChallengeField::ZERO,
        }
    }
}

#[derive(Clone, Debug)]
pub struct VerifierScratchPad<F: FieldEngine> {
    // ====== for evaluating cst, add and mul ======
    pub eq_evals_at_rz0: Vec<F::ChallengeField>,
    pub eq_evals_at_r_simd: Vec<F::ChallengeField>,
    pub eq_evals_at_r_mpi: Vec<F::ChallengeField>,

    pub eq_evals_at_rx: Vec<F::ChallengeField>,
    pub eq_evals_at_ry: Vec<F::ChallengeField>,

    pub eq_evals_first_part: Vec<F::ChallengeField>,
    pub eq_evals_second_part: Vec<F::ChallengeField>,

    pub r_simd: Vec<F::ChallengeField>,
    pub r_mpi: Vec<F::ChallengeField>,
    pub eq_r_simd_r_simd_xy: F::ChallengeField,
    pub eq_r_mpi_r_mpi_xy: F::ChallengeField,

    // ====== for deg2, deg3 eval ======
    pub gf2_deg2_eval_coef: F::ChallengeField, // 1 / x(x - 1)
    pub deg3_eval_at: [F::ChallengeField; 4],
    pub deg3_lag_denoms_inv: [F::ChallengeField; 4],
    // ====== for deg6 eval ======
    pub deg6_eval_at: [F::ChallengeField; 7],
    pub deg6_lag_denoms_inv: [F::ChallengeField; 7],
}

impl<F: FieldEngine> VerifierScratchPad<F> {
    pub fn new(circuit: &Circuit<F>, mpi_world_size: usize) -> Self {
        let mut max_num_var = circuit
            .layers
            .iter()
            .map(|layer| layer.output_var_num)
            .max()
            .unwrap();
        max_num_var = max(max_num_var, circuit.log_input_size());

        let max_io_size = 1usize << max_num_var;
        let simd_size = F::get_field_pack_size();

        let gf2_deg2_eval_coef = if F::FIELD_TYPE == FieldType::GF2Ext128 {
            (F::ChallengeField::X - F::ChallengeField::one())
                .mul_by_x()
                .inv()
                .unwrap()
        } else {
            F::ChallengeField::INV_2
        };

        let deg3_eval_at = if F::FIELD_TYPE == FieldType::GF2Ext128 {
            [
                F::ChallengeField::ZERO,
                F::ChallengeField::ONE,
                F::ChallengeField::X,
                F::ChallengeField::X.mul_by_x(),
            ]
        } else {
            [
                F::ChallengeField::ZERO,
                F::ChallengeField::ONE,
                F::ChallengeField::from(2),
                F::ChallengeField::from(3),
            ]
        };

        let mut deg3_lag_denoms_inv = [F::ChallengeField::ZERO; 4];
        for i in 0..4 {
            let mut denominator = F::ChallengeField::ONE;
            for j in 0..4 {
                if j == i {
                    continue;
                }
                denominator *= deg3_eval_at[i] - deg3_eval_at[j];
            }
            deg3_lag_denoms_inv[i] = denominator.inv().unwrap();
        }

        let deg6_eval_at = [
            F::ChallengeField::ZERO,
            F::ChallengeField::ONE,
            F::ChallengeField::from(2),
            F::ChallengeField::from(3),
            F::ChallengeField::from(4),
            F::ChallengeField::from(5),
            F::ChallengeField::from(6),
        ];

        let mut deg6_lag_denoms_inv = [F::ChallengeField::ZERO; 7];
        for i in 0..7 {
            let mut denominator = F::ChallengeField::ONE;
            for j in 0..7 {
                if j == i {
                    continue;
                }
                denominator *= deg6_eval_at[i] - deg6_eval_at[j];
            }
            deg6_lag_denoms_inv[i] = denominator.inv().unwrap();
        }

        Self {
            eq_evals_at_rz0: vec![F::ChallengeField::zero(); max_io_size],
            eq_evals_at_r_simd: vec![F::ChallengeField::zero(); simd_size],
            eq_evals_at_r_mpi: vec![F::ChallengeField::zero(); mpi_world_size],

            eq_evals_at_rx: vec![F::ChallengeField::zero(); max_io_size],
            eq_evals_at_ry: vec![F::ChallengeField::zero(); max_io_size],

            eq_evals_first_part: vec![
                F::ChallengeField::zero();
                max(max(max_io_size, simd_size), mpi_world_size)
            ],
            eq_evals_second_part: vec![
                F::ChallengeField::zero();
                max(max(max_io_size, simd_size), mpi_world_size)
            ],

            r_simd: vec![],
            r_mpi: vec![],
            eq_r_simd_r_simd_xy: F::ChallengeField::zero(),
            eq_r_mpi_r_mpi_xy: F::ChallengeField::zero(),

            gf2_deg2_eval_coef,
            deg3_eval_at,
            deg3_lag_denoms_inv,
            deg6_eval_at,
            deg6_lag_denoms_inv,
        }
    }
}
