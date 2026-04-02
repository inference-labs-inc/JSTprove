// Copyright 2025 Irreducible Inc.
use binius_core::word::Word;

use crate::compiler::gate;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Opcode {
    // Bitwise operations
    Band,
    Bxor,
    BxorMulti,
    Bor,
    Fax,

    // Selection
    Select,

    // Arithmetic
    IaddCinCout,
    Iadd32,
    IsubBinBout,
    Imul,
    Smul,

    // Shifts
    Shl,
    Sll32,
    Shr,
    Srl32,
    Sar,
    Sra32,
    Rotr,
    Rotr32,

    // Comparisons
    IcmpUlt,
    IcmpEq,

    // Assertions
    AssertEq,
    AssertZero,
    AssertNonZero,
    AssertFalse,
    AssertTrue,
    AssertEqCond,

    // Hints
    BigUintDivideHint,
    BigUintModPowHint,
    ModInverseHint,
    Secp256k1EndosplitHint,
}

/// The shape of an opcode is a description of it's inputs and outputs. It allows treating a gate as
/// a black box, correctly identifying its inputs or outputs.
pub struct OpcodeShape {
    /// The constants the gate with this opcode expects.
    pub const_in: &'static [Word],
    /// The number of inputs this opcode expects.
    ///
    /// In case this opcode has a dynamic shape, it specifies the fixed number of inputs.
    pub n_in: usize,
    /// The number of outputs this opcode provides.
    ///
    /// In case this opcode has a dynamic shape, it specifies the fixed number of outputs.
    pub n_out: usize,
    /// The number of wires of aux wires.
    ///
    /// Aux wires are neither inputs nor outputs, but are still being used within constraint
    /// system.
    ///
    /// In case this opcode has a dynamic shape, it specifies the fixed number of aux wires.
    pub n_aux: usize,
    /// The number of scratch wires.
    ///
    /// Scratch wires are the wires that are neither inputs nor outputs. They also do not
    /// get referenced in the constraint system. Those are only needed for the witness evaluation.
    ///
    /// In case this opcode has a dynamic shape, it specifies the fixed number of scratch wires.
    pub n_scratch: usize,
    /// The number of immediate operands.
    ///
    /// Those are the fixed constant parameters for the opcode. Those include the constant shift
    /// amounts and things like that.
    ///
    /// In case this opcode has a dynamic shape, it specifies the fixed number of immediates.
    pub n_imm: usize,
}

impl Opcode {
    pub fn shape(&self, dimensions: &[usize]) -> OpcodeShape {
        assert_eq!(self.is_const_shape(), dimensions.is_empty());

        match self {
            // Bitwise operations
            Opcode::Band => gate::band::shape(),
            Opcode::Bxor => gate::bxor::shape(),
            Opcode::BxorMulti => gate::bxor_multi::shape(dimensions),
            Opcode::Bor => gate::bor::shape(),
            Opcode::Fax => gate::fax::shape(),

            // Selection
            Opcode::Select => gate::select::shape(),

            // Arithmetic
            Opcode::IaddCinCout => gate::iadd_cin_cout::shape(),
            Opcode::Iadd32 => gate::iadd32::shape(),
            Opcode::IsubBinBout => gate::isub_bin_bout::shape(),
            Opcode::Imul => gate::imul::shape(),
            Opcode::Smul => gate::smul::shape(),

            // Shifts
            Opcode::Shr => gate::shr::shape(),
            Opcode::Shl => gate::shl::shape(),
            Opcode::Sll32 => gate::sll32::shape(),
            Opcode::Sar => gate::sar::shape(),
            Opcode::Srl32 => gate::srl32::shape(),
            Opcode::Sra32 => gate::sra32::shape(),
            Opcode::Rotr => gate::rotr::shape(),
            Opcode::Rotr32 => gate::rotr32::shape(),

            // Comparisons
            Opcode::IcmpUlt => gate::icmp_ult::shape(),
            Opcode::IcmpEq => gate::icmp_eq::shape(),

            // Assertions (no outputs)
            Opcode::AssertEq => gate::assert_eq::shape(),
            Opcode::AssertZero => gate::assert_zero::shape(),
            Opcode::AssertNonZero => gate::assert_non_zero::shape(),
            Opcode::AssertFalse => gate::assert_false::shape(),
            Opcode::AssertTrue => gate::assert_true::shape(),
            Opcode::AssertEqCond => gate::assert_eq_cond::shape(),

            // Hints (no constraints)
            Opcode::BigUintDivideHint => gate::biguint_divide_hint::shape(dimensions),
            Opcode::BigUintModPowHint => gate::biguint_mod_pow_hint::shape(dimensions),
            Opcode::ModInverseHint => gate::mod_inverse_hint::shape(dimensions),
            Opcode::Secp256k1EndosplitHint => gate::secp256k1_endosplit_hint::shape(),
        }
    }

    pub fn is_const_shape(&self) -> bool {
        #[allow(clippy::match_like_matches_macro)]
        match self {
            Opcode::BigUintDivideHint => false,
            Opcode::BigUintModPowHint => false,
            Opcode::ModInverseHint => false,
            Opcode::BxorMulti => false,
            _ => true,
        }
    }
}
