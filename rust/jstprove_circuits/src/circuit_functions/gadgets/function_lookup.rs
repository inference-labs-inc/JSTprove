use circuit_std_rs::logup::LogUpSingleKeyTable;
use ethnum::U256;
use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError, gadgets::range_check::LogupRangeCheckContext, hints::exp::EXP_HINT_KEY,
};

pub const MAX_FUNCTION_LOOKUP_BITS: usize = 24;

#[must_use]
pub fn function_lookup_bits(scale_exponent: u32) -> usize {
    let ideal = (scale_exponent as usize) + 6;
    ideal.min(MAX_FUNCTION_LOOKUP_BITS)
}

#[must_use]
#[allow(clippy::cast_sign_loss)]
pub fn i64_to_field<C: Config>(x: i64) -> CircuitField<C> {
    if x >= 0 {
        CircuitField::<C>::from_u256(U256::from(x as u64))
    } else {
        let mag = U256::from(x.unsigned_abs());
        CircuitField::<C>::from_u256(CircuitField::<C>::MODULUS - mag)
    }
}

pub struct FunctionLookupTable {
    table: LogUpSingleKeyTable,
}

impl FunctionLookupTable {
    #[allow(clippy::cast_possible_wrap)]
    pub fn build_signed<C: Config, B: RootAPI<C>>(
        api: &mut B,
        f: impl Fn(i64, u64) -> i64,
        n_bits: usize,
        scale: u64,
    ) -> Self {
        let half = 1i64 << (n_bits - 1);
        let table_size = 1usize << n_bits;

        let mut keys = Vec::with_capacity(table_size);
        let mut values = Vec::with_capacity(table_size);

        for i in 0..table_size {
            let x_q: i64 = (i as i64) - half;
            let y_q = f(x_q, scale);

            keys.push(api.constant(i64_to_field::<C>(x_q)));
            values.push(vec![api.constant(i64_to_field::<C>(y_q))]);
        }

        let mut table = LogUpSingleKeyTable::new(0);
        table.new_table(keys, values);

        Self { table }
    }

    fn build_unsigned<C: Config, B: RootAPI<C>>(
        api: &mut B,
        f: impl Fn(i64, u64) -> i64,
        n_bits: usize,
        scale: u64,
    ) -> Self {
        let table_size = 1usize << n_bits;
        let mut keys = Vec::with_capacity(table_size);
        let mut values = Vec::with_capacity(table_size);

        for i in 0..table_size {
            #[allow(clippy::cast_possible_wrap)]
            let y_q = f(i as i64, scale);
            keys.push(api.constant(CircuitField::<C>::from_u256(U256::from(i as u64))));
            values.push(vec![api.constant(i64_to_field::<C>(y_q))]);
        }

        let mut table = LogUpSingleKeyTable::new(0);
        table.new_table(keys, values);
        Self { table }
    }

    pub fn query(&mut self, x: Variable, y: Variable) {
        self.table.query(x, vec![y]);
    }

    pub fn finalize<C: Config, B: RootAPI<C>>(&mut self, api: &mut B) {
        self.table.final_check::<C, B>(api);
    }
}

pub struct DecomposedExpLookup {
    high_table: FunctionLookupTable,
    low_table: FunctionLookupTable,
    chunk_bits: usize,
    scale_bits: usize,
    half_var: Variable,
    chunk_var: Variable,
    scale_var: Variable,
    high_offset_var: Variable,
}

impl DecomposedExpLookup {
    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    pub fn build<C: Config, B: RootAPI<C>>(
        api: &mut B,
        total_bits: usize,
        scale: u64,
        compute_exp: impl Fn(i64, u64) -> i64 + Copy,
    ) -> Self {
        let chunk_bits = total_bits / 2;
        let high_bits = total_bits - chunk_bits;
        let chunk = 1i64 << chunk_bits;

        let high_table = FunctionLookupTable::build_signed::<C, B>(
            api,
            move |x_high, s| compute_exp(x_high.saturating_mul(chunk), s),
            high_bits,
            scale,
        );

        let low_table =
            FunctionLookupTable::build_unsigned::<C, B>(api, compute_exp, chunk_bits, scale);

        let half_var = api.constant(CircuitField::<C>::from_u256(U256::from(
            1u64 << (total_bits - 1),
        )));
        let chunk_var = api.constant(CircuitField::<C>::from_u256(U256::from(1u64 << chunk_bits)));
        let scale_var = api.constant(CircuitField::<C>::from_u256(U256::from(scale)));
        let high_offset_var = api.constant(CircuitField::<C>::from_u256(U256::from(
            1u64 << (high_bits - 1),
        )));

        let scale_bits = 64 - scale.leading_zeros() as usize;

        Self {
            high_table,
            low_table,
            chunk_bits,
            scale_bits,
            half_var,
            chunk_var,
            scale_var,
            high_offset_var,
        }
    }

    /// # Errors
    /// Returns `CircuitError` if any range check fails.
    pub fn verify_exp<C: Config, B: RootAPI<C>>(
        &mut self,
        api: &mut B,
        logup_ctx: &mut LogupRangeCheckContext,
        x: Variable,
    ) -> Result<Variable, CircuitError> {
        let x_shifted = api.add(x, self.half_var);

        let x_high_u = api.unconstrained_int_div(x_shifted, self.chunk_var);
        let x_low = api.unconstrained_mod(x_shifted, self.chunk_var);

        let recon = api.mul(x_high_u, self.chunk_var);
        let recon = api.add(recon, x_low);
        api.assert_is_equal(recon, x_shifted);

        logup_ctx.range_check::<C, B>(api, x_low, self.chunk_bits)?;

        let x_high = api.sub(x_high_u, self.high_offset_var);

        let x_high_times_chunk = api.mul(x_high, self.chunk_var);
        let exp_high_hint = api.new_hint(EXP_HINT_KEY, &[x_high_times_chunk, self.scale_var], 1);
        let exp_high = exp_high_hint[0];

        let exp_low_hint = api.new_hint(EXP_HINT_KEY, &[x_low, self.scale_var], 1);
        let exp_low = exp_low_hint[0];

        self.high_table.query(x_high, exp_high);
        self.low_table.query(x_low, exp_low);

        let product = api.mul(exp_high, exp_low);

        let y = api.unconstrained_int_div(product, self.scale_var);
        let remainder = api.unconstrained_mod(product, self.scale_var);
        let y_recon = api.mul(y, self.scale_var);
        let y_recon = api.add(y_recon, remainder);
        api.assert_is_equal(y_recon, product);
        logup_ctx.range_check::<C, B>(api, remainder, self.scale_bits)?;

        Ok(y)
    }

    pub fn finalize<C: Config, B: RootAPI<C>>(&mut self, api: &mut B) {
        self.high_table.finalize::<C, B>(api);
        self.low_table.finalize::<C, B>(api);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use expander_compiler::field::BN254Fr;

    type F = BN254Fr;

    #[test]
    fn function_lookup_bits_caps_at_max() {
        assert_eq!(function_lookup_bits(20), 24);
        assert_eq!(function_lookup_bits(30), 24);
    }

    #[test]
    fn function_lookup_bits_uses_ideal_when_small() {
        assert_eq!(function_lookup_bits(12), 18);
        assert_eq!(function_lookup_bits(8), 14);
    }

    #[test]
    fn i64_to_field_positive() {
        let f: F = i64_to_field::<expander_compiler::frontend::BN254Config>(42);
        assert_eq!(f.to_u256(), U256::from(42u64));
    }

    #[test]
    fn i64_to_field_zero() {
        let f: F = i64_to_field::<expander_compiler::frontend::BN254Config>(0);
        assert_eq!(f.to_u256(), U256::from(0u64));
    }

    #[test]
    fn i64_to_field_negative_roundtrips() {
        let f: F = i64_to_field::<expander_compiler::frontend::BN254Config>(-1);
        assert_eq!(f.to_u256(), F::MODULUS - U256::from(1u64));
    }

    #[test]
    fn i64_to_field_negative_large() {
        let val = -1_000_000i64;
        let f: F = i64_to_field::<expander_compiler::frontend::BN254Config>(val);
        let expected = F::MODULUS - U256::from(val.unsigned_abs());
        assert_eq!(f.to_u256(), expected);
    }

    #[test]
    fn decomposed_table_sizes() {
        let total = 24;
        let chunk = total / 2;
        let high_bits = total - chunk;
        assert_eq!(1usize << high_bits, 4096);
        assert_eq!(1usize << chunk, 4096);
    }
}
