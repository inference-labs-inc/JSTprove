use circuit_std_rs::logup::LogUpSingleKeyTable;
use ethnum::U256;
use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

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

    pub fn query(&mut self, x: Variable, y: Variable) {
        self.table.query(x, vec![y]);
    }

    pub fn finalize<C: Config, B: RootAPI<C>>(&mut self, api: &mut B) {
        self.table.final_check::<C, B>(api);
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
}
