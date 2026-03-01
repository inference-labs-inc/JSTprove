use expander_compiler::frontend::{BN254Config, RootAPI, Variable};
use halo2curves::bn256::Fr;

pub(crate) fn unconstrained_byte_decomposition<Builder: RootAPI<BN254Config>>(
    x: &Variable,
    builder: &mut Builder,
) -> Vec<Variable> {
    let mut res = vec![];
    let mut x = *x;
    let u8modulus = builder.constant(1 << 8);
    for _ in 0..256 / 8 {
        let byte = builder.unconstrained_mod(x, u8modulus);
        x = builder.unconstrained_int_div(x, u8modulus);

        res.push(byte);
    }
    res
}

// assert bit decomposition
// the constant_scalars are 2^8, 2^16, ... 2^248
pub fn byte_decomposition<Builder: RootAPI<BN254Config>>(
    x: &Variable,
    constant_scalars: &[Variable],
    builder: &mut Builder,
) -> Vec<Variable> {
    let bytes = unconstrained_byte_decomposition(x, builder);
    // todo: constraint each byte to be less than 256 via logup
    let inner_product = bytes.iter().zip(constant_scalars.iter()).fold(
        builder.constant(Fr::zero()),
        |acc, (byte, scalar)| {
            let product = builder.mul(byte, scalar);
            builder.add(acc, product)
        },
    );

    builder.assert_is_equal(x, inner_product);
    bytes
}
