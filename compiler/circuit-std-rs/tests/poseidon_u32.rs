use circuit_std_rs::{poseidon::poseidon_u32::*, poseidon::utils::*};

#[test]
// NOTE(HS) Poseidon Mersenne-31 Width-16 Sponge tested over input length 8
fn test_poseidon_u32x16_hash_to_state_input_len8() {
    let inputs = [114514; 8];
    let outputs = [
        1021105124, 1342990709, 1593716396, 2100280498, 330652568, 1371365483, 586650367,
        345482939, 849034538, 175601510, 1454280121, 1362077584, 528171622, 187534772, 436020341,
        1441052621,
    ];

    let params = PoseidonParams::new(
        POSEIDON_M31X16_RATE,
        16,
        POSEIDON_M31X16_FULL_ROUNDS,
        POSEIDON_M31X16_PARTIAL_ROUNDS,
    );
    let res = params.hash_to_state(&inputs);
    (0..params.width).for_each(|i| assert_eq!(res[i], outputs[i]));
}
