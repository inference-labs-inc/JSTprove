// Copyright 2025 Irreducible Inc.

use binius_field::BinaryField128bGhash as Ghash;

pub const M: usize = 4;

pub const BYTES_PER_GHASH: usize = 16;

pub const NUM_ROUNDS: usize = 8;

pub static B_FWD_COEFFS: [Ghash; 4] = [
    Ghash::new(0xb0b849b207a0f1c74c29e4d892ca33dd),
    Ghash::new(0xf2240891aac3c3a5855eeb8ce24c9523),
    Ghash::new(0xb017e96797ef1fbe9d79908388dd768d),
    Ghash::new(0x2bf9f0b94c2b2ceb00dbb86a2cb472bb),
];

pub static B_INV_COEFFS: [Ghash; 1 + 128] = [
    Ghash::new(0x1d124529af098017d515f025b2107a04),
    Ghash::new(0xb2733aa187f5954a397b9985012327dc),
    Ghash::new(0xf9ed22a697f45e3fac28266a3c9c8bf9),
    Ghash::new(0xb2563578203d5df23ed37df5274f9b62),
    Ghash::new(0x9f60ca3b6e570f9bb35f672c4cc71e78),
    Ghash::new(0x2b7839e6c8c6f35c4cfe4dd6ed9988bf),
    Ghash::new(0xd7e470d53c8a1441335c0e39f8e825ba),
    Ghash::new(0xc2a0d87c2f6de7f3656d51ee118d93ca),
    Ghash::new(0xb80cb728430ce418d8763ad073d43c72),
    Ghash::new(0xf1424d16f8ff5ad20c172fa0bce84a66),
    Ghash::new(0xf27af010f636dd9d8b47d92989d38c91),
    Ghash::new(0xeebb7636921fdc8bb35172c4f2904488),
    Ghash::new(0xf17407875a9069991c532b85716f8278),
    Ghash::new(0x72032c6e2ef39011b80918ec4884682e),
    Ghash::new(0xa93dd016787f32dafb792593a1c0664c),
    Ghash::new(0x2a72f7f35410abe48bf86284d7153568),
    Ghash::new(0x539206fa82bb0fd3fc257acfd85198cf),
    Ghash::new(0x6325125ff725e1435f66efa328075c4b),
    Ghash::new(0x73abb6e9157b3791bc1a05a15a0ba8e2),
    Ghash::new(0x3447131e361d3a430e26535b49f37884),
    Ghash::new(0xeeeb383e8b2e366b3b5d33b416320285),
    Ghash::new(0xa6ef061627c8b7d6b5ea5ddf3b3ceeee),
    Ghash::new(0x3f97a996898ed33fb4c0b023284209b7),
    Ghash::new(0xbdf02af9a8b31ed299580c95f214d5a3),
    Ghash::new(0xfded88f14050f0fc196849074ce0f9ad),
    Ghash::new(0x6c8b4d18e1ba055c6b631b36bde8d60c),
    Ghash::new(0x1027cad9488ab75d6741c4f5b9026779),
    Ghash::new(0x9f1c5193f4762dcadac6b8979fd40e1d),
    Ghash::new(0x7af3e53f8f00a454d215ee466e50388c),
    Ghash::new(0x5e3227be2a589427f2870f655f41ef5b),
    Ghash::new(0xd964ed03293469753b504cb624c60228),
    Ghash::new(0x43c3a9698d3135080fe4f57716b03036),
    Ghash::new(0x922b2fb49b13518ec3895e0eb37085b0),
    Ghash::new(0xaa09df0da87a0fceb212105e5b297b5a),
    Ghash::new(0x790c7ca5f5905488e82bd9d875c8715a),
    Ghash::new(0x369411c407a9ef883133bc16aceb21a2),
    Ghash::new(0xd577819f72b6158c1eca19e57d2e1e95),
    Ghash::new(0xac4d176698af651f201efd2cfea05ab6),
    Ghash::new(0x268f4c40a4553eb3a2afbde2b10df863),
    Ghash::new(0x8749898084983a777debda3686b5e752),
    Ghash::new(0xc723199815bae544773110e7054256a0),
    Ghash::new(0x960b23bc674dd468d75ff82b164067b4),
    Ghash::new(0xb94130b0a65d36009150e505b5e4a818),
    Ghash::new(0x742b2c52bba058cc8c9a364c0985136b),
    Ghash::new(0xdf298fbb15e6e58f9edb384f5c34673c),
    Ghash::new(0x4fdbc95e71fe2bdbcc6bee273caad844),
    Ghash::new(0x19d487689fe24e3a3eceabdefdcef9e5),
    Ghash::new(0xb01bf42d575ca9a0df60e3e99d4a4450),
    Ghash::new(0xe8a777b48d74755f5091a9d460f80660),
    Ghash::new(0x9f435ce2ff626ba06cb1ee4478a1356f),
    Ghash::new(0x6804034746f7cca00180116ebfe5b33a),
    Ghash::new(0xaef4cc5aa39d8f6957c15f9b173601eb),
    Ghash::new(0x97d15c1e09df82180a062c5268804182),
    Ghash::new(0x3109fb44aa8caba5fb41f60a49ab9f99),
    Ghash::new(0x67ff3e7094682c2c7ae9258280b00f7d),
    Ghash::new(0xbf6a8f87c80acf09f9456b27b9c44a04),
    Ghash::new(0x4cd2646e793fd4eab1319cb89fb72054),
    Ghash::new(0x076e92092b1f43709e7b9b2856ed53a9),
    Ghash::new(0xb987929f99110f41f4f44c08b778e929),
    Ghash::new(0x09b1c58023e9c9c267451053eed4b989),
    Ghash::new(0xb27b2fa5789c2749058ebed7e489d0a1),
    Ghash::new(0xcf9a22da173d17c475c43a59ed45535f),
    Ghash::new(0x77df29bf03c18bbd103431cf69e35840),
    Ghash::new(0x2a746d8aec21318be6985ff6dfa5d116),
    Ghash::new(0x3411ccb789ac3dfc5bbd1c8b967559f6),
    Ghash::new(0xd6faab5c3e02915dc279b22dbbf81919),
    Ghash::new(0x2b9a29fd7e4775061fbb90c2033f304d),
    Ghash::new(0x8a78a5986d399576aade221e1a4a4759),
    Ghash::new(0x8cc9cb6657270a712b2c776b7f231214),
    Ghash::new(0xa4206fef28f9fad8b05707b515fa176b),
    Ghash::new(0x71984ef63a6a49ddefa6b9cb6b8832c4),
    Ghash::new(0x2347fdcc33e95ee2a1bb2c4edc8da6a9),
    Ghash::new(0xc63f0aaf33ca8b99a03decedc48b2e94),
    Ghash::new(0x3237cb90cca2283888cb5729283a8b4d),
    Ghash::new(0xbe8d8bbd445bf3e3e2947257afff2516),
    Ghash::new(0xe7524df886b9fd83a2d4bdcfdb252124),
    Ghash::new(0x6cb5512f7bdd181f675d059cf05799ff),
    Ghash::new(0xc56e3f62ddedbabb35a200087a5640ce),
    Ghash::new(0xba0a5c1d5ce94e53f9efb731c6634b44),
    Ghash::new(0x4280ae86e53b14eb0202f4c147a6d14a),
    Ghash::new(0xb4dffae679af530c397b2d07ac62d79a),
    Ghash::new(0xd0311963a7fafcafd3660af96cc461e1),
    Ghash::new(0x7d4f16aea2ce0d4e4c8c545d4d0040dc),
    Ghash::new(0xda8304eff41a3ae357f1ec94fe9e062b),
    Ghash::new(0x1b7c9a9f3666986912070ec52f99f801),
    Ghash::new(0x004ccc9ddf6ca951f20c1b0ac31fa008),
    Ghash::new(0x14f6a26f68c0ffd76f8f5fd65aa79542),
    Ghash::new(0xd31406666953d392634f0e80af43fe0e),
    Ghash::new(0x020c264e630f8bf516200218fb1d1a0e),
    Ghash::new(0xdd5a2efdde3cae83465d1315d7af0852),
    Ghash::new(0x055702e27b6a93676389c452c5e4b41a),
    Ghash::new(0xc4684448e291abeeaa305e6c0e6af873),
    Ghash::new(0xd2c1419f07a837e617a10b6ac6d1f275),
    Ghash::new(0x8c605a36b3ecdd16edfc7c499e550871),
    Ghash::new(0xe034129a3ccbfd6590ebaa430dcad339),
    Ghash::new(0x5cb6429235829cdcdfc6a221725c755c),
    Ghash::new(0x5737eb97111ca259d691ae13f5763345),
    Ghash::new(0x755cd247d4267e960075f86302a39702),
    Ghash::new(0xb12a9aa970a91097fa8ad2f2bdfb1b20),
    Ghash::new(0x0dc789e36c5547c0571e479d7fd9a75f),
    Ghash::new(0x919900cb8cbe4e756f19ab79df7196ee),
    Ghash::new(0xbebfc69ba65f89dfecd241385602e25f),
    Ghash::new(0xcac2967113753aaa7f3d8b4c9df4b9aa),
    Ghash::new(0xf97eff95e366110b9b47509e730fd04f),
    Ghash::new(0xf939cdd3b4d28d6f3e30aeb60f0346a0),
    Ghash::new(0x982a7c68b7e0ae1603e324d5453d2b1d),
    Ghash::new(0x1a589c0f209aeca97224c0c9db79142b),
    Ghash::new(0x5219c08df0734e2b98dd90d13bc83d8f),
    Ghash::new(0x95f1c0b71be50da58a8f752a0724e53b),
    Ghash::new(0x8c494d44cb21e64e58401414631f3666),
    Ghash::new(0x07b5f194eaa7cf3c96287d6aaedd3051),
    Ghash::new(0x12e5008f6b6b3dd1d640a0a72a6c6d42),
    Ghash::new(0x3329d9133fa80c1e6aba028e9255256c),
    Ghash::new(0xb859b9f70c7d3da6cb8bc0127316f7ad),
    Ghash::new(0xc31f1a1e3e98194053a0c5759aba5125),
    Ghash::new(0x6f7deb6cd260c52090fa62f4b170cc55),
    Ghash::new(0x991089b8d80963ddcf856990b5b55a89),
    Ghash::new(0x36ebdacdc017e6cd38a1ccedfa6799d6),
    Ghash::new(0x6ee345c84ddb46a42c8139c52ef87729),
    Ghash::new(0x31bc991954d18dfb00a0e9f1c8177027),
    Ghash::new(0x576056edf9e7f6f349740dffa8adc7d5),
    Ghash::new(0xd7861327edc091fa6b27b5447b05255e),
    Ghash::new(0x1ad33492132898f582444df642286e8c),
    Ghash::new(0xfabef4a4b044b301e6d8a3ed7fd754a2),
    Ghash::new(0x03dbd4cca837cf33b42622b75fec035a),
    Ghash::new(0xbb8da06ec7ec3a9c2867a57e2cde5cc6),
    Ghash::new(0x7f0dfc7829c9ecef117778448e04cc6c),
    Ghash::new(0x128da4c873150f44e7e43bf88921c090),
    Ghash::new(0xd9bc6edaf1b4a689ba169b5554228af8),
];

pub static ROUND_CONSTANTS: [[Ghash; 4]; 1 + 2 * 8] = [
    [
        Ghash::new(0xd0e015190f5e0795f4a1d28234ffdf87),
        Ghash::new(0x6111731acd9a89f6b93ec3a23ec7681b),
        Ghash::new(0x15da8de707ee3f3918a34a96728b8d29),
        Ghash::new(0x2ea920e89fbb13a1ed2a216b1d232bfb),
    ],
    [
        Ghash::new(0x52c94b04b178f80fa586c37c798171a3),
        Ghash::new(0xf0e02cc3459757dc8ec9259341831bcb),
        Ghash::new(0x025c53bf6d6dee09307a55c008868354),
        Ghash::new(0x54258ece50ceb1e9b5a8e38e7bbcaa3d),
    ],
    [
        Ghash::new(0x77c332c3ebbfbaa73bffaec318055661),
        Ghash::new(0x5abbaeac51c6925d2aff59f11d164328),
        Ghash::new(0x9f8daadefaebb59d77ad096fb56e6462),
        Ghash::new(0x322de9fd29fb135d569895b07925cacf),
    ],
    [
        Ghash::new(0x95c521abe74b013381778cf1b46e6cbd),
        Ghash::new(0x85fcfe5a641b67ada8700b95fd906f70),
        Ghash::new(0x5d61e0714a000a6d491d8e0db8835a16),
        Ghash::new(0x3bea1f3ff2c984d47f91ec35358f1e16),
    ],
    [
        Ghash::new(0x66b6cd1b70d97c63163197d429553ad3),
        Ghash::new(0xd1be1516a276fb5f590b628b4f8db8b9),
        Ghash::new(0x60f3fccd56bfad8cb506e2f1adce50af),
        Ghash::new(0x205a1e29bd8a90dac5f776c450e88d10),
    ],
    [
        Ghash::new(0x2a02ee3e61560ba34c286e943b3d169f),
        Ghash::new(0x73e4bfe44495bfbb54f6fd20adedcf8b),
        Ghash::new(0x23e8322b8a8cf1d3b681c1b577cee348),
        Ghash::new(0x0d892aeb52c9b2d7096f9698ef0951df),
    ],
    [
        Ghash::new(0x8571aa085a872bfcd96c3a1ebbe26682),
        Ghash::new(0x3ae1e8efdff55ee0cb6d91af315d614b),
        Ghash::new(0xdcf3ec9dd466e76de5f9f9e9fd5dbdba),
        Ghash::new(0x5242d748a66f9abe17d4ca014c92081f),
    ],
    [
        Ghash::new(0x38176034b37ed0b51ea51bade772fd64),
        Ghash::new(0xdc10902a22159fdc02bc8fc3c43ee732),
        Ghash::new(0x54eafeed45ab9f55c2a668054e1433f2),
        Ghash::new(0x18c196024042197b29bba5779e6b4419),
    ],
    [
        Ghash::new(0xeeee77f02774996aff2fbdee3cafd5c6),
        Ghash::new(0xc09b024ff9e932be0fc9d294001fa6a7),
        Ghash::new(0x62c063613d99b026579fa6bf82e24cdb),
        Ghash::new(0x87df42121a98610c661fb711f68ee06e),
    ],
    [
        Ghash::new(0x40405fdefe20704cda9bccfaa1189228),
        Ghash::new(0xb683ec6a7ff2f49ae429c9f0e0e3f518),
        Ghash::new(0xa8e472626f047ffe07759441d3d2547f),
        Ghash::new(0x8c89cca408b7c95407db5cb1cc572a63),
    ],
    [
        Ghash::new(0xcaf9fccf7c70d8a7c2ec4142b9d5397e),
        Ghash::new(0x414420fface1c14f77cf760b5f7f980f),
        Ghash::new(0x0c8811f0d2dfb0dfe515f581db7c4d9d),
        Ghash::new(0x6303211a8c2a68f6103e2f03f7957ca4),
    ],
    [
        Ghash::new(0x10c51b39a067af1e593e9c46a1b77d37),
        Ghash::new(0x2319f7d203bbf1d50fe342d17066c9d3),
        Ghash::new(0x3dd51ba31a2018b8da50ec10ed923e6e),
        Ghash::new(0x676c6481332d0803cdea11539da55247),
    ],
    [
        Ghash::new(0xcd06f2d74ae291550c7cd3e34c3ca94d),
        Ghash::new(0x2aaea85e1163205f340a28bf015b3488),
        Ghash::new(0x48e0f19b5886ca1e9ce659ddf1dece57),
        Ghash::new(0x4b2ea0d6191010f82491ac06b6cbf7ad),
    ],
    [
        Ghash::new(0xb73fb2fc3f8a413ee8806392998452aa),
        Ghash::new(0x470372c4c732a0a7e61f858edaaa7f97),
        Ghash::new(0x647c2b2751c258135ccd956db51e7ffa),
        Ghash::new(0x079e5c6d2ec255a3206f1d231109247b),
    ],
    [
        Ghash::new(0x367538f6742ae83251d22a8112bf6fa1),
        Ghash::new(0x08539d5a297209d419a8906fee11e0a3),
        Ghash::new(0x3de8eec08b29b79911eb2420e4336284),
        Ghash::new(0x2fc0e3b4d3ea4de849486e07be859117),
    ],
    [
        Ghash::new(0xe25e1dbbcabb7885ae28db51bacbee5e),
        Ghash::new(0x9916b64185652326fda0575231b69ed2),
        Ghash::new(0x1ff4e4155cbf886110f5545da1f8722d),
        Ghash::new(0x4dcd1260ab92fbe0cec3b014367f626e),
    ],
    [
        Ghash::new(0x40ae8c84f958d259141595fed91e09cd),
        Ghash::new(0xf8222584576d0b5fa41584b91cf5f526),
        Ghash::new(0x5c021db63b4a3a27de7d38be3b26c52f),
        Ghash::new(0x509c5cb38660fee6ccc20b9abda1189b),
    ],
];

#[cfg(test)]
pub mod tests {
    use binius_field::{Field, Random, arithmetic_traits::Square};
    use binius_math::batch_invert::BatchInversion;
    use rand::{SeedableRng, rngs::StdRng};

    use super::{
        super::{
            linear_tables::{LINEAR_B_FWD_TABLE, LINEAR_B_INV_TABLE},
            permutation::{
                b_fwd_transform, b_inv_transform, constants_add, linearized_transform_scalar,
                mds_mul, sbox,
            },
        },
        *,
    };

    pub fn matrix_mul(matrix: &[Ghash; M * M], vector: &[Ghash; M]) -> [Ghash; M] {
        std::array::from_fn(|i| {
            // Row i: slice from i*N to (i+1)*N, dot product with vector
            matrix[i * M..(i + 1) * M]
                .iter()
                .zip(vector.iter())
                .map(|(&m_ij, &v_j)| m_ij * v_j)
                .sum()
        })
    }

    pub static INITIAL_CONSTANT: [Ghash; 4] = [
        Ghash::new(0xd0e015190f5e0795f4a1d28234ffdf87),
        Ghash::new(0x6111731acd9a89f6b93ec3a23ec7681b),
        Ghash::new(0x15da8de707ee3f3918a34a96728b8d29),
        Ghash::new(0x2ea920e89fbb13a1ed2a216b1d232bfb),
    ];

    pub static CONSTANTS_MATRIX: [Ghash; 16] = [
        Ghash::new(0x1cac9af051191d2c8ef96344bc8cbd0f),
        Ghash::new(0x3ceb350e8c2d2f4d4750ab0a854c3a4d),
        Ghash::new(0x09cc78f7eafef94d8e092e899156946b),
        Ghash::new(0x1d1ad17c8c7ee715a3d58f7eedb30dbb),
        Ghash::new(0x06bee572113950b3dd0cdae9dff5dfc5),
        Ghash::new(0x4799c3743a3560428b4bfa1a5cd3d295),
        Ghash::new(0x516883bf97b07fcf4cb89ac6bf636d0b),
        Ghash::new(0x0b681d1621d6aa1fdf1a9113b04d755b),
        Ghash::new(0xe9ba2f01051f6b4b3ddfe74c125e99e5),
        Ghash::new(0x1f58efc5ecf2e1b933e31cfb26b916d1),
        Ghash::new(0x36da671249b5444cf67efc573241fe19),
        Ghash::new(0x056cae18b867d486615a130556e5eb99),
        Ghash::new(0x1d8645dd4b1e46b78dd9df84956bcf11),
        Ghash::new(0x3efa34e48e3b218395efe6255339375b),
        Ghash::new(0x79bf44bae0d2379397d0812db56c5eff),
        Ghash::new(0x7fb3b7fce84776e39e538151daddae85),
    ];

    pub static CONSTANTS_CONSTANT: [Ghash; 4] = [
        Ghash::new(0x0d42981a71a7c2e493ca17e6bb10203f),
        Ghash::new(0x9ceed672ccb7030fca17ed48e18717b9),
        Ghash::new(0x0cec5c463024f2ba95c6ecdb4a349d15),
        Ghash::new(0xa46023576301b995990418f3a23e3c99),
    ];

    fn compute_round_constants() -> [[Ghash; M]; 1 + 2 * NUM_ROUNDS] {
        let mut round_keys = [[Ghash::ZERO; M]; 1 + 2 * NUM_ROUNDS];
        round_keys[0] = INITIAL_CONSTANT;

        let mut key_state = INITIAL_CONSTANT;
        let mut key_injection = INITIAL_CONSTANT;

        let mut round_key_index = 1;
        let mut inverter = BatchInversion::<Ghash>::new(M);
        for _ in 0..NUM_ROUNDS {
            for transform in [b_inv_transform, b_fwd_transform] {
                key_injection = matrix_mul(&CONSTANTS_MATRIX, &key_injection);
                constants_add(&mut key_injection, &CONSTANTS_CONSTANT);

                sbox(&mut key_state, transform, &mut inverter);
                mds_mul(&mut key_state);
                constants_add(&mut key_state, &key_injection);

                round_keys[round_key_index] = key_state;
                round_key_index += 1;
            }
        }

        round_keys
    }

    #[test]
    fn test_round_constants() {
        let round_constants = compute_round_constants();
        assert_eq!(round_constants, ROUND_CONSTANTS);
    }

    fn evaluate_b_fwd_linear(input: Ghash) -> Ghash {
        // Ignoring the affine coefficient B0, evaluate B1*x + B2*x^2 + B3*x^4
        B_FWD_COEFFS[1] * input
            + B_FWD_COEFFS[2] * input.square()
            + B_FWD_COEFFS[3] * input.square().square()
    }

    fn evaluate_b_fwd_affine(input: Ghash) -> Ghash {
        B_FWD_COEFFS[0] + evaluate_b_fwd_linear(input)
    }

    fn evaluate_b_inv_linear(input: Ghash) -> Ghash {
        let mut result = Ghash::ZERO;
        let mut x_power = input; // x^(2^0) = x

        // Evaluate sum(B_INV_COEFFS[i+1] * input^(2^i)) for i = 0..128
        // Skip constant term B_INV_COEFFS[0]
        for i in 0..128 {
            result += B_INV_COEFFS[i + 1] * x_power;
            x_power = x_power.square(); // x^(2^(i+1)) = (x^(2^i))^2
        }

        result
    }

    fn evaluate_b_inv_affine(input: Ghash) -> Ghash {
        B_INV_COEFFS[0] + evaluate_b_inv_linear(input)
    }

    #[test]
    fn test_linear_table_consistency() {
        let mut rng = StdRng::seed_from_u64(0);
        let input = Ghash::random(&mut rng);

        let mut b_fwd_result = input;
        linearized_transform_scalar(&mut b_fwd_result, &LINEAR_B_FWD_TABLE);
        let expected_b_fwd_result = evaluate_b_fwd_linear(input);
        assert_eq!(b_fwd_result, expected_b_fwd_result);

        let mut b_inv_result = input;
        linearized_transform_scalar(&mut b_inv_result, &LINEAR_B_INV_TABLE);
        let expected_b_inv_result = evaluate_b_inv_linear(input);
        assert_eq!(b_inv_result, expected_b_inv_result);
    }

    #[test]
    fn test_b_inverse_coeffs() {
        let mut rng = StdRng::seed_from_u64(0);
        let input = Ghash::random(&mut rng);

        let b_result = evaluate_b_fwd_affine(input);
        let recovered_input = evaluate_b_inv_affine(b_result);
        assert_eq!(
            recovered_input, input,
            "B_inverse verification failed: B_inv(B(x)) != x"
        );
    }
}
