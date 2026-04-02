// Copyright 2025 Irreducible Inc.

use binius_field::BinaryField128bGhash as Ghash;

pub const M: usize = 6;

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

pub const ROUND_CONSTANTS: [[Ghash; M]; 1 + 2 * NUM_ROUNDS] = [
    [
        Ghash::new(0xd0e015190f5e0795f4a1d28234ffdf87),
        Ghash::new(0x6111731acd9a89f6b93ec3a23ec7681b),
        Ghash::new(0x15da8de707ee3f3918a34a96728b8d29),
        Ghash::new(0x2ea920e89fbb13a1ed2a216b1d232bfb),
        Ghash::new(0x1cac9af051191d2c8ef96344bc8cbd0f),
        Ghash::new(0x3ceb350e8c2d2f4d4750ab0a854c3a4d),
    ],
    [
        Ghash::new(0xe8e46b76a404c4b9fb9ccc257eebb52c),
        Ghash::new(0xe12be62f399ab7f6f0d391aa43d111f),
        Ghash::new(0x8cd24d620049a85e9fca7f1a82b24575),
        Ghash::new(0x9134196e2a4f6643d720e7e17d5d6155),
        Ghash::new(0x801a72226c1bb12fc5fa2ca648ab4389),
        Ghash::new(0xcc930a049d59a9f159eed487b2b1db41),
    ],
    [
        Ghash::new(0x377785252c619dafa7c6f70050db5eba),
        Ghash::new(0xd8c470279b2f60e703297d3a48f34781),
        Ghash::new(0x6d07ea9a9e22d6ca035c6f84c216797),
        Ghash::new(0xc41bfc3f4b22043bca9a33cc6a6aae00),
        Ghash::new(0x55dcc5f6e50336ed4bcdad6a23bc4a5d),
        Ghash::new(0xd9642daf0fbb53ee0976778ee009e681),
    ],
    [
        Ghash::new(0xa13280fd8c7d7e2150894cb4a0a0dd96),
        Ghash::new(0xa4b51297ac45514d432a00dc34c2a8cc),
        Ghash::new(0x1e45b8cb510bc51faa3dc225cc36dba6),
        Ghash::new(0x1979585fb152fd8fc6967e9422868fd2),
        Ghash::new(0x2e04baf1b1566acadc9b1d5b2942c2c8),
        Ghash::new(0xffd35b46d1c35c53b28522b43b660159),
    ],
    [
        Ghash::new(0xb02df026790e55cc69b823a2b7dfd3ba),
        Ghash::new(0xec66a8fac8862b2c45843a2b7a9b29af),
        Ghash::new(0xa9e560ef4d3c0076cf8d6b94bdfb0898),
        Ghash::new(0xc56e1ed6dea993ece4f564b4ec5a434),
        Ghash::new(0x60e546921e56307c250f8a0735064af6),
        Ghash::new(0x85cc8b97f2d3f384866096f91bd627b5),
    ],
    [
        Ghash::new(0x2f936e61dcf44a719ce2d4997752c552),
        Ghash::new(0xea100b89ea4dbea306104f6cbcaf7730),
        Ghash::new(0x78ee9d4dde1866f1be20fc5d3225a471),
        Ghash::new(0xed9364d3d4072597d49449ccdd74196b),
        Ghash::new(0xc0c48365b28688406b77f700ee0a7a15),
        Ghash::new(0x3bf2f14a632d1dbc83779737f7f4ca43),
    ],
    [
        Ghash::new(0xf8f790e955a2e979c6268179afd19a9b),
        Ghash::new(0x2224b54c57b488fcd7a43440a4cbbe97),
        Ghash::new(0x3b8e1338408771be189ac7c8cbf21463),
        Ghash::new(0x6547c511c5a77ad2ea728ee6204ac283),
        Ghash::new(0x753e6ed7ce73c0f1697db0399d9b0973),
        Ghash::new(0xa99266d737016f4fd16b0f23a838050b),
    ],
    [
        Ghash::new(0xe6ff2ec311c6722e3f60fd9efe1fb28a),
        Ghash::new(0x28dad9ef1797cd06c15e4e8a1cf0bcdc),
        Ghash::new(0x44444b925bea2192fe04f2d8854cde64),
        Ghash::new(0xc53dec23ad61e0db3c2af60fc4f47e7e),
        Ghash::new(0x484cc2cb23ddfbccd8e90f3143d132d2),
        Ghash::new(0xf717e5e43240cea2f506653013c41123),
    ],
    [
        Ghash::new(0xd25ab18ab153ac32527e793a74aafa07),
        Ghash::new(0x8a3ec34a5ef348b7686caffae59ffc9d),
        Ghash::new(0xc560c870b90f45d21d4607f56ebd7721),
        Ghash::new(0xe39414d28ca4b425ccf80c31abd8ae2d),
        Ghash::new(0x9c2a85dcc4048127b435c0199b3b29ff),
        Ghash::new(0xadaa409205e183e1ac95e6f92f1d43d0),
    ],
    [
        Ghash::new(0xa53431abb1c8ebdad01e00eda12c466b),
        Ghash::new(0xa66273bde8debca0e822591ab1496b5),
        Ghash::new(0x72b8bba8b48f2485ac98313e10212a73),
        Ghash::new(0x48fd620807602baea745b3bd12feca1c),
        Ghash::new(0xcf4a6a04386ee4222dc8ffc3a4d562e6),
        Ghash::new(0x96dc648bb3119c91f113d51194a85820),
    ],
    [
        Ghash::new(0x66c470d06e5f7211f98a9c68c1fa17c),
        Ghash::new(0xc7614b3d5a97fa5f7879a309878a9dd8),
        Ghash::new(0x53e1050f5bdf4b7eae0195ad35dbaeaa),
        Ghash::new(0x8385c9ec48e14fd5ec230e8974ab0c65),
        Ghash::new(0xa977ac08c7e6b5b3e6948d24d81da15),
        Ghash::new(0x29253ebc78e9261c58cc3140c7581355),
    ],
    [
        Ghash::new(0x5f31289e38eef8ff80e98d29e000ea37),
        Ghash::new(0x4f8c759141e50d8438f488de2679f820),
        Ghash::new(0xf4dc8a27ea07f3d4feb34a32e61eaaf2),
        Ghash::new(0xc8b283172ea7de37ee97b4112ec8e22f),
        Ghash::new(0xcb13d7b37ba1c466f6094884d3469da2),
        Ghash::new(0x4bae7acd67ba3de4e5f6b6ea6d4c6bac),
    ],
    [
        Ghash::new(0x3f47078acec655bc33272905c9d750f4),
        Ghash::new(0xa58fef1654a01bcf630804c8c27910e1),
        Ghash::new(0x923c97fc1029185b06a109e1be6924c5),
        Ghash::new(0x2eedb745d45085a746a15c1fb5e2ae1e),
        Ghash::new(0x144e4497996ff1c99d7a03410a443422),
        Ghash::new(0x40ec3d5a6112778c8997de4ae7f7d82e),
    ],
    [
        Ghash::new(0x4577e50a3e827e9f3e73893b21e58ced),
        Ghash::new(0x7a370d0173a2c0541593ed813c4174f8),
        Ghash::new(0x87394d70d4ae0f6828e7a93875a68b92),
        Ghash::new(0xc1560cbda08dd31b936373127702c481),
        Ghash::new(0xf5fee9656ca1bc3c2b813e558d15e83a),
        Ghash::new(0xeec0f749f981c7f621dceee24b4c0db8),
    ],
    [
        Ghash::new(0x4180e8ff79cda2b4ccb01a124d07a773),
        Ghash::new(0x6db908873ac7627985498d9d64418458),
        Ghash::new(0x734a0331ccc7b06a54dc33cb191ff5a6),
        Ghash::new(0x578e478c6b4d4625d8d5a1a7e99c8a98),
        Ghash::new(0x1f01a86286bd8078d9143ef0e1bfd6d2),
        Ghash::new(0x8f831a4d9a0e5685ed3a2ab4ec1492dd),
    ],
    [
        Ghash::new(0x5c3f7446fd05199857a7061949601cae),
        Ghash::new(0x3c62d0ca7faef4745a8646a4497084d2),
        Ghash::new(0x5fe18dbc6e5f7ae8bae52881bc87b32b),
        Ghash::new(0x836cc41840685efc5f821768fa5a7a0b),
        Ghash::new(0x9fa4c5dca8677c17b3cc1aa5bf35831f),
        Ghash::new(0x32b3cb0dd6c40e007188d1544c941d02),
    ],
    [
        Ghash::new(0x9ff1ff4417b1c815f9dbff4f62d07ebd),
        Ghash::new(0x7f0b5b1123fc8a30c3fab6efe18d2f0a),
        Ghash::new(0xe6f9119237bea8838e7553928d5cbe51),
        Ghash::new(0x2ed980b39d9ee4af6bda8599ee17f39c),
        Ghash::new(0x79ee6ad7ccca965879b31429e31f3689),
        Ghash::new(0x286668e58f3dc1de9369fb0ff898ed83),
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

    pub static INITIAL_CONSTANT: [Ghash; M] = [
        Ghash::new(0xd0e015190f5e0795f4a1d28234ffdf87),
        Ghash::new(0x6111731acd9a89f6b93ec3a23ec7681b),
        Ghash::new(0x15da8de707ee3f3918a34a96728b8d29),
        Ghash::new(0x2ea920e89fbb13a1ed2a216b1d232bfb),
        Ghash::new(0x1cac9af051191d2c8ef96344bc8cbd0f),
        Ghash::new(0x3ceb350e8c2d2f4d4750ab0a854c3a4d),
    ];

    pub static CONSTANTS_MATRIX: [Ghash; M * M] = [
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
        Ghash::new(0x0d42981a71a7c2e493ca17e6bb10203f),
        Ghash::new(0x9ceed672ccb7030fca17ed48e18717b9),
        Ghash::new(0x0cec5c463024f2ba95c6ecdb4a349d15),
        Ghash::new(0xa46023576301b995990418f3a23e3c99),
        Ghash::new(0x6227628fea1dc5465d33ae5d97eb83af),
        Ghash::new(0x2ef550b7cda1e2856d94bf331473f6f1),
        Ghash::new(0x375f4232fa7ca41a698bc6ae2e3a8499),
        Ghash::new(0x2d2f859caba39ed7012b9ab22c303529),
        Ghash::new(0x5672a0afa29cac300d0bf7b6399257b7),
        Ghash::new(0xebd30b6903f1fe11f3ae1aee7aa41593),
        Ghash::new(0x39c54aa0738c373a6f66f0ef2f18bd8d),
        Ghash::new(0x083a52acda7952eab89afcd0d6a1a9f9),
        Ghash::new(0x07c782c6beef83babafc31f177df4359),
        Ghash::new(0x1ed204d4affa05b8e5ca6278e76adc93),
        Ghash::new(0x5ce4a5c3be5e0bc5a543ee1ec99323cb),
        Ghash::new(0xd54d1bf8ea15dafe5f49f51af3c2df7),
        Ghash::new(0x22510fa8fffa288adc4e99bc400b9fa9),
        Ghash::new(0x78ca292f525f6a68f8135f4f4b2b4491),
        Ghash::new(0x0ef5f99b1b4a36e979afa9e365bfa159),
        Ghash::new(0x7a999d6e5f0b133d3a2f3f77630cffa1),
        Ghash::new(0xb1ae2573d4a15bb6d21d9ad73aef8469),
        Ghash::new(0x9c348d3213f5a0782535d2146c4869e1),
    ];

    pub static CONSTANTS_CONSTANT: [Ghash; M] = [
        Ghash::new(0xabd5e3e6c590952db4ee8dd307ef621f),
        Ghash::new(0x22681ba7c4bf168c2cf0fc6f51bedceb),
        Ghash::new(0x41baa8130ac4ddf61a74f42c784257a1),
        Ghash::new(0x26f2ab43dac65cb011ac77cfb11d7ccd),
        Ghash::new(0x29218b482b8935fa1552b09907e977e3),
        Ghash::new(0x1cb764573d0acdfe818a702d07fb9d),
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
