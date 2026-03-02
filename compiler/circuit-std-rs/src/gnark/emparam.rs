use num_bigint::BigInt;

#[derive(Default, Clone, Copy, Debug)]
pub struct Bls12381Fp {}
impl Bls12381Fp {
    pub fn nb_limbs() -> u32 {
        48
    }
    pub fn bits_per_limb() -> u32 {
        8
    }
    pub fn is_prime() -> bool {
        true
    }
    pub fn modulus() -> BigInt {
        let hex_str = "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab";
        BigInt::parse_bytes(hex_str.as_bytes(), 16).unwrap()
    }
}
#[derive(Default, Clone)]
pub struct Bls12381Fr {}
impl Bls12381Fr {
    pub fn nb_limbs() -> u32 {
        32
    }
    pub fn bits_per_limb() -> u32 {
        8
    }
    pub fn is_prime() -> bool {
        true
    }
    pub fn modulus() -> BigInt {
        let hex_str = "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001";
        BigInt::parse_bytes(hex_str.as_bytes(), 16).unwrap()
    }
}
pub trait FieldParams {
    fn nb_limbs() -> u32;
    fn bits_per_limb() -> u32;
    fn is_prime() -> bool;
    fn modulus() -> BigInt;
}

impl FieldParams for Bls12381Fr {
    fn nb_limbs() -> u32 {
        Bls12381Fr::nb_limbs()
    }
    fn bits_per_limb() -> u32 {
        Bls12381Fr::bits_per_limb()
    }
    fn is_prime() -> bool {
        Bls12381Fr::is_prime()
    }
    fn modulus() -> BigInt {
        Bls12381Fr::modulus()
    }
}

impl FieldParams for Bls12381Fp {
    fn nb_limbs() -> u32 {
        Bls12381Fp::nb_limbs()
    }
    fn bits_per_limb() -> u32 {
        Bls12381Fp::bits_per_limb()
    }
    fn is_prime() -> bool {
        Bls12381Fp::is_prime()
    }
    fn modulus() -> BigInt {
        Bls12381Fp::modulus()
    }
}

// CurveParams defines parameters of an elliptic curve in short Weierstrass form
// given by the equation
//
//	Y² = X³ + aX + b
//
// The base point is defined by (Gx, Gy).
#[derive(Clone, Debug)]
pub struct CurveParams {
    pub a: BigInt,
    pub b: BigInt,
    pub gx: BigInt,
    pub gy: BigInt,
    // Enable following items if needed in the future.
    // pub gm: Vec<[BigInt; 2]>, m*base point coords
    // pub eigenvalue: Option<BigInt>, endomorphism eigenvalue
    // pub third_root_one: Option<BigInt>, endomorphism image scaler
}

impl CurveParams {
    pub fn get_bls12381_params() -> Self {
        let gx = BigInt::parse_bytes(b"3685416753713387016781088315183077757961620795782546409894578378688607592378376318836054947676345821548104185464507", 10).unwrap();
        let gy = BigInt::parse_bytes(b"1339506544944476473020471379941921221584933875938349620426543736416511423956333506472724655353366534992391756441569", 10).unwrap();

        CurveParams {
            a: BigInt::ZERO,
            b: BigInt::from(4),
            gx,
            gy,
            // gm: compute_bls12381_table(),
            // eigenvalue: Some(lambda),
            // third_root_one: Some(omega),
        }
    }

    // TODO add get_bn254_params, get_secp256k1_params etc. in the future.
}
