#![allow(clippy::op_ref)]

use crate::ff_ext::Legendre;
use digest::{core_api::BlockSizeUser, Digest};
use ff::{Field, FromUniformBytes, PrimeField};
use pasta_curves::arithmetic::CurveExt;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

pub enum Method {
    SSWU,
    SVDW,
}

pub struct Suite<C: CurveExt, D: Digest + BlockSizeUser, const L: usize> {
    domain: Vec<u8>,
    map_to_curve: Box<dyn Fn(C::Base) -> C>,
    _marker: std::marker::PhantomData<D>,
}

pub(crate) fn expand_message<D: Digest + BlockSizeUser>(
    domain_prefix: &[u8],
    domain: &[u8],
    message: &[u8],
    out_len: usize,
) -> Vec<u8> {
    assert!(
        domain_prefix.len() + domain.len() < 256,
        "long dst is not supported yet"
    );

    let mut h = D::new();
    h.update(vec![0; D::block_size()]);
    h.update(message);
    h.update([(out_len >> 8) as u8, out_len as u8, 0]);
    h.update(domain_prefix);
    h.update(domain);
    h.update([(domain.len() + domain_prefix.len()) as u8]);
    let b_0 = h.finalize();

    let mut h = D::new();
    h.update(&b_0);
    h.update([1]);
    h.update(domain_prefix);
    h.update(domain);
    h.update([(domain.len() + domain_prefix.len()) as u8]);
    let mut b_i = h.finalize();

    let output_size = <D as Digest>::output_size();
    let ell = (out_len + output_size - 1) / output_size;
    let mut out = vec![0u8; out_len];

    for i in 1..ell {
        let mut h = D::new();
        b_0.iter()
            .zip(b_i.iter())
            .for_each(|(b_0, b_i)| h.update([*b_0 ^ *b_i]));
        h.update([1 + i as u8]);
        h.update(domain_prefix);
        h.update(domain);
        h.update([(domain.len() + domain_prefix.len()) as u8]);

        out.iter_mut()
            .skip((i - 1) * output_size)
            .zip(b_i.iter())
            .for_each(|(out, b_i)| *out = *b_i);

        b_i = h.finalize();
    }

    out.iter_mut()
        .skip((ell - 1) * output_size)
        .zip(b_i.iter())
        .for_each(|(out, b_i)| *out = *b_i);

    out
}

#[allow(clippy::type_complexity)]
pub fn hash_to_curve<'a, C, D: Digest + BlockSizeUser + 'a, const L: usize>(
    domain_prefix: &'a str,
    suite: Suite<C, D, L>,
) -> Box<dyn Fn(&[u8]) -> C + 'a>
where
    C: CurveExt,
    C::Base: Legendre,
    C::Base: FromUniformBytes<L>,
{
    Box::new(move |message| suite.hash_to_curve(domain_prefix, message))
}

impl<C: CurveExt, D: Digest + BlockSizeUser, const L: usize> Suite<C, D, L>
where
    C::Base: Legendre + FromUniformBytes<L>,
{
    pub(crate) fn new(domain: &[u8], z: C::Base, method: Method) -> Self {
        // Check for the target bits of  security `k`. Currently, the target security is 128 bits.
        // See: <https://www.ietf.org/archive/id/draft-irtf-cfrg-hash-to-curve-16.html#section-5.1>
        // L = ceil((ceil(log2(p)) + k) / 8)
        assert!((C::Base::NUM_BITS as usize + 128) / 8 <= L);

        let map_to_curve: Box<dyn Fn(C::Base) -> C> = match method {
            Method::SSWU => Box::new(move |u| sswu_map_to_curve::<C>(u, z)),
            Method::SVDW => {
                let [c1, c2, c3, c4] = svdw_precomputed_constants::<C>(z);
                Box::new(move |u| svdw_map_to_curve::<C>(u, c1, c2, c3, c4, z))
            }
        };

        Self {
            map_to_curve,
            domain: domain.to_vec(),
            _marker: std::marker::PhantomData,
        }
    }

    pub(crate) fn hash_to_field(&self, domain_prefix: &[u8], message: &[u8]) -> (C::Base, C::Base) {
        let out = expand_message::<D>(domain_prefix, &self.domain[..], message, L * 2);

        let u0 = {
            let mut out = out[0..L].to_vec();
            out.reverse();
            let out: [u8; L] = out.try_into().unwrap();
            C::Base::from_uniform_bytes(&out)
        };

        let u1 = {
            let mut out = out[L..L * 2].to_vec();
            out.reverse();
            let out: [u8; L] = out.try_into().unwrap();
            C::Base::from_uniform_bytes(&out)
        };

        (u0, u1)
    }

    pub fn hash_to_curve(&self, domain_prefix: &str, message: &[u8]) -> C {
        let (u0, u1) = self.hash_to_field(domain_prefix.as_bytes(), message);
        (self.map_to_curve)(u0) + (self.map_to_curve)(u1)
    }
}

pub(crate) fn svdw_precomputed_constants<C: CurveExt>(z: C::Base) -> [C::Base; 4] {
    let a = C::a();
    let b = C::b();
    let one = C::Base::ONE;
    let three = one + one + one;
    let four = three + one;
    let tmp = three * z.square() + four * a;

    // 1. c1 = g(Z)
    let c1 = (z.square() + a) * z + b;
    // 2. c2 = -Z / 2
    let c2 = -z * C::Base::TWO_INV;
    // 3. c3 = sqrt(-g(Z) * (3 * Z^2 + 4 * A))    # sgn0(c3) MUST equal 0
    let c3 = {
        let c3 = (-c1 * tmp).sqrt().unwrap();
        C::Base::conditional_select(&c3, &-c3, c3.is_odd())
    };
    // 4. c4 = -4 * g(Z) / (3 * Z^2 + 4 * A)
    let c4 = -four * c1 * tmp.invert().unwrap();

    [c1, c2, c3, c4]
}

// Implementation of <https://datatracker.ietf.org/doc/html/rfc9380#name-simplified-swu-method>
#[allow(clippy::too_many_arguments)]
pub(crate) fn sswu_map_to_curve<C>(u: C::Base, z: C::Base) -> C
where
    C: CurveExt,
{
    // Implement https://datatracker.ietf.org/doc/html/rfc9380#name-sqrt_ratio-for-any-field
    // Copied from ff sqrt_ratio_generic substituting F::ROOT_OF_UNITY for input Z
    fn sqrt_ratio<F: PrimeField>(num: &F, div: &F, z: &F) -> (Choice, F) {
        // General implementation:
        //
        // a = num * inv0(div)
        //   = {    0    if div is zero
        //     { num/div otherwise
        //
        // b = z * a
        //   = {      0      if div is zero
        //     { z*num/div otherwise

        // Since z is non-square, a and b are either both zero (and both square), or
        // only one of them is square. We can therefore choose the square root to return
        // based on whether a is square, but for the boolean output we need to handle the
        // num != 0 && div == 0 case specifically.

        let a = div.invert().unwrap_or(F::ZERO) * num;
        let b = a * z;
        let sqrt_a = a.sqrt();
        let sqrt_b = b.sqrt();

        let num_is_zero = num.is_zero();
        let div_is_zero = div.is_zero();
        let is_square = sqrt_a.is_some();
        let is_nonsquare = sqrt_b.is_some();
        assert!(bool::from(
            num_is_zero | div_is_zero | (is_square ^ is_nonsquare)
        ));

        (
            is_square & (num_is_zero | !div_is_zero),
            CtOption::conditional_select(&sqrt_b, &sqrt_a, is_square).unwrap(),
        )
    }

    let zero = C::Base::ZERO;
    let one = C::Base::ONE;
    let a = C::a();
    let b = C::b();

    //1.  tv1 = u^2
    let tv1 = u.square();
    //2.  tv1 = Z * tv1
    let tv1 = z * tv1;
    //3.  tv2 = tv1^2
    let tv2 = tv1.square();
    //4.  tv2 = tv2 + tv1
    let tv2 = tv2 + tv1;
    //5.  tv3 = tv2 + 1
    let tv3 = tv2 + one;
    //6.  tv3 = B * tv3
    let tv3 = b * tv3;
    //7.  tv4 = CMOV(Z, -tv2, tv2 != 0) # tv4 = z if tv2 is 0 else tv4 = -tv2
    let tv2_is_not_zero = !tv2.ct_eq(&zero);
    let tv4 = C::Base::conditional_select(&z, &-tv2, tv2_is_not_zero);
    //8.  tv4 = A * tv4
    let tv4 = a * tv4;
    //9.  tv2 = tv3^2
    let tv2 = tv3.square();
    //10. tv6 = tv4^2
    let tv6 = tv4.square();
    //11. tv5 = A * tv6
    let tv5 = a * tv6;
    //12. tv2 = tv2 + tv5
    let tv2 = tv2 + tv5;
    //13. tv2 = tv2 * tv3
    let tv2 = tv2 * tv3;
    //14. tv6 = tv6 * tv4
    let tv6 = tv6 * tv4;
    //15. tv5 = B * tv6
    let tv5 = b * tv6;
    //16. tv2 = tv2 + tv5
    let tv2 = tv2 + tv5;
    //17.   x = tv1 * tv3
    let x = tv1 * tv3;
    //18. (is_gx1_square, y1) = sqrt_ratio(tv2, tv6)
    let (is_gx1_square, y1) = sqrt_ratio(&tv2, &tv6, &z);
    //19.   y = tv1 * u
    let y = tv1 * u;
    //20.   y = y * y1
    let y = y * y1;
    //21.   x = CMOV(x, tv3, is_gx1_square)
    let x = C::Base::conditional_select(&x, &tv3, is_gx1_square);
    //22.   y = CMOV(y, y1, is_gx1_square)
    let y = C::Base::conditional_select(&y, &y1, is_gx1_square);
    //23.  e1 = sgn0(u) == sgn0(y)
    let e1 = u.is_odd().ct_eq(&y.is_odd());
    //24.   y = CMOV(-y, y, e1) # Select correct sign of y
    let y = C::Base::conditional_select(&-y, &y, e1);
    //25.   x = x / tv4
    let x = x * tv4.invert().unwrap();
    //26. return (x, y)
    C::new_jacobian(x, y, one).unwrap()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn svdw_map_to_curve<C>(
    u: C::Base,
    c1: C::Base,
    c2: C::Base,
    c3: C::Base,
    c4: C::Base,
    z: C::Base,
) -> C
where
    C: CurveExt,
    C::Base: Legendre,
{
    let one = C::Base::ONE;
    let a = C::a();
    let b = C::b();

    // 1. tv1 = u^2
    let tv1 = u.square();
    // 2. tv1 = tv1 * c1
    let tv1 = tv1 * c1;
    // 3. tv2 = 1 + tv1
    let tv2 = one + tv1;
    // 4. tv1 = 1 - tv1
    let tv1 = one - tv1;
    // 5. tv3 = tv1 * tv2
    let tv3 = tv1 * tv2;
    // 6. tv3 = inv0(tv3)
    let tv3 = tv3.invert().unwrap_or(C::Base::ZERO);
    // 7. tv4 = u * tv1
    let tv4 = u * tv1;
    // 8. tv4 = tv4 * tv3
    let tv4 = tv4 * tv3;
    // 9. tv4 = tv4 * c3
    let tv4 = tv4 * c3;
    // 10. x1 = c2 - tv4
    let x1 = c2 - tv4;
    // 11. gx1 = x1^2
    let gx1 = x1.square();
    // 12. gx1 = gx1 + A
    let gx1 = gx1 + a;
    // 13. gx1 = gx1 * x1
    let gx1 = gx1 * x1;
    // 14. gx1 = gx1 + B
    let gx1 = gx1 + b;
    // 15. e1 = is_square(gx1)
    let e1 = !gx1.ct_quadratic_non_residue();
    // 16. x2 = c2 + tv4
    let x2 = c2 + tv4;
    // 17. gx2 = x2^2
    let gx2 = x2.square();
    // 18. gx2 = gx2 + A
    let gx2 = gx2 + a;
    // 19. gx2 = gx2 * x2
    let gx2 = gx2 * x2;
    // 20. gx2 = gx2 + B
    let gx2 = gx2 + b;
    // 21. e2 = is_square(gx2) AND NOT e1    # Avoid short-circuit logic ops
    let e2 = !gx2.ct_quadratic_non_residue() & (!e1);
    // 22. x3 = tv2^2
    let x3 = tv2.square();
    // 23. x3 = x3 * tv3
    let x3 = x3 * tv3;
    // 24. x3 = x3^2
    let x3 = x3.square();
    // 25. x3 = x3 * c4
    let x3 = x3 * c4;
    // 26. x3 = x3 + Z
    let x3 = x3 + z;
    // 27. x = CMOV(x3, x1, e1)    # x = x1 if gx1 is square, else x = x3
    let x = C::Base::conditional_select(&x3, &x1, e1);
    // 28. x = CMOV(x, x2, e2)    # x = x2 if gx2 is square and gx1 is not
    let x = C::Base::conditional_select(&x, &x2, e2);
    // 29. gx = x^2
    let gx = x.square();
    // 30. gx = gx + A
    let gx = gx + a;
    // 31. gx = gx * x
    let gx = gx * x;
    // 32. gx = gx + B
    let gx = gx + b;
    // 33. y = sqrt(gx)
    let y = gx.sqrt().unwrap();
    // 34. e3 = sgn0(u) == sgn0(y)
    let e3 = u.is_odd().ct_eq(&y.is_odd());
    // 35. y = CMOV(-y, y, e3)    # Select correct sign of y
    let y = C::Base::conditional_select(&-y, &y, e3);
    // 36. return (x, y)
    C::new_jacobian(x, y, one).unwrap()
}

#[cfg(test)]
mod test {

    use super::*;
    use sha2::Sha256;
    use sha2::Sha512;
    use std::marker::PhantomData;

    #[test]
    fn test_expand_message() {
        // Test vectors are taken from:
        // https://www.ietf.org/archive/id/draft-irtf-cfrg-hash-to-curve-16.html#name-expand_message_xmdsha-256

        struct Test<D: Digest + BlockSizeUser> {
            msg: &'static [u8],
            expect: Vec<u8>,
            _marker: PhantomData<D>,
        }

        impl<D: Digest + BlockSizeUser> Test<D> {
            fn new(msg: &'static [u8], expect: &str) -> Self {
                Self {
                    msg,
                    expect: crate::tests::hex_to_bytes(expect),
                    _marker: PhantomData,
                }
            }

            fn run(&self, domain_prefix: &[u8], domain: &[u8]) {
                let outlen = self.expect.len();
                let out = expand_message::<D>(domain_prefix, domain, self.msg, outlen);
                assert_eq!(out, self.expect);
            }
        }
        [
            // out len 0x20
            Test::<Sha256>::new(
                b"",
                "68a985b87eb6b46952128911f2a4412bbc302a9d759667f87f7a21d803f07235",
            ),
            Test::<Sha256>::new(
                b"abc",
                "d8ccab23b5985ccea865c6c97b6e5b8350e794e603b4b97902f53a8a0d605615",
            ),
            Test::<Sha256>::new(
                b"abcdef0123456789",
                "eff31487c770a893cfb36f912fbfcbff40d5661771ca4b2cb4eafe524333f5c1",
            ),
            Test::<Sha256>::new(
                b"q128_qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
                "b23a1d2b4d97b2ef7785562a7e8bac7eed54ed6e97e29aa51bfe3f12ddad1ff9",
            ),
            Test::<Sha256>::new(
                b"a512_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "4623227bcc01293b8c130bf771da8c298dede7383243dc0993d2d94823958c4c",
            ),
            // out len 0x80
            Test::<Sha256>::new(
                b"",
                "af84c27ccfd45d41914fdff5df25293e221afc53d8ad2ac06d5e3e29485dadbee0d121587713a3e0dd4d5e69e93eb7cd4f5df4cd103e188cf60cb02edc3edf18eda8576c412b18ffb658e3dd6ec849469b979d444cf7b26911a08e63cf31f9dcc541708d3491184472c2c29bb749d4286b004ceb5ee6b9a7fa5b646c993f0ced",
            ),
            Test::<Sha256>::new(
                b"abc",
                "abba86a6129e366fc877aab32fc4ffc70120d8996c88aee2fe4b32d6c7b6437a647e6c3163d40b76a73cf6a5674ef1d890f95b664ee0afa5359a5c4e07985635bbecbac65d747d3d2da7ec2b8221b17b0ca9dc8a1ac1c07ea6a1e60583e2cb00058e77b7b72a298425cd1b941ad4ec65e8afc50303a22c0f99b0509b4c895f40",
            ),
            Test::<Sha256>::new(
                b"abcdef0123456789",
                "ef904a29bffc4cf9ee82832451c946ac3c8f8058ae97d8d629831a74c6572bd9ebd0df635cd1f208e2038e760c4994984ce73f0d55ea9f22af83ba4734569d4bc95e18350f740c07eef653cbb9f87910d833751825f0ebefa1abe5420bb52be14cf489b37fe1a72f7de2d10be453b2c9d9eb20c7e3f6edc5a60629178d9478df",
            ),
            Test::<Sha256>::new(
                b"q128_qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
                "80be107d0884f0d881bb460322f0443d38bd222db8bd0b0a5312a6fedb49c1bbd88fd75d8b9a09486c60123dfa1d73c1cc3169761b17476d3c6b7cbbd727acd0e2c942f4dd96ae3da5de368d26b32286e32de7e5a8cb2949f866a0b80c58116b29fa7fabb3ea7d520ee603e0c25bcaf0b9a5e92ec6a1fe4e0391d1cdbce8c68a",
            ),
            Test::<Sha256>::new(
                b"a512_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "546aff5444b5b79aa6148bd81728704c32decb73a3ba76e9e75885cad9def1d06d6792f8a7d12794e90efed817d96920d728896a4510864370c207f99bd4a608ea121700ef01ed879745ee3e4ceef777eda6d9e5e38b90c86ea6fb0b36504ba4a45d22e86f6db5dd43d98a294bebb9125d5b794e9d2a81181066eb954966a487",
            ),

        ]
        .iter()
        .for_each(|test| {
            test.run(b"QUUX-V01-CS02-with-expander-",b"SHA256-128");
        });

        [
            // out len 0x20
            Test::<Sha512>::new(
                b"",
                "6b9a7312411d92f921c6f68ca0b6380730a1a4d982c507211a90964c394179ba",
            ),
            Test::<Sha512>::new(
                b"abc",
                "0da749f12fbe5483eb066a5f595055679b976e93abe9be6f0f6318bce7aca8dc",
            ),
            Test::<Sha512>::new(
                b"abcdef0123456789",
                "087e45a86e2939ee8b91100af1583c4938e0f5fc6c9db4b107b83346bc967f58",
            ),
            Test::<Sha512>::new(
                b"q128_qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
                "7336234ee9983902440f6bc35b348352013becd88938d2afec44311caf8356b3",
            ),
            Test::<Sha512>::new(
                b"a512_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "57b5f7e766d5be68a6bfe1768e3c2b7f1228b3e4b3134956dd73a59b954c66f4",
            ),
            // out len 0x80
            Test::<Sha512>::new(
                b"",
                "41b037d1734a5f8df225dd8c7de38f851efdb45c372887be655212d07251b921b052b62eaed99b46f72f2ef4cc96bfaf254ebbbec091e1a3b9e4fb5e5b619d2e0c5414800a1d882b62bb5cd1778f098b8eb6cb399d5d9d18f5d5842cf5d13d7eb00a7cff859b605da678b318bd0e65ebff70bec88c753b159a805d2c89c55961",
            ),
            Test::<Sha512>::new(
                b"abc",
                "7f1dddd13c08b543f2e2037b14cefb255b44c83cc397c1786d975653e36a6b11bdd7732d8b38adb4a0edc26a0cef4bb45217135456e58fbca1703cd6032cb1347ee720b87972d63fbf232587043ed2901bce7f22610c0419751c065922b488431851041310ad659e4b23520e1772ab29dcdeb2002222a363f0c2b1c972b3efe1",
            ),
            Test::<Sha512>::new(
                b"abcdef0123456789",
                "3f721f208e6199fe903545abc26c837ce59ac6fa45733f1baaf0222f8b7acb0424814fcb5eecf6c1d38f06e9d0a6ccfbf85ae612ab8735dfdf9ce84c372a77c8f9e1c1e952c3a61b7567dd0693016af51d2745822663d0c2367e3f4f0bed827feecc2aaf98c949b5ed0d35c3f1023d64ad1407924288d366ea159f46287e61ac",
            ),
            Test::<Sha512>::new(
                b"q128_qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
                "b799b045a58c8d2b4334cf54b78260b45eec544f9f2fb5bd12fb603eaee70db7317bf807c406e26373922b7b8920fa29142703dd52bdf280084fb7ef69da78afdf80b3586395b433dc66cde048a258e476a561e9deba7060af40adf30c64249ca7ddea79806ee5beb9a1422949471d267b21bc88e688e4014087a0b592b695ed",
            ),
            Test::<Sha512>::new(
                b"a512_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "05b0bfef265dcee87654372777b7c44177e2ae4c13a27f103340d9cd11c86cb2426ffcad5bd964080c2aee97f03be1ca18e30a1f14e27bc11ebbd650f305269cc9fb1db08bf90bfc79b42a952b46daf810359e7bc36452684784a64952c343c52e5124cd1f71d474d5197fefc571a92929c9084ffe1112cf5eea5192ebff330b",
            ),
        ]
        .iter()
        .for_each(|test| {
            test.run(b"QUUX-V01-CS02-with-expander-", b"SHA512-256");
        });
    }
}
