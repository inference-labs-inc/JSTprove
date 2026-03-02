use std::str::FromStr;

use crate::gnark::element::{new_internal_element, value_of, Element};
use crate::gnark::emparam::Bls12381Fp;
use crate::gnark::emulated::field_bls12381::e2::CurveF;
use crate::sha256::m31_utils::{big_less_than, from_binary};
use crate::utils::simple_select;
use expander_compiler::frontend::M31Config;
use expander_compiler::{
    declare_circuit,
    frontend::{Config, Define, RootAPI, Variable},
};
use num_bigint::BigInt;
use std::fmt::{Debug, Formatter, Result};

use super::point::AffinePoint;

const M_COMPRESSED_SMALLEST: u8 = 0b100 << 5;
const M_COMPRESSED_LARGEST: u8 = 0b101 << 5;

pub type G1Affine = AffinePoint<Bls12381Fp>;

impl G1Affine {
    pub fn new(x: Element<Bls12381Fp>, y: Element<Bls12381Fp>) -> Self {
        Self { x, y }
    }
    pub fn from_vars(x: Vec<Variable>, y: Vec<Variable>) -> Self {
        Self {
            x: Element::new(x, 0, false, false, false, Variable::default()),
            y: Element::new(y, 0, false, false, false, Variable::default()),
        }
    }
    pub fn one<C: Config, B: RootAPI<C>>(native: &mut B) -> Self {
        //g1Gen.X.SetString("3685416753713387016781088315183077757961620795782546409894578378688607592378376318836054947676345821548104185464507")
        //g1Gen.Y.SetString("1339506544944476473020471379941921221584933875938349620426543736416511423956333506472724655353366534992391756441569")
        Self {
            x: value_of::<C, B, Bls12381Fp>(native, Box::new("3685416753713387016781088315183077757961620795782546409894578378688607592378376318836054947676345821548104185464507".to_string())),
            y: value_of::<C, B, Bls12381Fp>(native, Box::new("1339506544944476473020471379941921221584933875938349620426543736416511423956333506472724655353366534992391756441569".to_string())),
        }
    }
}

impl Debug for G1Affine {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("G1Affine")
            .field("x", &self.x)
            .field("y", &self.y)
            .finish()
    }
}

pub struct G1 {
    pub curve_f: CurveF,
    pub w: Element<Bls12381Fp>,
}

impl G1 {
    pub fn new<C: Config, B: RootAPI<C>>(native: &mut B) -> Self {
        let curve_f = CurveF::new(native, Bls12381Fp {});
        let w = value_of::<C, B, Bls12381Fp>( native, Box::new("4002409555221667392624310435006688643935503118305586438271171395842971157480381377015405980053539358417135540939436".to_string()));

        Self { curve_f, w }
    }
    pub fn add<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        p: &G1Affine,
        q: &G1Affine,
    ) -> G1Affine {
        let qypy = self.curve_f.sub(native, &q.y, &p.y);
        let qxpx = self.curve_f.sub(native, &q.x, &p.x);
        let λ = self.curve_f.div(native, &qypy, &qxpx);

        let λλ = self.curve_f.mul(native, &λ, &λ);
        let qxpx = self.curve_f.add(native, &p.x, &q.x);
        let xr = self.curve_f.sub(native, &λλ, &qxpx);

        let pxrx = self.curve_f.sub(native, &p.x, &xr);
        let λpxrx = self.curve_f.mul(native, &λ, &pxrx);
        let yr = self.curve_f.sub(native, &λpxrx, &p.y);

        G1Affine { x: xr, y: yr }
    }

    pub fn phi<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, q: &G1Affine) -> G1Affine {
        let x = self.curve_f.mul(native, &q.x, &self.w);
        G1Affine { x, y: q.y.clone() }
    }

    pub fn double<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, p: &G1Affine) -> G1Affine {
        let xx3a = self.curve_f.mul(native, &p.x, &p.x);
        let two = value_of::<C, B, Bls12381Fp>(native, Box::new(2));
        let three = value_of::<C, B, Bls12381Fp>(native, Box::new(3));
        let xx3a = self.curve_f.mul(native, &xx3a, &three);
        let y1 = self.curve_f.mul(native, &p.y, &two);
        let λ = self.curve_f.div(native, &xx3a, &y1);

        let x1 = self.curve_f.mul(native, &p.x, &two);
        let λλ = self.curve_f.mul(native, &λ, &λ);
        let xr = self.curve_f.sub(native, &λλ, &x1);

        let pxrx = self.curve_f.sub(native, &p.x, &xr);
        let λpxrx = self.curve_f.mul(native, &λ, &pxrx);
        let yr = self.curve_f.sub(native, &λpxrx, &p.y);

        G1Affine { x: xr, y: yr }
    }

    pub fn double_n<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        p: &G1Affine,
        n: usize,
    ) -> G1Affine {
        let mut pn: G1Affine = p.clone();
        for _ in 0..n {
            pn = self.double(native, &pn);
        }
        pn
    }

    pub fn double_and_add<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        p: &G1Affine,
        q: &G1Affine,
    ) -> G1Affine {
        // compute λ1 = (q.y-p.y)/(q.x-p.x)
        let yqyp = self.curve_f.sub(native, &q.y, &p.y);
        let xqxp = self.curve_f.sub(native, &q.x, &p.x);
        let λ1 = self.curve_f.div(native, &yqyp, &xqxp);

        // compute x1 = λ1²-p.x-q.x
        let λ1λ1 = self.curve_f.mul(native, &λ1, &λ1);
        let xqxp = self.curve_f.add(native, &p.x, &q.x);
        let x2 = self.curve_f.sub(native, &λ1λ1, &xqxp);

        // omit y1 computation
        // compute λ1 = -λ1-1*p.y/(x1-p.x)
        let ypyp = self.curve_f.add(native, &p.y, &p.y);
        let x2xp = self.curve_f.sub(native, &x2, &p.x);
        let mut λ2 = self.curve_f.div(native, &ypyp, &x2xp);
        λ2 = self.curve_f.add(native, &λ1, &λ2);
        λ2 = self.curve_f.neg(native, &λ2);

        // compute x3 =λ2²-p.x-x3
        let λ2λ2 = self.curve_f.mul(native, &λ2, &λ2);
        let mut x3 = self.curve_f.sub(native, &λ2λ2, &p.x);
        x3 = self.curve_f.sub(native, &x3, &x2);

        // compute y3 = λ2*(p.x - x3)-p.y
        let mut y3 = self.curve_f.sub(native, &p.x, &x3);
        y3 = self.curve_f.mul(native, &λ2, &y3);
        y3 = self.curve_f.sub(native, &y3, &p.y);

        G1Affine { x: x3, y: y3 }
    }

    pub fn scalar_mul_by_seed_square<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        q: &G1Affine,
    ) -> G1Affine {
        let mut z = self.double(native, q);
        z = self.add(native, q, &z);
        z = self.double(native, &z);
        z = self.double_and_add(native, &z, q);
        z = self.double_n(native, &z, 2);
        z = self.double_and_add(native, &z, q);
        z = self.double_n(native, &z, 8);
        z = self.double_and_add(native, &z, q);
        let mut t0 = self.double(native, &z);
        t0 = self.add(native, &z, &t0);
        let mut t0 = self.double(native, &t0);
        t0 = self.double_and_add(native, &t0, &z);
        t0 = self.double_n(native, &t0, 2);
        t0 = self.double_and_add(native, &t0, &z);
        t0 = self.double_n(native, &t0, 8);
        t0 = self.double_and_add(native, &t0, &z);
        t0 = self.double_n(native, &t0, 31);
        z = self.add(native, &t0, &z);
        z = self.double_n(native, &z, 32);
        z = self.double_and_add(native, &z, q);
        z = self.double_n(native, &z, 32);
        z
    }

    pub fn assert_is_equal<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        a: &G1Affine,
        b: &G1Affine,
    ) {
        self.curve_f.assert_is_equal(native, &a.x, &b.x);
        self.curve_f.assert_is_equal(native, &a.y, &b.y);
    }
    pub fn copy_g1<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, q: &G1Affine) -> G1Affine {
        let copy_q_acc_x = self.curve_f.copy(native, &q.x);
        let copy_q_acc_y = self.curve_f.copy(native, &q.y);
        G1Affine {
            x: copy_q_acc_x,
            y: copy_q_acc_y,
        }
    }
    pub fn uncompressed<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        bytes: &[Variable],
    ) -> G1Affine {
        let mut buf_x = bytes.to_vec();
        let buf0 = native.to_binary(buf_x[0], 8);
        let pad = vec![native.constant(0); 5];
        let m_data = from_binary(native, [pad, buf0[5..].to_vec()].concat()); //buf0 & mMask
        let buf0_and_non_mask = from_binary(native, buf0[..5].to_vec()); //buf0 & ^mMask
        buf_x[0] = buf0_and_non_mask;

        //get p.x
        let rev_buf = buf_x.iter().rev().cloned().collect::<Vec<_>>();
        let px = new_internal_element(rev_buf, 0);

        //get YSquared
        let ysquared = self.curve_f.mul(native, &px, &px);
        let ysquared = self.curve_f.mul(native, &ysquared, &px);
        let b_curve_coeff = value_of::<C, B, Bls12381Fp>(native, Box::new(4));
        let ysquared = self.curve_f.add(native, &ysquared, &b_curve_coeff);

        let inputs = vec![ysquared.clone()];
        let outputs = self
            .curve_f
            .new_hint(native, "myhint.getelementsqrthint", 2, inputs);

        //is_square should be one
        let is_square = outputs[0].clone();
        let one = self.curve_f.one_const.clone();
        self.curve_f.assert_is_equal(native, &is_square, &one);

        //get Y
        let y = outputs[1].clone();
        //y^2 = ysquared
        let y_squared = self.curve_f.mul(native, &y, &y);
        self.curve_f.assert_is_equal(native, &y_squared, &ysquared);

        //if y is lexicographically largest
        let half_fp = BigInt::from_str("4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787").unwrap() / 2;
        let half_fp_var = value_of::<C, B, Bls12381Fp>(native, Box::new(half_fp));
        let is_large = big_less_than(
            native,
            Bls12381Fp::bits_per_limb() as usize,
            Bls12381Fp::nb_limbs() as usize,
            &half_fp_var.limbs,
            &y.limbs,
        );

        //if Y > -Y --> check if mData == mCompressedSmallest
        //if Y <= -Y --> check if mData == mCompressedLargest
        let m_compressed_largest = native.constant(M_COMPRESSED_LARGEST as u32);
        let m_compressed_smallest = native.constant(M_COMPRESSED_SMALLEST as u32);
        let check_m_data = simple_select(
            native,
            is_large,
            m_compressed_smallest,
            m_compressed_largest,
        );

        let check_res = native.sub(m_data, check_m_data);
        let neg_flag = native.is_zero(check_res);

        let neg_y = self.curve_f.neg(native, &y);

        let y = self.curve_f.select(native, neg_flag, &neg_y, &y);

        //TBD: subgroup check, do we need to do that? Since we are pretty sure that the public key bytes are correct, its unmashalling must be on the right curve
        G1Affine { x: px, y }
    }
    pub fn hash_to_fp<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        data: &[Variable],
    ) -> (Element<Bls12381Fp>, Element<Bls12381Fp>) {
        let u = self.curve_f.hash_to_fp(native, data, 2);
        (u[0].clone(), u[1].clone())
    }
    pub fn g1_isogeny<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        p: &G1Affine,
    ) -> G1Affine {
        let mut p = G1Affine {
            x: p.x.clone(),
            y: p.y.clone(),
        };
        let den1 = self.g1_isogeny_y_denominator(native, &p.x);
        let den0 = self.g1_isogeny_x_denominator(native, &p.x);
        p.y = self.g1_isogeny_y_numerator(native, &p.x, &p.y);
        p.x = self.g1_isogeny_x_numerator(native, &p.x);

        let den0 = self.curve_f.inverse(native, &den0);
        let den1 = self.curve_f.inverse(native, &den1);

        p.x = self.curve_f.mul(native, &p.x, &den0);
        p.y = self.curve_f.mul(native, &p.y, &den1);
        p
    }
    pub fn g1_isogeny_y_denominator<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        x: &Element<Bls12381Fp>,
    ) -> Element<Bls12381Fp> {
        let coeffs = vec![
            value_of::<C, B, Bls12381Fp>(native, Box::new("3396434800020507717552209507749485772788165484415495716688989613875369612529138640646200921379825018840894888371137".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("3907278185868397906991868466757978732688957419873771881240086730384895060595583602347317992689443299391009456758845".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("854914566454823955479427412036002165304466268547334760894270240966182605542146252771872707010378658178126128834546".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("3496628876382137961119423566187258795236027183112131017519536056628828830323846696121917502443333849318934945158166".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("1828256966233331991927609917644344011503610008134915752990581590799656305331275863706710232159635159092657073225757".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("1362317127649143894542621413133849052553333099883364300946623208643344298804722863920546222860227051989127113848748".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("3443845896188810583748698342858554856823966611538932245284665132724280883115455093457486044009395063504744802318172".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("3484671274283470572728732863557945897902920439975203610275006103818288159899345245633896492713412187296754791689945".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("3755735109429418587065437067067640634211015783636675372165599470771975919172394156249639331555277748466603540045130".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("3459661102222301807083870307127272890283709299202626530836335779816726101522661683404130556379097384249447658110805".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("742483168411032072323733249644347333168432665415341249073150659015707795549260947228694495111018381111866512337576".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("1662231279858095762833829698537304807741442669992646287950513237989158777254081548205552083108208170765474149568658".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("1668238650112823419388205992952852912407572045257706138925379268508860023191233729074751042562151098884528280913356".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("369162719928976119195087327055926326601627748362769544198813069133429557026740823593067700396825489145575282378487".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("2164195715141237148945939585099633032390257748382945597506236650132835917087090097395995817229686247227784224263055".to_string())),
        ];
        self.g1_eval_polynomial(native, true, coeffs, x)
    }
    pub fn g1_isogeny_x_denominator<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        x: &Element<Bls12381Fp>,
    ) -> Element<Bls12381Fp> {
        let coeffs = vec![
            value_of::<C, B, Bls12381Fp>(native, Box::new("1353092447850172218905095041059784486169131709710991428415161466575141675351394082965234118340787683181925558786844".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("2822220997908397120956501031591772354860004534930174057793539372552395729721474912921980407622851861692773516917759".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("1717937747208385987946072944131378949849282930538642983149296304709633281382731764122371874602115081850953846504985".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("501624051089734157816582944025690868317536915684467868346388760435016044027032505306995281054569109955275640941784".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("3025903087998593826923738290305187197829899948335370692927241015584233559365859980023579293766193297662657497834014".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("2224140216975189437834161136818943039444741035168992629437640302964164227138031844090123490881551522278632040105125".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("1146414465848284837484508420047674663876992808692209238763293935905506532411661921697047880549716175045414621825594".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("3179090966864399634396993677377903383656908036827452986467581478509513058347781039562481806409014718357094150199902".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("1549317016540628014674302140786462938410429359529923207442151939696344988707002602944342203885692366490121021806145".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("1442797143427491432630626390066422021593505165588630398337491100088557278058060064930663878153124164818522816175370".to_string())),
        ];
        self.g1_eval_polynomial(native, true, coeffs, x)
    }
    pub fn g1_isogeny_y_numerator<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        x: &Element<Bls12381Fp>,
        y: &Element<Bls12381Fp>,
    ) -> Element<Bls12381Fp> {
        let coeffs = vec![
            value_of::<C, B, Bls12381Fp>(native, Box::new("1393399195776646641963150658816615410692049723305861307490980409834842911816308830479576739332720113414154429643571".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("2968610969752762946134106091152102846225411740689724909058016729455736597929366401532929068084731548131227395540630".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("122933100683284845219599644396874530871261396084070222155796123161881094323788483360414289333111221370374027338230".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("303251954782077855462083823228569901064301365507057490567314302006681283228886645653148231378803311079384246777035".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("1353972356724735644398279028378555627591260676383150667237975415318226973994509601413730187583692624416197017403099".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("3443977503653895028417260979421240655844034880950251104724609885224259484262346958661845148165419691583810082940400".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("718493410301850496156792713845282235942975872282052335612908458061560958159410402177452633054233549648465863759602".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("1466864076415884313141727877156167508644960317046160398342634861648153052436926062434809922037623519108138661903145".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("1536886493137106337339531461344158973554574987550750910027365237255347020572858445054025958480906372033954157667719".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("2171468288973248519912068884667133903101171670397991979582205855298465414047741472281361964966463442016062407908400".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("3915937073730221072189646057898966011292434045388986394373682715266664498392389619761133407846638689998746172899634".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("3802409194827407598156407709510350851173404795262202653149767739163117554648574333789388883640862266596657730112910".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("1707589313757812493102695021134258021969283151093981498394095062397393499601961942449581422761005023512037430861560".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("349697005987545415860583335313370109325490073856352967581197273584891698473628451945217286148025358795756956811571".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("885704436476567581377743161796735879083481447641210566405057346859953524538988296201011389016649354976986251207243".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("3370924952219000111210625390420697640496067348723987858345031683392215988129398381698161406651860675722373763741188".to_string())),
        ];
        let dst = self.g1_eval_polynomial(native, false, coeffs, x);
        self.curve_f.mul(native, &dst, y)
    }
    pub fn g1_isogeny_x_numerator<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        x: &Element<Bls12381Fp>,
    ) -> Element<Bls12381Fp> {
        let coeffs = vec![
            value_of::<C, B, Bls12381Fp>(native, Box::new("2712959285290305970661081772124144179193819192423276218370281158706191519995889425075952244140278856085036081760695".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("3564859427549639835253027846704205725951033235539816243131874237388832081954622352624080767121604606753339903542203".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("2051387046688339481714726479723076305756384619135044672831882917686431912682625619320120082313093891743187631791280".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("3612713941521031012780325893181011392520079402153354595775735142359240110423346445050803899623018402874731133626465".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("2247053637822768981792833880270996398470828564809439728372634811976089874056583714987807553397615562273407692740057".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("3415427104483187489859740871640064348492611444552862448295571438270821994900526625562705192993481400731539293415811".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("2067521456483432583860405634125513059912765526223015704616050604591207046392807563217109432457129564962571408764292".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("3650721292069012982822225637849018828271936405382082649291891245623305084633066170122780668657208923883092359301262".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("1239271775787030039269460763652455868148971086016832054354147730155061349388626624328773377658494412538595239256855".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("3479374185711034293956731583912244564891370843071137483962415222733470401948838363051960066766720884717833231600798".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("2492756312273161536685660027440158956721981129429869601638362407515627529461742974364729223659746272460004902959995".to_string())),
            value_of::<C, B, Bls12381Fp>(native, Box::new("1058488477413994682556770863004536636444795456512795473806825292198091015005841418695586811009326456605062948114985".to_string())),
        ];
        self.g1_eval_polynomial(native, false, coeffs, x)
    }
    pub fn g1_eval_polynomial<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        monic: bool,
        coefficients: Vec<Element<Bls12381Fp>>,
        x: &Element<Bls12381Fp>,
    ) -> Element<Bls12381Fp> {
        let mut dst = coefficients[coefficients.len() - 1].clone();
        if monic {
            dst = self.curve_f.add(native, &dst, x);
        }
        for i in (0..coefficients.len() - 1).rev() {
            dst = self.curve_f.mul(native, &dst, x);
            dst = self.curve_f.add(native, &dst, &coefficients[i]);
        }
        dst
    }
    pub fn map_to_g1<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        in0: &Element<Bls12381Fp>,
        in1: &Element<Bls12381Fp>,
    ) -> G1Affine {
        let out0: G1Affine = self.map_to_curve1(native, in0);
        let out1 = self.map_to_curve1(native, in1);
        let out = self.add(native, &out0, &out1);
        let new_out = self.g1_isogeny(native, &out);
        self.clear_cofactor(native, &new_out)
    }
    pub fn mul_windowed<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        q: &G1Affine,
        s: BigInt,
    ) -> G1Affine {
        let double_q = self.double(native, q);
        let triple_q = self.add(native, &double_q, q);
        let ops = vec![q.clone(), double_q, triple_q];

        let b = s.to_bytes_be();
        let b = &b.1[1..];
        let mut res = ops[2].clone();

        res = self.double(native, &res);
        res = self.double(native, &res);
        res = self.add(native, &res, &ops[0]);

        res = self.double(native, &res);
        res = self.double(native, &res);

        res = self.double(native, &res);
        res = self.double(native, &res);
        res = self.add(native, &res, &ops[1]);

        for w in b {
            let mut mask = 0xc0;
            for j in 0..4 {
                res = self.double(native, &res);
                res = self.double(native, &res);
                let c = (w & mask) >> (6 - 2 * j);
                if c != 0 {
                    res = self.add(native, &res, &ops[(c - 1) as usize]);
                }
                mask >>= 2;
            }
        }
        res
    }
    pub fn clear_cofactor<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        p: &G1Affine,
    ) -> G1Affine {
        let x_big = BigInt::from_str("15132376222941642752").expect("Invalid string for BigInt");

        let res = self.mul_windowed(native, p, x_big.clone());
        self.add(native, &res, p)
    }
    pub fn map_to_curve1<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        in0: &Element<Bls12381Fp>,
    ) -> G1Affine {
        let a = value_of::<C, B, Bls12381Fp>(native, Box::new("12190336318893619529228877361869031420615612348429846051986726275283378313155663745811710833465465981901188123677".to_string()));
        let b = value_of::<C, B, Bls12381Fp>(native, Box::new("2906670324641927570491258158026293881577086121416628140204402091718288198173574630967936031029026176254968826637280".to_string()));

        //tv1.Square(u)
        let tv1 = self.curve_f.mul(native, in0, in0);

        //g1MulByZ(&tv1, &tv1)
        let tv1_mul_z = self.curve_f.add(native, &tv1, &tv1);
        let tv1_mul_z = self.curve_f.add(native, &tv1_mul_z, &tv1_mul_z);
        let tv1_mul_z = self.curve_f.add(native, &tv1_mul_z, &tv1);
        let tv1_mul_z = self.curve_f.add(native, &tv1_mul_z, &tv1_mul_z);
        let tv1_mul_z = self.curve_f.add(native, &tv1_mul_z, &tv1);

        //tv2.Square(&tv1)
        let tv2 = self.curve_f.mul(native, &tv1_mul_z, &tv1_mul_z);
        //tv2.Add(&tv2, &tv1)
        let tv2 = self.curve_f.add(native, &tv2, &tv1_mul_z);

        let tv4 = self.curve_f.one_const.clone();
        let tv3 = self.curve_f.add(native, &tv2, &tv4);
        let tv3 = self.curve_f.mul(native, &tv3, &b);

        let a_neg = self.curve_f.neg(native, &a);
        //tv2.Neg(&tv2) + tv2.Mul(&tv2, &sswuIsoCurveCoeffA)
        let tv2 = self.curve_f.mul(native, &a_neg, &tv2);

        //tv4.Mul(&tv4, &sswuIsoCurveCoeffA), since they are constant, we skip the mul and get the res value directly
        let tv4 = value_of::<C, B, Bls12381Fp>(native, Box::new("134093699507829814821517650980559345626771735832728306571853989028117161444712301203928819168120125800913069360447".to_string()));

        let tv2_zero = self.curve_f.is_zero(native, &tv2);

        //tv4.Select(int(tv2NZero), &tv2, &tv4)
        let tv4 = self.curve_f.select(native, tv2_zero, &tv4, &tv2);

        let tv3_div_tv4 = self.curve_f.div(native, &tv3, &tv4);

        //tv2 = (tv3^2 + tv4^2*a) * tv3 + tv4^3*b
        //tv6 = tv4^3
        //need sqrt(tv2/tv6) = sqrt(
        //tv3^3 + tv3*tv4^2*a + tv4^3*b)/tv4^3 = tv3_div^3 + tv3_div*a + b)
        //)

        //tv3_div^2
        let tv3_div_tv4_sq = self.curve_f.mul(native, &tv3_div_tv4, &tv3_div_tv4);
        //tv3_div^3
        let tv3_div_tv4_cub = self.curve_f.mul(native, &tv3_div_tv4, &tv3_div_tv4_sq);
        //tv3_div * a
        let tv3_div_tv4_a = self.curve_f.mul(native, &a, &tv3_div_tv4);
        //tv3_div^3 + tv3_div*a
        let ratio_tmp = self.curve_f.add(native, &tv3_div_tv4_cub, &tv3_div_tv4_a);
        //ratio = tv3_div^3 + tv3_div*a + b
        let y_sq = self.curve_f.add(native, &ratio_tmp, &b);

        //if ratio has square root, then y = sqrt(ratio), otherwise, y = new_y = sqrt(Z * ratio) * tv1 * u
        //here, we calculate new_y^2 = Z * ratio * tv1^2 * u^2, here tv1 = u^2 * Z, so we get new_y^2 = ratio * tv1^3

        //x = tv1 * tv3
        let x1 = self.curve_f.mul(native, &tv1_mul_z, &tv3_div_tv4);

        //tv1^2
        let tv1_mul_z_sq = self.curve_f.mul(native, &tv1_mul_z, &tv1_mul_z);
        //tv1^3
        let tv1_mul_z_cub = self.curve_f.mul(native, &tv1_mul_z_sq, &tv1_mul_z);

        //new_y^2 = ratio * tv1^3
        let y1_sq = self.curve_f.mul(native, &tv1_mul_z_cub, &y_sq);

        let inputs = vec![y_sq.clone(), y1_sq.clone(), in0.clone()];
        let output = self
            .curve_f
            .new_hint(native, "myhint.getsqrtx0x1fqnewhint", 2, inputs);
        let is_square = self.curve_f.is_zero(native, &output[0]); // is_square = 0 if y_sq has not square root, 1 otherwise
        let res_y = output[1].clone();

        let res_y_sq = self.curve_f.mul(native, &res_y, &res_y);

        let expected_y_sq = self.curve_f.select(native, is_square, &y1_sq, &res_y_sq);

        self.curve_f
            .assert_is_equal(native, &expected_y_sq, &res_y_sq);

        let sgn_in = self.curve_f.get_element_sign(native, in0);
        let sgn_y = self.curve_f.get_element_sign(native, &res_y);

        native.assert_is_equal(sgn_in, sgn_y);

        let out_b0 = self.curve_f.select(native, is_square, &x1, &tv3_div_tv4);
        let out_b1 = res_y.clone();
        G1Affine {
            x: out_b0,
            y: out_b1,
        }
    }
}

declare_circuit!(G1AddCircuit {
    p: [[Variable; 48]; 2],
    q: [[Variable; 48]; 2],
    r: [[Variable; 48]; 2],
});

impl Define<M31Config> for G1AddCircuit<Variable> {
    fn define<Builder: RootAPI<M31Config>>(&self, builder: &mut Builder) {
        let mut g1 = G1::new(builder);
        let p1_g1 = G1Affine::from_vars(self.p[0].to_vec(), self.p[1].to_vec());
        let p2_g1 = G1Affine::from_vars(self.q[0].to_vec(), self.q[1].to_vec());
        let r_g1 = G1Affine::from_vars(self.r[0].to_vec(), self.r[1].to_vec());
        let mut r = g1.add(builder, &p1_g1, &p2_g1);
        for _ in 0..16 {
            r = g1.add(builder, &r, &p2_g1);
        }
        g1.curve_f.assert_is_equal(builder, &r.x, &r_g1.x);
        g1.curve_f.assert_is_equal(builder, &r.y, &r_g1.y);
        g1.curve_f.check_mul(builder);
        g1.curve_f.table.final_check(builder);
        g1.curve_f.table.final_check(builder);
        g1.curve_f.table.final_check(builder);
    }
}

declare_circuit!(G1UncompressCircuit {
    x: [Variable; 48],
    y: [[Variable; 48]; 2],
});

impl Define<M31Config> for G1UncompressCircuit<Variable> {
    fn define<Builder: RootAPI<M31Config>>(&self, builder: &mut Builder) {
        let mut g1 = G1::new(builder);
        let public_key = g1.uncompressed(builder, &self.x);
        let expected_g1 = G1Affine::from_vars(self.y[0].to_vec(), self.y[1].to_vec());
        g1.curve_f
            .assert_is_equal(builder, &public_key.x, &expected_g1.x);
        g1.curve_f
            .assert_is_equal(builder, &public_key.y, &expected_g1.y);
        g1.curve_f.check_mul(builder);
        g1.curve_f.table.final_check(builder);
        g1.curve_f.table.final_check(builder);
        g1.curve_f.table.final_check(builder);
    }
}

declare_circuit!(HashToG1Circuit {
    msg: [Variable; 32],
    out: [[Variable; 48]; 2],
});

impl Define<M31Config> for HashToG1Circuit<Variable> {
    fn define<Builder: RootAPI<M31Config>>(&self, builder: &mut Builder) {
        let mut g1 = G1::new(builder);
        let (hm0, hm1) = g1.hash_to_fp(builder, &self.msg);
        let res = g1.map_to_g1(builder, &hm0, &hm1);
        let target_out = G1Affine::from_vars(self.out[0].to_vec(), self.out[1].to_vec());
        g1.assert_is_equal(builder, &res, &target_out);
        g1.curve_f.check_mul(builder);
        g1.curve_f.table.final_check(builder);
        g1.curve_f.table.final_check(builder);
        g1.curve_f.table.final_check(builder);
    }
}

#[cfg(test)]
mod tests {
    use super::G1AddCircuit;
    use super::G1UncompressCircuit;
    // use super::MapToG1Circuit;
    use super::HashToG1Circuit;
    use crate::utils::register_hint;
    use expander_compiler::frontend::*;
    use expander_compiler::{
        compile::CompileOptions,
        frontend::{compile, HintRegistry, M31},
    };
    use extra::debug_eval;
    use num_bigint::BigInt;
    use num_traits::Num;

    #[test]
    fn test_g1_add() {
        compile(&G1AddCircuit::default(), CompileOptions::default()).unwrap();
        let mut hint_registry = HintRegistry::<M31>::new();
        register_hint(&mut hint_registry);
        let mut assignment = G1AddCircuit::<M31> {
            p: [[M31::from(0); 48]; 2],
            q: [[M31::from(0); 48]; 2],
            r: [[M31::from(0); 48]; 2],
        };
        let p1_x_bytes = [
            169, 204, 143, 202, 195, 182, 32, 187, 150, 46, 27, 88, 137, 82, 209, 11, 255, 228,
            147, 72, 218, 149, 56, 139, 243, 28, 49, 146, 210, 5, 238, 232, 111, 204, 78, 170, 83,
            191, 222, 173, 137, 165, 150, 240, 62, 27, 213, 8,
        ];
        let p1_y_bytes = [
            85, 56, 238, 125, 65, 131, 108, 201, 186, 2, 96, 151, 226, 80, 22, 2, 111, 141, 203,
            67, 50, 147, 209, 102, 238, 82, 12, 96, 172, 239, 2, 177, 184, 146, 208, 150, 63, 214,
            239, 198, 101, 74, 169, 226, 148, 53, 104, 1,
        ];
        let p2_x_bytes = [
            108, 4, 52, 16, 255, 115, 116, 198, 234, 60, 202, 181, 169, 240, 221, 33, 38, 178, 114,
            195, 169, 16, 147, 33, 62, 116, 10, 191, 25, 163, 79, 192, 140, 43, 109, 235, 157, 42,
            15, 48, 115, 213, 48, 51, 19, 165, 178, 17,
        ];
        let p2_y_bytes = [
            130, 146, 65, 1, 211, 117, 217, 145, 69, 140, 76, 106, 43, 160, 192, 247, 96, 225, 2,
            72, 219, 238, 254, 202, 9, 210, 253, 111, 73, 49, 26, 145, 68, 161, 64, 101, 238, 0,
            236, 128, 164, 92, 95, 30, 143, 178, 6, 20,
        ];
        let res_x_bytes = [
            148, 92, 212, 64, 35, 246, 218, 14, 150, 169, 177, 191, 61, 6, 4, 120, 60, 253, 36,
            139, 95, 95, 14, 122, 89, 3, 62, 198, 100, 50, 114, 221, 144, 187, 29, 15, 203, 89,
            220, 29, 120, 25, 153, 169, 184, 184, 133, 16,
        ];
        let res_y_bytes = [
            41, 226, 254, 238, 50, 145, 74, 128, 160, 125, 237, 161, 93, 66, 241, 104, 218, 230,
            154, 134, 24, 204, 225, 220, 175, 115, 243, 57, 238, 157, 161, 175, 213, 34, 145, 106,
            226, 230, 19, 110, 196, 196, 229, 104, 152, 64, 12, 6,
        ];

        for i in 0..48 {
            assignment.p[0][i] = M31::from(p1_x_bytes[i]);
            assignment.p[1][i] = M31::from(p1_y_bytes[i]);
            assignment.q[0][i] = M31::from(p2_x_bytes[i]);
            assignment.q[1][i] = M31::from(p2_y_bytes[i]);
            assignment.r[0][i] = M31::from(res_x_bytes[i]);
            assignment.r[1][i] = M31::from(res_y_bytes[i]);
        }

        debug_eval(&G1AddCircuit::default(), &assignment, hint_registry);
    }

    #[test]
    fn test_uncompress_g1() {
        // compile(&G1UncompressCircuit::default(), CompileOptions::default()).unwrap();
        let mut hint_registry = HintRegistry::<M31>::new();
        register_hint(&mut hint_registry);
        let mut assignment = G1UncompressCircuit::<M31> {
            x: [M31::default(); 48],
            y: [[M31::default(); 48]; 2],
        };
        let x_bigint = BigInt::from_str_radix("a637bd4aefa20593ff82bdf832db2a98ca60c87796bca1d04a5a0206d52b4ede0e906d903360e04b69f8daec631f79fe", 16).unwrap();

        let x_bytes = x_bigint.to_bytes_be();

        let y_a0_bigint = BigInt::from_str_radix("956996561804650125715590823042978408716123343953697897618645235063950952926609558156980737775438019700668816652798", 10).unwrap();
        let y_a1_bigint = BigInt::from_str_radix("3556009343530533802204184826723274316816769528634825602353881354158551671080148026501040863742187196667680827782849", 10).unwrap();

        let y_a0_bytes = y_a0_bigint.to_bytes_le();
        let y_a1_bytes = y_a1_bigint.to_bytes_le();

        for i in 0..48 {
            assignment.x[i] = M31::from(x_bytes.1[i] as u32);
            assignment.y[0][i] = M31::from(y_a0_bytes.1[i] as u32);
            assignment.y[1][i] = M31::from(y_a1_bytes.1[i] as u32);
        }

        debug_eval(&G1UncompressCircuit::default(), &assignment, hint_registry);
    }

    #[test]
    fn test_hash_to_g1() {
        // compile(&HashToG2Circuit::default(), CompileOptions::default()).unwrap();
        let mut hint_registry = HintRegistry::<M31>::new();
        register_hint(&mut hint_registry);
        let mut assignment = HashToG1Circuit::<M31> {
            msg: [M31::from(0); 32],
            out: [[M31::from(0); 48]; 2],
        };
        let x_bigint = BigInt::from_str_radix(
            "8c944f8caa55d007728a2fc6e7ff3068dde103ed63fb399c59c24f1f826de4c7",
            16,
        )
        .unwrap();

        let x_bytes = x_bigint.to_bytes_be();

        let y_a0_bigint = BigInt::from_str_radix("931508203449116360366484402715781657513658072828297647050637028707500425620237136600612884240951972079295402518955", 10).unwrap();
        let y_a1_bigint = BigInt::from_str_radix("519166679736366508158130784988422711323587004159773257823344793142122588338441738530109373213103052261922442631575", 10).unwrap();
        let y_a0_bytes = y_a0_bigint.to_bytes_le();
        let y_a1_bytes = y_a1_bigint.to_bytes_le();

        for i in 0..32 {
            assignment.msg[i] = M31::from(x_bytes.1[i] as u32);
        }
        for i in 0..48 {
            assignment.out[0][i] = M31::from(y_a0_bytes.1[i] as u32);
            assignment.out[1][i] = M31::from(y_a1_bytes.1[i] as u32);
        }

        debug_eval(&HashToG1Circuit::default(), &assignment, hint_registry);
    }
}
