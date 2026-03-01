use crate::gnark::element::{value_of, Element};
use crate::gnark::emparam::Bls12381Fp;
use crate::gnark::field::GField;
use expander_compiler::frontend::{Config, RootAPI, Variable};
use num_bigint::BigInt;
use std::collections::HashMap;

pub type CurveF = GField<Bls12381Fp>;
#[derive(Default, Clone, Debug)]
pub struct GE2 {
    pub a0: Element<Bls12381Fp>,
    pub a1: Element<Bls12381Fp>,
}
impl GE2 {
    pub fn from_vars(x: Vec<Variable>, y: Vec<Variable>) -> Self {
        GE2 {
            a0: Element::new(x, 0, false, false, false, Variable::default()),
            a1: Element::new(y, 0, false, false, false, Variable::default()),
        }
    }
}

pub struct Ext2 {
    pub curve_f: CurveF,
    non_residues: HashMap<u32, HashMap<u32, GE2>>,
}

impl Ext2 {
    pub fn new<C: Config, B: RootAPI<C>>(api: &mut B) -> Self {
        let mut _non_residues: HashMap<u32, HashMap<u32, GE2>> = HashMap::new();
        let mut pwrs: HashMap<u32, HashMap<u32, GE2>> = HashMap::new();
        let a1_1_0 = value_of::<C, B, Bls12381Fp>(api, Box::new("3850754370037169011952147076051364057158807420970682438676050522613628423219637725072182697113062777891589506424760".to_string()));
        let a1_1_1 = value_of::<C, B, Bls12381Fp>(api, Box::new("151655185184498381465642749684540099398075398968325446656007613510403227271200139370504932015952886146304766135027".to_string()));
        let a1_2_0 = value_of::<C, B, Bls12381Fp>(api, Box::new("0".to_string()));
        let a1_2_1 = value_of::<C, B, Bls12381Fp>(api, Box::new("4002409555221667392624310435006688643935503118305586438271171395842971157480381377015405980053539358417135540939436".to_string()));
        let a1_3_0 = value_of::<C, B, Bls12381Fp>(api, Box::new("1028732146235106349975324479215795277384839936929757896155643118032610843298655225875571310552543014690878354869257".to_string()));
        let a1_3_1 = value_of::<C, B, Bls12381Fp>(api, Box::new("1028732146235106349975324479215795277384839936929757896155643118032610843298655225875571310552543014690878354869257".to_string()));
        let a1_4_0 = value_of::<C, B, Bls12381Fp>(api, Box::new("4002409555221667392624310435006688643935503118305586438271171395842971157480381377015405980053539358417135540939437".to_string()));
        let a1_4_1 = value_of::<C, B, Bls12381Fp>(api, Box::new("0".to_string()));
        let a1_5_0 = value_of::<C, B, Bls12381Fp>(api, Box::new("877076961050607968509681729531255177986764537961432449499635504522207616027455086505066378536590128544573588734230".to_string()));
        let a1_5_1 = value_of::<C, B, Bls12381Fp>(api, Box::new("3125332594171059424908108096204648978570118281977575435832422631601824034463382777937621250592425535493320683825557".to_string()));
        let a2_1_0 = value_of::<C, B, Bls12381Fp>(api, Box::new("793479390729215512621379701633421447060886740281060493010456487427281649075476305620758731620351".to_string()));
        let a2_2_0 = value_of::<C, B, Bls12381Fp>(api, Box::new("793479390729215512621379701633421447060886740281060493010456487427281649075476305620758731620350".to_string()));
        let a2_3_0 = value_of::<C, B, Bls12381Fp>(api, Box::new("4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559786".to_string()));
        let a2_4_0 = value_of::<C, B, Bls12381Fp>(api, Box::new("4002409555221667392624310435006688643935503118305586438271171395842971157480381377015405980053539358417135540939436".to_string()));
        let a2_5_0 = value_of::<C, B, Bls12381Fp>(api, Box::new("4002409555221667392624310435006688643935503118305586438271171395842971157480381377015405980053539358417135540939437".to_string()));
        pwrs.insert(1, HashMap::new());
        pwrs.get_mut(&1).unwrap().insert(
            1,
            GE2 {
                a0: a1_1_0,
                a1: a1_1_1,
            },
        );
        pwrs.get_mut(&1).unwrap().insert(
            2,
            GE2 {
                a0: a1_2_0,
                a1: a1_2_1,
            },
        );
        pwrs.get_mut(&1).unwrap().insert(
            3,
            GE2 {
                a0: a1_3_0,
                a1: a1_3_1,
            },
        );
        pwrs.get_mut(&1).unwrap().insert(
            4,
            GE2 {
                a0: a1_4_0,
                a1: a1_4_1,
            },
        );
        pwrs.get_mut(&1).unwrap().insert(
            5,
            GE2 {
                a0: a1_5_0,
                a1: a1_5_1,
            },
        );
        pwrs.insert(2, HashMap::new());
        let a_zero = value_of::<C, B, Bls12381Fp>(api, Box::new("0".to_string()));
        pwrs.get_mut(&2).unwrap().insert(
            1,
            GE2 {
                a0: a2_1_0,
                a1: a_zero,
            },
        );
        let a_zero = value_of::<C, B, Bls12381Fp>(api, Box::new("0".to_string()));
        pwrs.get_mut(&2).unwrap().insert(
            2,
            GE2 {
                a0: a2_2_0,
                a1: a_zero,
            },
        );
        let a_zero = value_of::<C, B, Bls12381Fp>(api, Box::new("0".to_string()));
        pwrs.get_mut(&2).unwrap().insert(
            3,
            GE2 {
                a0: a2_3_0,
                a1: a_zero,
            },
        );
        let a_zero = value_of::<C, B, Bls12381Fp>(api, Box::new("0".to_string()));
        pwrs.get_mut(&2).unwrap().insert(
            4,
            GE2 {
                a0: a2_4_0,
                a1: a_zero,
            },
        );
        let a_zero = value_of::<C, B, Bls12381Fp>(api, Box::new("0".to_string()));
        pwrs.get_mut(&2).unwrap().insert(
            5,
            GE2 {
                a0: a2_5_0,
                a1: a_zero,
            },
        );
        let fp = CurveF::new(api, Bls12381Fp {});
        Ext2 {
            curve_f: fp,
            non_residues: pwrs,
        }
    }
    pub fn one(&mut self) -> GE2 {
        let z0 = self.curve_f.one_const.clone();
        let z1 = self.curve_f.zero_const.clone();
        GE2 { a0: z0, a1: z1 }
    }
    pub fn zero(&mut self) -> GE2 {
        let z0 = self.curve_f.zero_const.clone();
        let z1 = self.curve_f.zero_const.clone();
        GE2 { a0: z0, a1: z1 }
    }
    pub fn is_zero<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, z: &GE2) -> Variable {
        let a0 = self.curve_f.is_zero(native, &z.a0);
        let a1 = self.curve_f.is_zero(native, &z.a1);
        native.and(a0, a1)
    }
    pub fn get_e2_sign<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        x: &GE2,
        a0_zero_flag: Variable,
    ) -> Variable {
        let bit_a0 = self.curve_f.get_element_sign(native, &x.a0);
        let bit_a1 = self.curve_f.get_element_sign(native, &x.a1);
        let sgn2 = native.mul(a0_zero_flag, bit_a1);
        let tmp0 = native.add(bit_a0, sgn2);
        let tmp1 = native.mul(bit_a0, sgn2);
        native.sub(tmp0, tmp1)
    }
    pub fn add<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, x: &GE2, y: &GE2) -> GE2 {
        let z0 = self.curve_f.add(native, &x.a0, &y.a0);
        let z1 = self.curve_f.add(native, &x.a1, &y.a1);
        GE2 { a0: z0, a1: z1 }
    }
    pub fn sub<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, x: &GE2, y: &GE2) -> GE2 {
        let z0 = self.curve_f.sub(native, &x.a0, &y.a0);
        let z1 = self.curve_f.sub(native, &x.a1, &y.a1);
        GE2 { a0: z0, a1: z1 }
    }
    pub fn double<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, x: &GE2) -> GE2 {
        let two = BigInt::from(2);
        let z0 = self.curve_f.mul_const(native, &x.a0, two.clone());
        let z1 = self.curve_f.mul_const(native, &x.a1, two.clone());
        GE2 { a0: z0, a1: z1 }
    }
    pub fn neg<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, x: &GE2) -> GE2 {
        let z0 = self.curve_f.neg(native, &x.a0);
        let z1 = self.curve_f.neg(native, &x.a1);
        GE2 { a0: z0, a1: z1 }
    }
    pub fn mul<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, x: &GE2, y: &GE2) -> GE2 {
        let v0 = self.curve_f.mul(native, &x.a0, &y.a0);
        let v1 = self.curve_f.mul(native, &x.a1, &y.a1);
        let b0 = self.curve_f.sub(native, &v0, &v1);
        let mut b1 = self.curve_f.add(native, &x.a0, &x.a1);
        let mut tmp = self.curve_f.add(native, &y.a0, &y.a1);
        b1 = self.curve_f.mul(native, &b1, &tmp);
        tmp = self.curve_f.add(native, &v0, &v1);
        b1 = self.curve_f.sub(native, &b1, &tmp);
        GE2 { a0: b0, a1: b1 }
    }
    pub fn mul_by_element<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        x: &GE2,
        y: &Element<Bls12381Fp>,
    ) -> GE2 {
        let v0 = self.curve_f.mul(native, &x.a0, y);
        let v1 = self.curve_f.mul(native, &x.a1, y);
        GE2 { a0: v0, a1: v1 }
    }
    pub fn mul_by_const_element<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        x: &GE2,
        y: &BigInt,
    ) -> GE2 {
        let z0 = self.curve_f.mul_const(native, &x.a0, y.clone());
        let z1 = self.curve_f.mul_const(native, &x.a1, y.clone());
        GE2 { a0: z0, a1: z1 }
    }
    pub fn mul_by_non_residue<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, x: &GE2) -> GE2 {
        let a = self.curve_f.sub(native, &x.a0, &x.a1);
        let b = self.curve_f.add(native, &x.a0, &x.a1);
        GE2 { a0: a, a1: b }
    }
    pub fn square<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, x: &GE2) -> GE2 {
        let a = self.curve_f.add(native, &x.a0, &x.a1);
        let b = self.curve_f.sub(native, &x.a0, &x.a1);
        let a = self.curve_f.mul(native, &a, &b);
        let b = self.curve_f.mul(native, &x.a0, &x.a1);
        let b = self.curve_f.mul_const(native, &b, BigInt::from(2));
        GE2 { a0: a, a1: b }
    }
    pub fn div<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, x: &GE2, y: &GE2) -> GE2 {
        let inputs = vec![x.a0.clone(), x.a1.clone(), y.a0.clone(), y.a1.clone()];
        let output = self.curve_f.new_hint(native, "myhint.dive2hint", 2, inputs);
        let div = GE2 {
            a0: output[0].clone(),
            a1: output[1].clone(),
        };
        let _x = self.mul(native, &div, y);
        self.assert_isequal(native, x, &_x);
        div
    }
    pub fn inverse_div<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, x: &GE2) -> GE2 {
        self.div(
            native,
            &GE2 {
                a0: self.curve_f.one_const.clone(),
                a1: self.curve_f.zero_const.clone(),
            },
            x,
        )
    }
    pub fn inverse<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, x: &GE2) -> GE2 {
        let inputs = vec![x.a0.clone(), x.a1.clone()];
        let output = self
            .curve_f
            .new_hint(native, "myhint.inversee2hint", 2, inputs);
        let inv = GE2 {
            a0: output[0].clone(),
            a1: output[1].clone(),
        };
        let one = GE2 {
            a0: self.curve_f.one_const.clone(),
            a1: self.curve_f.zero_const.clone(),
        };
        let _one = self.mul(native, &inv, x);
        self.assert_isequal(native, &one, &_one);
        inv
    }
    pub fn assert_isequal<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, x: &GE2, y: &GE2) {
        self.curve_f.assert_is_equal(native, &x.a0, &y.a0);
        self.curve_f.assert_is_equal(native, &x.a1, &y.a1);
    }
    pub fn select<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        selector: Variable,
        z1: &GE2,
        z0: &GE2,
    ) -> GE2 {
        let a0 = self.curve_f.select(native, selector, &z1.a0, &z0.a0);
        let a1 = self.curve_f.select(native, selector, &z1.a1, &z0.a1);
        GE2 { a0, a1 }
    }
    pub fn conjugate<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, x: &GE2) -> GE2 {
        let z0 = x.a0.clone();
        let z1 = self.curve_f.neg(native, &x.a1);
        GE2 { a0: z0, a1: z1 }
    }
    pub fn mul_by_non_residue_generic<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        x: &GE2,
        power: u32,
        coef: u32,
    ) -> GE2 {
        let y = self
            .non_residues
            .get(&power)
            .unwrap()
            .get(&coef)
            .unwrap()
            .clone();
        self.mul(native, x, &y)
    }
    pub fn mul_by_non_residue1_power1<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        x: &GE2,
    ) -> GE2 {
        self.mul_by_non_residue_generic(native, x, 1, 1)
    }
    pub fn mul_by_non_residue1_power2<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        x: &GE2,
    ) -> GE2 {
        let element = value_of::<C, B, Bls12381Fp>(native, Box::new("4002409555221667392624310435006688643935503118305586438271171395842971157480381377015405980053539358417135540939436".to_string()));
        let a = self.curve_f.mul(native, &x.a1, &element);
        let a = self.curve_f.neg(native, &a);
        let b = self.curve_f.mul(native, &x.a0, &element);
        GE2 { a0: a, a1: b }
    }
    pub fn mul_by_non_residue1_power3<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        x: &GE2,
    ) -> GE2 {
        self.mul_by_non_residue_generic(native, x, 1, 3)
    }
    pub fn mul_by_non_residue1_power4<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        x: &GE2,
    ) -> GE2 {
        let element = value_of::<C, B, Bls12381Fp>(native, Box::new("4002409555221667392624310435006688643935503118305586438271171395842971157480381377015405980053539358417135540939437".to_string()));
        let a = self.curve_f.mul(native, &x.a0, &element);
        let b = self.curve_f.mul(native, &x.a1, &element);
        GE2 { a0: a, a1: b }
    }
    pub fn mul_by_non_residue1_power5<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        x: &GE2,
    ) -> GE2 {
        self.mul_by_non_residue_generic(native, x, 1, 5)
    }
    pub fn mul_by_non_residue2_power1<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        x: &GE2,
    ) -> GE2 {
        let element = value_of::<C, B, Bls12381Fp>(native, Box::new("793479390729215512621379701633421447060886740281060493010456487427281649075476305620758731620351".to_string()));
        let a = self.curve_f.mul(native, &x.a0, &element);
        let b = self.curve_f.mul(native, &x.a1, &element);
        GE2 { a0: a, a1: b }
    }
    pub fn mul_by_non_residue2_power2<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        x: &GE2,
    ) -> GE2 {
        let element = value_of::<C, B, Bls12381Fp>(native, Box::new("793479390729215512621379701633421447060886740281060493010456487427281649075476305620758731620350".to_string()));
        let a = self.curve_f.mul(native, &x.a0, &element);
        let b = self.curve_f.mul(native, &x.a1, &element);
        GE2 { a0: a, a1: b }
    }
    pub fn mul_by_non_residue2_power3<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        x: &GE2,
    ) -> GE2 {
        let element = value_of::<C, B, Bls12381Fp>(native, Box::new("4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559786".to_string()));
        let a = self.curve_f.mul(native, &x.a0, &element);
        let b = self.curve_f.mul(native, &x.a1, &element);
        GE2 { a0: a, a1: b }
    }
    pub fn mul_by_non_residue2_power4<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        x: &GE2,
    ) -> GE2 {
        let element = value_of::<C, B, Bls12381Fp>(native, Box::new("4002409555221667392624310435006688643935503118305586438271171395842971157480381377015405980053539358417135540939436".to_string()));
        let a = self.curve_f.mul(native, &x.a0, &element);
        let b = self.curve_f.mul(native, &x.a1, &element);
        GE2 { a0: a, a1: b }
    }
    pub fn mul_by_non_residue2_power5<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        x: &GE2,
    ) -> GE2 {
        let element = value_of::<C, B, Bls12381Fp>(native, Box::new("4002409555221667392624310435006688643935503118305586438271171395842971157480381377015405980053539358417135540939437".to_string()));
        let a = self.curve_f.mul(native, &x.a0, &element);
        let b = self.curve_f.mul(native, &x.a1, &element);
        GE2 { a0: a, a1: b }
    }
    pub fn non_residue<C: Config, B: RootAPI<C>>(&mut self, _native: &mut B) -> GE2 {
        let one = self.curve_f.one_const.clone();
        GE2 {
            a0: one.clone(),
            a1: one.clone(),
        }
    }
    pub fn copy<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, x: &GE2) -> GE2 {
        let inputs = vec![x.a0.clone(), x.a1.clone()];
        let output = self
            .curve_f
            .new_hint(native, "myhint.copye2hint", 2, inputs);
        let res = GE2 {
            a0: output[0].clone(),
            a1: output[1].clone(),
        };
        self.assert_isequal(native, x, &res);
        res
    }
}
