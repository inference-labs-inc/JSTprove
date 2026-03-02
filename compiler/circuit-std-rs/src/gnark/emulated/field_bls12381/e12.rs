use expander_compiler::frontend::{Config, RootAPI, Variable};

use super::e2::GE2;
use super::e6::{Ext6, GE6};
#[derive(Default, Clone)]
pub struct GE12 {
    pub c0: GE6,
    pub c1: GE6,
}

pub struct Ext12 {
    pub ext6: Ext6,
}

impl Ext12 {
    pub fn new<C: Config, B: RootAPI<C>>(api: &mut B) -> Self {
        Self {
            ext6: Ext6::new(api),
        }
    }
    pub fn zero(&mut self) -> GE12 {
        let zero = self.ext6.ext2.curve_f.zero_const.clone();
        GE12 {
            c0: GE6 {
                b0: GE2 {
                    a0: zero.clone(),
                    a1: zero.clone(),
                },
                b1: GE2 {
                    a0: zero.clone(),
                    a1: zero.clone(),
                },
                b2: GE2 {
                    a0: zero.clone(),
                    a1: zero.clone(),
                },
            },
            c1: GE6 {
                b0: GE2 {
                    a0: zero.clone(),
                    a1: zero.clone(),
                },
                b1: GE2 {
                    a0: zero.clone(),
                    a1: zero.clone(),
                },
                b2: GE2 {
                    a0: zero.clone(),
                    a1: zero.clone(),
                },
            },
        }
    }
    pub fn one(&mut self) -> GE12 {
        let one = self.ext6.ext2.curve_f.one_const.clone();
        let zero = self.ext6.ext2.curve_f.zero_const.clone();
        GE12 {
            c0: GE6 {
                b0: GE2 {
                    a0: one.clone(),
                    a1: zero.clone(),
                },
                b1: GE2 {
                    a0: zero.clone(),
                    a1: zero.clone(),
                },
                b2: GE2 {
                    a0: zero.clone(),
                    a1: zero.clone(),
                },
            },
            c1: GE6 {
                b0: GE2 {
                    a0: zero.clone(),
                    a1: zero.clone(),
                },
                b1: GE2 {
                    a0: zero.clone(),
                    a1: zero.clone(),
                },
                b2: GE2 {
                    a0: zero.clone(),
                    a1: zero.clone(),
                },
            },
        }
    }
    pub fn is_zero<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, z: &GE12) -> Variable {
        let c0 = self.ext6.is_zero(native, &z.c0);
        let c1 = self.ext6.is_zero(native, &z.c1);
        native.and(c0, c1)
    }
    pub fn add<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, x: &GE12, y: &GE12) -> GE12 {
        let z0 = self.ext6.add(native, &x.c0, &y.c0);
        let z1 = self.ext6.add(native, &x.c1, &y.c1);
        GE12 { c0: z0, c1: z1 }
    }
    pub fn sub<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, x: &GE12, y: &GE12) -> GE12 {
        let z0 = self.ext6.sub(native, &x.c0, &y.c0);
        let z1 = self.ext6.sub(native, &x.c1, &y.c1);
        GE12 { c0: z0, c1: z1 }
    }
    pub fn conjugate<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, x: &GE12) -> GE12 {
        let z1 = self.ext6.neg(native, &x.c1);
        GE12 {
            c0: x.c0.clone(),
            c1: z1,
        }
    }
    pub fn mul<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, x: &GE12, y: &GE12) -> GE12 {
        let a = self.ext6.add(native, &x.c0, &x.c1);
        let b = self.ext6.add(native, &y.c0, &y.c1);
        let a = self.ext6.mul(native, &a, &b);
        let b = self.ext6.mul(native, &x.c0, &y.c0);
        let c = self.ext6.mul(native, &x.c1, &y.c1);
        let d = self.ext6.add(native, &c, &b);
        let z1 = self.ext6.sub(native, &a, &d);
        let z0 = self.ext6.mul_by_non_residue(native, &c);
        let z0 = self.ext6.add(native, &z0, &b);
        GE12 { c0: z0, c1: z1 }
    }
    pub fn square<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, x: &GE12) -> GE12 {
        let c0 = self.ext6.sub(native, &x.c0, &x.c1);
        let c3 = self.ext6.mul_by_non_residue(native, &x.c1);
        let c3 = self.ext6.sub(native, &x.c0, &c3);
        let c2 = self.ext6.mul(native, &x.c0, &x.c1);
        let c0 = self.ext6.mul(native, &c0, &c3);
        let c0 = self.ext6.add(native, &c0, &c2);
        let z1 = self.ext6.double(native, &c2);
        let c2 = self.ext6.mul_by_non_residue(native, &c2);
        let z0 = self.ext6.add(native, &c0, &c2);
        GE12 { c0: z0, c1: z1 }
    }

    pub fn cyclotomic_square<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        x: &GE12,
    ) -> GE12 {
        let t0 = self.ext6.ext2.square(native, &x.c1.b1);
        let t1 = self.ext6.ext2.square(native, &x.c0.b0);
        let mut t6 = self.ext6.ext2.add(native, &x.c1.b1, &x.c0.b0);
        t6 = self.ext6.ext2.square(native, &t6);
        t6 = self.ext6.ext2.sub(native, &t6, &t0);
        t6 = self.ext6.ext2.sub(native, &t6, &t1);
        let t2 = self.ext6.ext2.square(native, &x.c0.b2);
        let t3 = self.ext6.ext2.square(native, &x.c1.b0);
        let mut t7 = self.ext6.ext2.add(native, &x.c0.b2, &x.c1.b0);
        t7 = self.ext6.ext2.square(native, &t7);
        t7 = self.ext6.ext2.sub(native, &t7, &t2);
        t7 = self.ext6.ext2.sub(native, &t7, &t3);
        let t4 = self.ext6.ext2.square(native, &x.c1.b2);
        let t5 = self.ext6.ext2.square(native, &x.c0.b1);
        let mut t8 = self.ext6.ext2.add(native, &x.c1.b2, &x.c0.b1);
        t8 = self.ext6.ext2.square(native, &t8);
        t8 = self.ext6.ext2.sub(native, &t8, &t4);
        t8 = self.ext6.ext2.sub(native, &t8, &t5);
        t8 = self.ext6.ext2.mul_by_non_residue(native, &t8);
        let t0 = self.ext6.ext2.mul_by_non_residue(native, &t0);
        let t0 = self.ext6.ext2.add(native, &t0, &t1);
        let t2 = self.ext6.ext2.mul_by_non_residue(native, &t2);
        let t2 = self.ext6.ext2.add(native, &t2, &t3);
        let t4 = self.ext6.ext2.mul_by_non_residue(native, &t4);
        let t4 = self.ext6.ext2.add(native, &t4, &t5);
        let z00 = self.ext6.ext2.sub(native, &t0, &x.c0.b0);
        let z00 = self.ext6.ext2.double(native, &z00);
        let z00 = self.ext6.ext2.add(native, &z00, &t0);
        let z01 = self.ext6.ext2.sub(native, &t2, &x.c0.b1);
        let z01 = self.ext6.ext2.double(native, &z01);
        let z01 = self.ext6.ext2.add(native, &z01, &t2);
        let z02 = self.ext6.ext2.sub(native, &t4, &x.c0.b2);
        let z02 = self.ext6.ext2.double(native, &z02);
        let z02 = self.ext6.ext2.add(native, &z02, &t4);
        let z10 = self.ext6.ext2.add(native, &t8, &x.c1.b0);
        let z10 = self.ext6.ext2.double(native, &z10);
        let z10 = self.ext6.ext2.add(native, &z10, &t8);
        let z11 = self.ext6.ext2.add(native, &t6, &x.c1.b1);
        let z11 = self.ext6.ext2.double(native, &z11);
        let z11 = self.ext6.ext2.add(native, &z11, &t6);
        let z12 = self.ext6.ext2.add(native, &t7, &x.c1.b2);
        let z12 = self.ext6.ext2.double(native, &z12);
        let z12 = self.ext6.ext2.add(native, &z12, &t7);
        GE12 {
            c0: GE6 {
                b0: z00,
                b1: z01,
                b2: z02,
            },
            c1: GE6 {
                b0: z10,
                b1: z11,
                b2: z12,
            },
        }
    }
    pub fn assert_isequal<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, x: &GE12, y: &GE12) {
        self.ext6.assert_isequal(native, &x.c0, &y.c0);
        self.ext6.assert_isequal(native, &x.c1, &y.c1);
    }
    pub fn div<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, x: &GE12, y: &GE12) -> GE12 {
        let inputs = vec![
            x.c0.b0.a0.clone(),
            x.c0.b0.a1.clone(),
            x.c0.b1.a0.clone(),
            x.c0.b1.a1.clone(),
            x.c0.b2.a0.clone(),
            x.c0.b2.a1.clone(),
            x.c1.b0.a0.clone(),
            x.c1.b0.a1.clone(),
            x.c1.b1.a0.clone(),
            x.c1.b1.a1.clone(),
            x.c1.b2.a0.clone(),
            x.c1.b2.a1.clone(),
            y.c0.b0.a0.clone(),
            y.c0.b0.a1.clone(),
            y.c0.b1.a0.clone(),
            y.c0.b1.a1.clone(),
            y.c0.b2.a0.clone(),
            y.c0.b2.a1.clone(),
            y.c1.b0.a0.clone(),
            y.c1.b0.a1.clone(),
            y.c1.b1.a0.clone(),
            y.c1.b1.a1.clone(),
            y.c1.b2.a0.clone(),
            y.c1.b2.a1.clone(),
        ];
        let output = self
            .ext6
            .ext2
            .curve_f
            .new_hint(native, "myhint.dive12hint", 24, inputs);
        let div = GE12 {
            c0: GE6 {
                b0: GE2 {
                    a0: output[0].clone(),
                    a1: output[1].clone(),
                },
                b1: GE2 {
                    a0: output[2].clone(),
                    a1: output[3].clone(),
                },
                b2: GE2 {
                    a0: output[4].clone(),
                    a1: output[5].clone(),
                },
            },
            c1: GE6 {
                b0: GE2 {
                    a0: output[6].clone(),
                    a1: output[7].clone(),
                },
                b1: GE2 {
                    a0: output[8].clone(),
                    a1: output[9].clone(),
                },
                b2: GE2 {
                    a0: output[10].clone(),
                    a1: output[11].clone(),
                },
            },
        };
        let _x = self.mul(native, &div, y);
        self.assert_isequal(native, x, &_x);
        div
    }
    pub fn inverse<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, x: &GE12) -> GE12 {
        let inputs = vec![
            x.c0.b0.a0.clone(),
            x.c0.b0.a1.clone(),
            x.c0.b1.a0.clone(),
            x.c0.b1.a1.clone(),
            x.c0.b2.a0.clone(),
            x.c0.b2.a1.clone(),
            x.c1.b0.a0.clone(),
            x.c1.b0.a1.clone(),
            x.c1.b1.a0.clone(),
            x.c1.b1.a1.clone(),
            x.c1.b2.a0.clone(),
            x.c1.b2.a1.clone(),
        ];
        let output = self
            .ext6
            .ext2
            .curve_f
            .new_hint(native, "myhint.inversee12hint", 12, inputs);
        let inv = GE12 {
            c0: GE6 {
                b0: GE2 {
                    a0: output[0].clone(),
                    a1: output[1].clone(),
                },
                b1: GE2 {
                    a0: output[2].clone(),
                    a1: output[3].clone(),
                },
                b2: GE2 {
                    a0: output[4].clone(),
                    a1: output[5].clone(),
                },
            },
            c1: GE6 {
                b0: GE2 {
                    a0: output[6].clone(),
                    a1: output[7].clone(),
                },
                b1: GE2 {
                    a0: output[8].clone(),
                    a1: output[9].clone(),
                },
                b2: GE2 {
                    a0: output[10].clone(),
                    a1: output[11].clone(),
                },
            },
        };
        let one = self.one();
        let _one = self.mul(native, &inv, x);
        self.assert_isequal(native, &one, &_one);
        inv
    }
    pub fn copy<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, x: &GE12) -> GE12 {
        let inputs = vec![
            x.c0.b0.a0.clone(),
            x.c0.b0.a1.clone(),
            x.c0.b1.a0.clone(),
            x.c0.b1.a1.clone(),
            x.c0.b2.a0.clone(),
            x.c0.b2.a1.clone(),
            x.c1.b0.a0.clone(),
            x.c1.b0.a1.clone(),
            x.c1.b1.a0.clone(),
            x.c1.b1.a1.clone(),
            x.c1.b2.a0.clone(),
            x.c1.b2.a1.clone(),
        ];
        let output = self
            .ext6
            .ext2
            .curve_f
            .new_hint(native, "myhint.copye12hint", 12, inputs);
        let res = GE12 {
            c0: GE6 {
                b0: GE2 {
                    a0: output[0].clone(),
                    a1: output[1].clone(),
                },
                b1: GE2 {
                    a0: output[2].clone(),
                    a1: output[3].clone(),
                },
                b2: GE2 {
                    a0: output[4].clone(),
                    a1: output[5].clone(),
                },
            },
            c1: GE6 {
                b0: GE2 {
                    a0: output[6].clone(),
                    a1: output[7].clone(),
                },
                b1: GE2 {
                    a0: output[8].clone(),
                    a1: output[9].clone(),
                },
                b2: GE2 {
                    a0: output[10].clone(),
                    a1: output[11].clone(),
                },
            },
        };
        self.assert_isequal(native, x, &res);
        res
    }
    pub fn select<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        selector: Variable,
        z1: &GE12,
        z0: &GE12,
    ) -> GE12 {
        let c0 = self.ext6.select(native, selector, &z1.c0, &z0.c0);
        let c1 = self.ext6.select(native, selector, &z1.c1, &z0.c1);
        GE12 { c0, c1 }
    }

    /////// pairing ///////
    pub fn mul_by_014<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        z: &GE12,
        c0: &GE2,
        c1: &GE2,
    ) -> GE12 {
        let a = self.ext6.mul_by_01(native, &z.c0, c0, c1);
        let b = GE6 {
            b0: self.ext6.ext2.mul_by_non_residue(native, &z.c1.b2),
            b1: z.c1.b0.clone(),
            b2: z.c1.b1.clone(),
        };
        let one = self.ext6.ext2.one();
        let d = self.ext6.ext2.add(native, c1, &one);
        let zc1 = self.ext6.add(native, &z.c1, &z.c0);
        let zc1 = self.ext6.mul_by_01(native, &zc1, c0, &d);
        let tmp = self.ext6.add(native, &b, &a);
        let zc1 = self.ext6.sub(native, &zc1, &tmp);
        let zc0 = self.ext6.mul_by_non_residue(native, &b);
        let zc0 = self.ext6.add(native, &zc0, &a);
        GE12 { c0: zc0, c1: zc1 }
    }
    pub fn mul_014_by_014<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        d0: &GE2,
        d1: &GE2,
        c0: &GE2,
        c1: &GE2,
    ) -> [GE2; 5] {
        let x0 = self.ext6.ext2.mul(native, c0, d0);
        let x1 = self.ext6.ext2.mul(native, c1, d1);
        let x04 = self.ext6.ext2.add(native, c0, d0);
        let tmp = self.ext6.ext2.add(native, c0, c1);
        let x01 = self.ext6.ext2.add(native, d0, d1);
        let x01 = self.ext6.ext2.mul(native, &x01, &tmp);
        let tmp = self.ext6.ext2.add(native, &x1, &x0);
        let x01 = self.ext6.ext2.sub(native, &x01, &tmp);
        let x14 = self.ext6.ext2.add(native, c1, d1);
        let z_c0_b0 = self.ext6.ext2.non_residue(native);
        let z_c0_b0 = self.ext6.ext2.add(native, &z_c0_b0, &x0);
        [z_c0_b0, x01, x1, x04, x14]
    }
    pub fn expt<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, x: &GE12) -> GE12 {
        let z = self.cyclotomic_square(native, x);
        let z = self.mul(native, x, &z);
        let z = self.n_square_gs_with_hint(native, &z, 2);
        let z = self.mul(native, x, &z);
        let z = self.n_square_gs_with_hint(native, &z, 3);
        let z = self.mul(native, x, &z);
        let z = self.n_square_gs_with_hint(native, &z, 9);
        let z = self.mul(native, x, &z);
        let z = self.n_square_gs_with_hint(native, &z, 32);
        let z = self.mul(native, x, &z);
        let z = self.n_square_gs_with_hint(native, &z, 15);
        self.cyclotomic_square(native, &z)
    }
    pub fn n_square_gs<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        z: &GE12,
        n: usize,
    ) -> GE12 {
        let mut new_z = z.clone();
        for _ in 0..n {
            new_z = self.cyclotomic_square(native, &new_z);
        }
        new_z
    }
    pub fn n_square_gs_with_hint<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        z: &GE12,
        n: usize,
    ) -> GE12 {
        let mut copy_z = self.copy(native, z);
        for _ in 0..n - 1 {
            let z = self.cyclotomic_square(native, &copy_z);
            copy_z = self.copy(native, &z);
        }
        self.cyclotomic_square(native, &copy_z)
    }
    pub fn assert_final_exponentiation_is_one<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        x: &GE12,
    ) {
        let inputs = vec![
            x.c0.b0.a0.clone(),
            x.c0.b0.a1.clone(),
            x.c0.b1.a0.clone(),
            x.c0.b1.a1.clone(),
            x.c0.b2.a0.clone(),
            x.c0.b2.a1.clone(),
            x.c1.b0.a0.clone(),
            x.c1.b0.a1.clone(),
            x.c1.b1.a0.clone(),
            x.c1.b1.a1.clone(),
            x.c1.b2.a0.clone(),
            x.c1.b2.a1.clone(),
        ];
        let output = self
            .ext6
            .ext2
            .curve_f
            .new_hint(native, "myhint.finalexphint", 18, inputs);
        let residue_witness = GE12 {
            c0: GE6 {
                b0: GE2 {
                    a0: output[0].clone(),
                    a1: output[1].clone(),
                },
                b1: GE2 {
                    a0: output[2].clone(),
                    a1: output[3].clone(),
                },
                b2: GE2 {
                    a0: output[4].clone(),
                    a1: output[5].clone(),
                },
            },
            c1: GE6 {
                b0: GE2 {
                    a0: output[6].clone(),
                    a1: output[7].clone(),
                },
                b1: GE2 {
                    a0: output[8].clone(),
                    a1: output[9].clone(),
                },
                b2: GE2 {
                    a0: output[10].clone(),
                    a1: output[11].clone(),
                },
            },
        };
        let scaling_factor = GE12 {
            c0: GE6 {
                b0: GE2 {
                    a0: output[12].clone(),
                    a1: output[13].clone(),
                },
                b1: GE2 {
                    a0: output[14].clone(),
                    a1: output[15].clone(),
                },
                b2: GE2 {
                    a0: output[16].clone(),
                    a1: output[17].clone(),
                },
            },
            c1: self.zero().c1,
        };
        let t0 = self.frobenius(native, &residue_witness);
        let t1 = self.expt(native, &residue_witness);
        let t0 = self.mul(native, &t0, &t1);
        let t1 = self.mul(native, x, &scaling_factor);
        self.assert_isequal(native, &t0, &t1);
    }

    pub fn frobenius<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, x: &GE12) -> GE12 {
        let z00 = self.ext6.ext2.conjugate(native, &x.c0.b0);
        let z01 = self.ext6.ext2.conjugate(native, &x.c0.b1);
        let z02 = self.ext6.ext2.conjugate(native, &x.c0.b2);
        let z10 = self.ext6.ext2.conjugate(native, &x.c1.b0);
        let z11 = self.ext6.ext2.conjugate(native, &x.c1.b1);
        let z12 = self.ext6.ext2.conjugate(native, &x.c1.b2);

        let z01 = self.ext6.ext2.mul_by_non_residue1_power2(native, &z01);
        let z02 = self.ext6.ext2.mul_by_non_residue1_power4(native, &z02);
        let z10 = self.ext6.ext2.mul_by_non_residue1_power1(native, &z10);
        let z11 = self.ext6.ext2.mul_by_non_residue1_power3(native, &z11);
        let z12 = self.ext6.ext2.mul_by_non_residue1_power5(native, &z12);
        GE12 {
            c0: GE6 {
                b0: z00,
                b1: z01,
                b2: z02,
            },
            c1: GE6 {
                b0: z10,
                b1: z11,
                b2: z12,
            },
        }
    }
    pub fn frobenius_square<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, x: &GE12) -> GE12 {
        let z00 = x.c0.b0.clone();
        let z01 = self.ext6.ext2.mul_by_non_residue2_power2(native, &x.c0.b1);
        let z02 = self.ext6.ext2.mul_by_non_residue2_power4(native, &x.c0.b2);
        let z10 = self.ext6.ext2.mul_by_non_residue2_power1(native, &x.c1.b0);
        let z11 = self.ext6.ext2.mul_by_non_residue2_power3(native, &x.c1.b1);
        let z12 = self.ext6.ext2.mul_by_non_residue2_power5(native, &x.c1.b2);
        GE12 {
            c0: GE6 {
                b0: z00,
                b1: z01,
                b2: z02,
            },
            c1: GE6 {
                b0: z10,
                b1: z11,
                b2: z12,
            },
        }
    }
}
