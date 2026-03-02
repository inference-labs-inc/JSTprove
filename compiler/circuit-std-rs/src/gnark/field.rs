use crate::gnark::element::{new_internal_element, value_of, Element};
use crate::gnark::emparam::FieldParams;
use crate::gnark::utils::{hash_to_fp_variable, nb_multiplication_res_limbs, sub_padding};
use crate::logup::LogUpRangeProofTable;
use crate::utils::simple_select;
use expander_compiler::frontend::{Config, RootAPI, Variable};
use num_bigint::BigInt;
use num_traits::Signed;
use num_traits::ToPrimitive;
use num_traits::Zero;
use std::collections::HashMap;

pub struct MulCheck<T: FieldParams> {
    a: Element<T>,
    b: Element<T>,
    r: Element<T>,
    k: Element<T>,
    c: Element<T>,
    p: Element<T>,
}
impl<T: FieldParams> MulCheck<T> {
    pub fn eval_round1<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, at: Vec<Variable>) {
        self.c = eval_with_challenge(native, self.c.clone(), at.clone());
        self.r = eval_with_challenge(native, self.r.clone(), at.clone());
        self.k = eval_with_challenge(native, self.k.clone(), at.clone());
        if !self.p.is_empty() {
            self.p = eval_with_challenge(native, self.p.clone(), at.clone());
        }
    }
    pub fn eval_round2<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, at: Vec<Variable>) {
        self.a = eval_with_challenge(native, self.a.clone(), at.clone());
        self.b = eval_with_challenge(native, self.b.clone(), at.clone());
    }
    pub fn check<C: Config, B: RootAPI<C>>(&self, native: &mut B, pval: Variable, ccoef: Variable) {
        let mut new_peval = pval;
        if !self.p.is_empty() {
            new_peval = self.p.evaluation
        };
        let ls = native.mul(self.a.evaluation, self.b.evaluation);
        let rs_tmp1 = native.mul(new_peval, self.k.evaluation);
        let rs_tmp2 = native.mul(self.c.evaluation, ccoef);
        let rs_tmp3 = native.add(self.r.evaluation, rs_tmp1);
        let rs = native.add(rs_tmp3, rs_tmp2);
        native.assert_is_equal(ls, rs);
    }
    pub fn clean_evaluations(&mut self) {
        self.a.evaluation = Variable::default();
        self.a.is_evaluated = false;
        self.b.evaluation = Variable::default();
        self.b.is_evaluated = false;
        self.r.evaluation = Variable::default();
        self.r.is_evaluated = false;
        self.k.evaluation = Variable::default();
        self.k.is_evaluated = false;
        self.c.evaluation = Variable::default();
        self.c.is_evaluated = false;
        self.p.evaluation = Variable::default();
        self.p.is_evaluated = false;
    }
}
pub struct GField<T: FieldParams> {
    _f_params: T,
    max_of: u32,
    n_const: Element<T>,
    nprev_const: Element<T>,
    pub zero_const: Element<T>,
    pub one_const: Element<T>,
    short_one_const: Element<T>,
    constrained_limbs: HashMap<usize, ()>,
    pub table: LogUpRangeProofTable,
    //checker: Box<dyn Rangechecker>, we use lookup rangeproof instead
    mul_checks: Vec<MulCheck<T>>,
}

impl<T: FieldParams> GField<T> {
    pub fn new<C: Config, B: RootAPI<C>>(native: &mut B, f_params: T) -> Self {
        let mut field = GField {
            _f_params: f_params,
            max_of: 30 - 2 - T::bits_per_limb(),
            n_const: Element::<T>::default(),
            nprev_const: Element::<T>::default(),
            zero_const: Element::<T>::default(),
            one_const: Element::<T>::default(),
            short_one_const: Element::<T>::default(),
            constrained_limbs: HashMap::new(),
            table: LogUpRangeProofTable::new(8),
            mul_checks: Vec::new(),
        };
        field.n_const = value_of::<C, B, T>(native, Box::new(T::modulus()));
        field.nprev_const = value_of::<C, B, T>(native, Box::new(T::modulus() - 1));
        field.zero_const = value_of::<C, B, T>(native, Box::new(0));
        field.one_const = value_of::<C, B, T>(native, Box::new(1));
        field.short_one_const = new_internal_element::<T>(vec![native.constant(1); 1], 0);
        field.table.initial(native);
        field
    }
    pub fn max_overflow(&self) -> u64 {
        30 - 2 - 8
    }
    pub fn is_zero<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        a: &Element<T>,
    ) -> Variable {
        let ca = self.reduce(native, a, false);
        let mut res0;
        let total_overflow = ca.limbs.len() as i32 - 1;
        if total_overflow > self.max_overflow() as i32 {
            res0 = native.is_zero(ca.limbs[0]);
            for i in 1..ca.limbs.len() {
                let tmp = native.is_zero(ca.limbs[i]);
                res0 = native.mul(res0, tmp);
            }
        } else {
            let mut limb_sum = ca.limbs[0];
            for i in 1..ca.limbs.len() {
                limb_sum = native.add(limb_sum, ca.limbs[i]);
            }
            res0 = native.is_zero(limb_sum);
        }
        res0
    }
    pub fn get_element_sign<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        x: &Element<T>,
    ) -> Variable {
        native.to_binary(x.limbs[0], 30)[0]
    }
    pub fn select<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        selector: Variable,
        a: &Element<T>,
        b: &Element<T>,
    ) -> Element<T> {
        self.enforce_width_conditional(native, &a.clone());
        self.enforce_width_conditional(native, &b.clone());
        let overflow = std::cmp::max(a.overflow, b.overflow);
        let nb_limbs = std::cmp::max(a.limbs.len(), b.limbs.len());
        let mut limbs = vec![native.constant(0); nb_limbs];
        let mut normalize = |limbs: Vec<Variable>| -> Vec<Variable> {
            if limbs.len() < nb_limbs {
                let mut tail = vec![native.constant(0); nb_limbs - limbs.len()];
                for cur_tail in &mut tail {
                    *cur_tail = native.constant(0);
                }
                return limbs.iter().chain(tail.iter()).cloned().collect();
            }
            limbs
        };
        let a_norm_limbs = normalize(a.limbs.clone());
        let b_norm_limbs = normalize(b.limbs.clone());
        for i in 0..limbs.len() {
            limbs[i] = simple_select(native, selector, a_norm_limbs[i], b_norm_limbs[i]);
        }
        new_internal_element::<T>(limbs, overflow)
    }
    pub fn enforce_width_conditional<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        a: &Element<T>,
    ) -> bool {
        let mut did_constrain = false;
        if a.internal {
            return false;
        }
        for i in 0..a.limbs.len() {
            let value_id = a.limbs[i].id();
            if let std::collections::hash_map::Entry::Vacant(e) =
                self.constrained_limbs.entry(value_id)
            {
                e.insert(());
            } else {
                did_constrain = true;
            }
        }
        self.enforce_width(native, a, true);
        did_constrain
    }
    pub fn enforce_width<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        a: &Element<T>,
        mod_width: bool,
    ) {
        for i in 0..a.limbs.len() {
            let mut limb_nb_bits = T::bits_per_limb() as u64;
            if mod_width && i == a.limbs.len() - 1 {
                limb_nb_bits = ((T::modulus().bits() - 1) % T::bits_per_limb() as u64) + 1;
            }
            //range check
            self.table
                .rangeproof(native, a.limbs[i], limb_nb_bits as usize);
        }
    }
    pub fn wrap_hint<C: Config, B: RootAPI<C>>(
        &self,
        native: &mut B,
        nonnative_inputs: Vec<Element<T>>,
    ) -> Vec<Variable> {
        let mut res = vec![
            native.constant(T::bits_per_limb()),
            native.constant(T::nb_limbs()),
        ];
        res.extend(self.n_const.limbs.clone());
        res.push(native.constant(nonnative_inputs.len() as u32));
        for nonnative_input in &nonnative_inputs {
            res.push(native.constant(nonnative_input.limbs.len() as u32));
            res.extend(nonnative_input.limbs.clone());
        }
        res
    }
    pub fn new_hint<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        hf_name: &str,
        nb_outputs: usize,
        inputs: Vec<Element<T>>,
    ) -> Vec<Element<T>> {
        let native_inputs = self.wrap_hint(native, inputs);
        let nb_native_outputs = T::nb_limbs() as usize * nb_outputs;
        let native_outputs = native.new_hint(hf_name, &native_inputs, nb_native_outputs);
        let mut outputs = vec![];
        for i in 0..nb_outputs {
            let tmp_output = self.pack_limbs(
                native,
                native_outputs[i * T::nb_limbs() as usize..(i + 1) * T::nb_limbs() as usize]
                    .to_vec(),
                true,
            );
            outputs.push(tmp_output);
        }
        outputs
    }
    pub fn pack_limbs<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        limbs: Vec<Variable>,
        strict: bool,
    ) -> Element<T> {
        let e = new_internal_element::<T>(limbs, 0);
        self.enforce_width(native, &e, strict);
        e
    }
    pub fn reduce<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        a: &Element<T>,
        strict: bool,
    ) -> Element<T> {
        self.enforce_width_conditional(native, a);
        if a.mod_reduced {
            return a.clone();
        }
        if !strict && a.overflow == 0 {
            return a.clone();
        }
        let p = Element::<T>::default();
        let one = self.one_const.clone();
        self.mul_mod(native, a, &one, 0, &p).clone()
    }
    pub fn mul_mod<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        a: &Element<T>,
        b: &Element<T>,
        _: usize,
        p: &Element<T>,
    ) -> Element<T> {
        self.enforce_width_conditional(native, a);
        self.enforce_width_conditional(native, b);
        let (k, r, c) = self.call_mul_hint(native, a, b, true);
        let mc = MulCheck {
            a: a.clone(),
            b: b.clone(),
            c,
            k,
            r: r.clone(),
            p: p.clone(),
        };
        self.mul_checks.push(mc);
        r
    }
    pub fn mul_pre_cond(&self, a: &Element<T>, b: &Element<T>) -> u32 {
        let nb_res_limbs = nb_multiplication_res_limbs(a.limbs.len(), b.limbs.len());
        let nb_limbs_overflow = if nb_res_limbs > 0 {
            (nb_res_limbs as f64).log2().ceil() as u32
        } else {
            1
        };
        T::bits_per_limb() + nb_limbs_overflow + a.overflow + b.overflow
    }
    pub fn call_mul_hint<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        a: &Element<T>,
        b: &Element<T>,
        is_mul_mod: bool,
    ) -> (Element<T>, Element<T>, Element<T>) {
        let next_overflow = self.mul_pre_cond(a, b);
        let next_overflow = if !is_mul_mod {
            a.overflow
        } else {
            next_overflow
        };
        let nb_limbs = T::nb_limbs() as usize;
        let nb_bits = T::bits_per_limb() as usize;
        let modbits = T::modulus().bits() as usize;
        let a_limbs_len = a.limbs.len();
        let b_limbs_len = b.limbs.len();
        let nb_quo_limbs = (nb_multiplication_res_limbs(a_limbs_len, b_limbs_len) * nb_bits
            + next_overflow as usize
            + 1
            - modbits
            + nb_bits
            - 1)
            / nb_bits;
        let nb_rem_limbs = nb_limbs;
        let nb_carry_limbs = std::cmp::max(
            nb_multiplication_res_limbs(a_limbs_len, b_limbs_len),
            nb_multiplication_res_limbs(nb_quo_limbs, nb_limbs),
        ) - 1;
        let mut hint_inputs = vec![
            native.constant(nb_bits as u32),
            native.constant(nb_limbs as u32),
            native.constant(a.limbs.len() as u32),
            native.constant(nb_quo_limbs as u32),
        ];
        let modulus_limbs = self.n_const.limbs.clone();
        hint_inputs.extend(modulus_limbs);
        hint_inputs.extend(a.limbs.clone());
        hint_inputs.extend(b.limbs.clone());
        let ret = native.new_hint(
            "myhint.mulhint",
            &hint_inputs,
            nb_quo_limbs + nb_rem_limbs + nb_carry_limbs,
        );
        let quo = self.pack_limbs(native, ret[..nb_quo_limbs].to_vec(), false);
        let rem = if is_mul_mod {
            self.pack_limbs(
                native,
                ret[nb_quo_limbs..nb_quo_limbs + nb_rem_limbs].to_vec(),
                true,
            )
        } else {
            Element::default()
        };
        let carries = new_internal_element::<T>(ret[nb_quo_limbs + nb_rem_limbs..].to_vec(), 0);
        (quo, rem, carries)
    }
    pub fn check_zero<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        a: Element<T>,
        p: Option<Element<T>>,
    ) {
        self.enforce_width_conditional(native, &a.clone());
        let b = self.short_one_const.clone();
        let (k, r, c) = self.call_mul_hint(native, &a, &b, false);
        let mc = MulCheck {
            a,
            b,
            c,
            k,
            r: r.clone(),
            p: p.unwrap_or_default(),
        };
        self.mul_checks.push(mc);
    }
    pub fn copy<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, x: &Element<T>) -> Element<T> {
        let inputs = vec![x.clone()];
        let output = self.new_hint(native, "myhint.copyelementhint", 1, inputs);
        let res = output[0].clone();
        self.assert_is_equal(native, x, &res);
        res
    }
    pub fn assert_is_equal<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        a: &Element<T>,
        b: &Element<T>,
    ) {
        self.enforce_width_conditional(native, a);
        self.enforce_width_conditional(native, b);
        let diff = self.sub(native, b, a);
        self.check_zero(native, diff, None);
    }
    pub fn add<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        a: &Element<T>,
        b: &Element<T>,
    ) -> Element<T> {
        self.enforce_width_conditional(native, &a.clone());
        self.enforce_width_conditional(native, &b.clone());
        let mut new_a = a.clone();
        let mut new_b = b.clone();
        if a.overflow + 1 > self.max_of {
            new_a = self.reduce(native, a, false);
        }
        if b.overflow + 1 > self.max_of {
            new_b = self.reduce(native, b, false);
        }
        let next_overflow = std::cmp::max(new_a.overflow, new_b.overflow) + 1;
        let nb_limbs = std::cmp::max(new_a.limbs.len(), new_b.limbs.len());
        let mut limbs = vec![native.constant(0); nb_limbs];
        for (i, limb) in limbs.iter_mut().enumerate() {
            if i < new_a.limbs.len() {
                *limb = native.add(*limb, new_a.limbs[i]);
            }
            if i < new_b.limbs.len() {
                *limb = native.add(*limb, new_b.limbs[i]);
            }
        }
        new_internal_element::<T>(limbs, next_overflow)
    }
    pub fn sub<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        a: &Element<T>,
        b: &Element<T>,
    ) -> Element<T> {
        self.enforce_width_conditional(native, &a.clone());
        self.enforce_width_conditional(native, &b.clone());
        let mut new_a = a.clone();
        let mut new_b = b.clone();
        if a.overflow + 1 > self.max_of {
            new_a = self.reduce(native, a, false);
        }
        if b.overflow + 2 > self.max_of {
            new_b = self.reduce(native, b, false);
        }
        let next_overflow = std::cmp::max(new_a.overflow, new_b.overflow + 1) + 1;
        let nb_limbs = std::cmp::max(new_a.limbs.len(), new_b.limbs.len());
        let pad_limbs = sub_padding(
            &T::modulus(),
            T::bits_per_limb(),
            new_b.overflow,
            nb_limbs as u32,
        );
        let mut limbs = vec![native.constant(0); nb_limbs];
        for i in 0..limbs.len() {
            limbs[i] = native.constant(pad_limbs[i].to_u64().unwrap() as u32);
            if i < new_a.limbs.len() {
                limbs[i] = native.add(limbs[i], new_a.limbs[i]);
            }
            if i < new_b.limbs.len() {
                limbs[i] = native.sub(limbs[i], new_b.limbs[i]);
            }
        }
        new_internal_element::<T>(limbs, next_overflow)
    }
    pub fn neg<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, a: &Element<T>) -> Element<T> {
        let zero = self.zero_const.clone();
        self.sub(native, &zero, a)
    }
    pub fn mul<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        a: &Element<T>,
        b: &Element<T>,
    ) -> Element<T> {
        self.enforce_width_conditional(native, a);
        self.enforce_width_conditional(native, b);

        //calculate a*b's overflow and reduce if necessary
        let mut next_overflow = self.mul_pre_cond(a, b);
        let mut new_a = a.clone();
        let mut new_b = b.clone();
        if next_overflow > self.max_of {
            if a.overflow < b.overflow {
                new_b = self.reduce(native, b, false);
            } else {
                new_a = self.reduce(native, a, false);
            }
        }
        next_overflow = self.mul_pre_cond(&new_a, &new_b);
        if next_overflow > self.max_of {
            if new_a.overflow < new_b.overflow {
                new_b = self.reduce(native, &new_b, false);
            } else {
                new_a = self.reduce(native, &new_a, false);
            }
        }

        //calculate a*b
        self.mul_mod(native, &new_a, &new_b, 0, &Element::<T>::default())
    }
    pub fn div<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        a: &Element<T>,
        b: &Element<T>,
    ) -> Element<T> {
        self.enforce_width_conditional(native, a);
        self.enforce_width_conditional(native, b);
        //calculate a/b's overflow and reduce if necessary
        let zero_element = self.zero_const.clone();
        let mut mul_of = self.mul_pre_cond(&zero_element, b);
        let mut new_a = a.clone();
        let mut new_b = b.clone();
        if mul_of > self.max_of {
            new_b = self.reduce(native, &new_b, false);
            mul_of = 0;
        }
        if new_a.overflow + 1 > self.max_of {
            new_a = self.reduce(native, &new_a, false);
        }
        if mul_of + 2 > self.max_of {
            new_b = self.reduce(native, &new_b, false);
        }

        //calculate a/b
        let div = self.compute_division_hint(native, a.limbs.clone(), b.limbs.clone());
        let e = self.pack_limbs(native, div, true);
        let res = self.mul(native, &e, &new_b);
        self.assert_is_equal(native, &res, &new_a);
        e
    }
    pub fn inverse<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        b: &Element<T>,
    ) -> Element<T> {
        self.enforce_width_conditional(native, b);
        //calculate 1/b's overflow and reduce if necessary
        let zero_element = self.zero_const.clone();
        let mut mul_of = self.mul_pre_cond(&zero_element, b);
        let mut new_b = b.clone();
        if mul_of > self.max_of {
            new_b = self.reduce(native, &new_b, false);
            mul_of = 0;
        }
        if mul_of + 2 > self.max_of {
            new_b = self.reduce(native, &new_b, false);
        }
        // let next_overflow = std::cmp::max(new_a.overflow, new_b.overflow+1) + 1;

        //calculate 1/b
        let inv = self.compute_inverse_hint(native, b.limbs.clone());
        let e = self.pack_limbs(native, inv, true);
        let res = self.mul(native, &e, &new_b);
        let one = self.one_const.clone();
        self.assert_is_equal(native, &res, &one);
        e
    }
    pub fn compute_inverse_hint<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        in_limbs: Vec<Variable>,
    ) -> Vec<Variable> {
        let mut hint_inputs = vec![
            native.constant(T::bits_per_limb()),
            native.constant(T::nb_limbs()),
        ];
        let modulus_limbs = self.n_const.limbs.clone();
        hint_inputs.extend(modulus_limbs);
        hint_inputs.extend(in_limbs);
        native.new_hint("myhint.invhint", &hint_inputs, T::nb_limbs() as usize)
    }
    pub fn compute_division_hint<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        nom_limbs: Vec<Variable>,
        denom_limbs: Vec<Variable>,
    ) -> Vec<Variable> {
        let mut hint_inputs = vec![
            native.constant(T::bits_per_limb()),
            native.constant(T::nb_limbs()),
            native.constant(denom_limbs.len() as u32),
            native.constant(nom_limbs.len() as u32),
        ];
        let modulus_limbs = self.n_const.limbs.clone();
        hint_inputs.extend(modulus_limbs);
        hint_inputs.extend(nom_limbs);
        hint_inputs.extend(denom_limbs);
        native.new_hint("myhint.divhint", &hint_inputs, T::nb_limbs() as usize)
    }
    pub fn mul_const<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        a: &Element<T>,
        c: BigInt,
    ) -> Element<T> {
        if c.is_negative() {
            let neg_a = self.neg(native, a);
            return self.mul_const(native, &neg_a, -c);
        } else if c.is_zero() {
            return self.zero_const.clone();
        }
        let cbl = c.bits();
        if cbl > self.max_overflow() {
            panic!(
                "constant bit length {} exceeds max {}",
                cbl,
                self.max_overflow()
            );
        }
        let next_overflow = a.overflow + cbl as u32;
        let mut new_a = a.clone();
        if next_overflow > self.max_of {
            new_a = self.reduce(native, a, false);
        }
        let mut limbs = vec![native.constant(0); new_a.limbs.len()];
        for i in 0..new_a.limbs.len() {
            limbs[i] = native.mul(new_a.limbs[i], c.to_u64().unwrap() as u32);
        }
        new_internal_element::<T>(limbs, new_a.overflow + cbl as u32)
    }
    pub fn check_mul<C: Config, B: RootAPI<C>>(&mut self, native: &mut B) {
        let commitment = native.get_random_value();
        // let commitment = native.constant(1); //TBD
        let mut coefs_len = T::nb_limbs() as usize;
        for i in 0..self.mul_checks.len() {
            coefs_len = std::cmp::max(coefs_len, self.mul_checks[i].a.limbs.len());
            coefs_len = std::cmp::max(coefs_len, self.mul_checks[i].b.limbs.len());
            coefs_len = std::cmp::max(coefs_len, self.mul_checks[i].c.limbs.len());
            coefs_len = std::cmp::max(coefs_len, self.mul_checks[i].k.limbs.len());
        }
        let mut at = vec![commitment; coefs_len];
        for i in 1..at.len() {
            at[i] = native.mul(at[i - 1], commitment);
        }
        for i in 0..self.mul_checks.len() {
            self.mul_checks[i].eval_round1(native, at.clone());
        }
        for i in 0..self.mul_checks.len() {
            self.mul_checks[i].eval_round2(native, at.clone());
        }
        let pval = eval_with_challenge(native, self.n_const.clone(), at.clone());
        let coef = BigInt::from(1) << T::bits_per_limb();
        let ccoef = native.sub(coef.to_u64().unwrap() as u32, commitment);
        for i in 0..self.mul_checks.len() {
            self.mul_checks[i].check(native, pval.evaluation, ccoef);
        }
        for i in 0..self.mul_checks.len() {
            self.mul_checks[i].clean_evaluations();
        }
    }
    pub fn hash_to_fp<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        msg: &[Variable],
        len: usize,
    ) -> Vec<Element<T>> {
        let signature_dst: &[u8] = b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_";
        let mut dst = vec![];
        for c in signature_dst {
            dst.push(native.constant(*c as u32));
        }
        let hm = hash_to_fp_variable(native, msg, &dst, len);
        let mut xs_limbs = vec![];
        let n = T::bits_per_limb();
        if n != 8 {
            panic!("only support 8 bits per limb for now");
        }
        let k = T::nb_limbs() as usize;
        if k > 64 {
            panic!("only support <= 64 limbs for now");
        }
        for element in &hm {
            let mut x = vec![];
            for j in 0..k {
                x.push(element[k - 1 - j]);
            }
            xs_limbs.push(x);
        }
        let shift = value_of(
            native,
            Box::new("340282366920938463463374607431768211456".to_string()),
        );
        let mut x_elements = vec![];
        for i in 0..xs_limbs.len() {
            let mut x_element = new_internal_element(xs_limbs[i].clone(), 0);
            x_element = self.mul(native, &x_element, &shift);
            let mut x_rem = vec![native.constant(0); k];
            for (j, rem) in x_rem.iter_mut().enumerate().take(k) {
                if j < (64 - k) {
                    *rem = hm[i][63 - j];
                }
            }
            x_element = self.add(native, &x_element, &new_internal_element(x_rem, 0));
            x_element = self.reduce(native, &x_element, true);
            x_elements.push(x_element);
        }
        x_elements
    }
}
pub fn eval_with_challenge<C: Config, B: RootAPI<C>, T: FieldParams>(
    native: &mut B,
    a: Element<T>,
    at: Vec<Variable>,
) -> Element<T> {
    if a.is_evaluated {
        return a;
    }
    if (at.len() as i64) < (a.limbs.len() as i64) - 1 {
        panic!("evaluation powers less than limbs");
    }
    let mut sum = native.constant(0);
    if !a.limbs.is_empty() {
        sum = native.mul(a.limbs[0], 1);
    }
    for i in 1..a.limbs.len() {
        let tmp = native.mul(a.limbs[i], at[i - 1]);
        sum = native.add(sum, tmp);
    }
    let mut ret = a.clone();
    ret.is_evaluated = true;
    ret.evaluation = sum;
    ret
}
