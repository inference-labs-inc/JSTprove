use expander_compiler::frontend::{Config, HintRegistry, RootAPI, Variable, M31};

use crate::{
    gnark::hints::{
        copy_e12_hint, copy_e2_hint, copy_element_hint, copy_vars_hint, div_e12_hint, div_e2_hint,
        div_e6_by_6_hint, div_e6_hint, div_hint, final_exp_hint, get_e2_sqrt_hint,
        get_element_sqrt_hint, get_sqrt_x0x1_fq2_new_hint, get_sqrt_x0x1_fq_new_hint, inv_hint,
        inverse_e12_hint, inverse_e2_hint, inverse_e6_hint, mul_hint, simple_rangecheck_hint,
    },
    logup::{query_count_by_key_hint, query_count_hint, rangeproof_hint},
};

pub fn simple_select<C: Config, B: RootAPI<C>>(
    native: &mut B,
    selector: Variable,
    a: Variable,
    b: Variable,
) -> Variable {
    let tmp = native.sub(a, b);
    let tmp2 = native.mul(tmp, selector);
    native.add(b, tmp2)
}

//return i0 if selector0 and selector 1 are 0
//return i1 if selector0 is 1 and selector1 is 0
//return i2 if selector0 is 0 and selector1 is 1
//return d if selector0 and selector1 are 1
pub fn simple_lookup2<C: Config, B: RootAPI<C>>(
    native: &mut B,
    selector0: Variable,
    selector1: Variable,
    i0: Variable,
    i1: Variable,
    i2: Variable,
    i3: Variable,
) -> Variable {
    let tmp0 = simple_select(native, selector0, i1, i0);
    let tmp1 = simple_select(native, selector0, i3, i2);
    simple_select(native, selector1, tmp1, tmp0)
}

pub fn register_hint(hint_registry: &mut HintRegistry<M31>) {
    hint_registry.register("myhint.mulhint", mul_hint);
    hint_registry.register("myhint.simple_rangecheck_hint", simple_rangecheck_hint);
    hint_registry.register("myhint.querycounthint", query_count_hint);
    hint_registry.register("myhint.querycountbykeyhint", query_count_by_key_hint);
    hint_registry.register("myhint.copyvarshint", copy_vars_hint);
    hint_registry.register("myhint.divhint", div_hint);
    hint_registry.register("myhint.invhint", inv_hint);
    hint_registry.register("myhint.copyelementhint", copy_element_hint);
    hint_registry.register("myhint.dive2hint", div_e2_hint);
    hint_registry.register("myhint.inversee2hint", inverse_e2_hint);
    hint_registry.register("myhint.copye2hint", copy_e2_hint);
    hint_registry.register("myhint.dive6hint", div_e6_hint);
    hint_registry.register("myhint.inversee6hint", inverse_e6_hint);
    hint_registry.register("myhint.dive6by6hint", div_e6_by_6_hint);
    hint_registry.register("myhint.dive12hint", div_e12_hint);
    hint_registry.register("myhint.inversee12hint", inverse_e12_hint);
    hint_registry.register("myhint.copye12hint", copy_e12_hint);
    hint_registry.register("myhint.finalexphint", final_exp_hint);
    hint_registry.register("myhint.rangeproofhint", rangeproof_hint);
    hint_registry.register("myhint.getsqrtx0x1fq2newhint", get_sqrt_x0x1_fq2_new_hint);
    hint_registry.register("myhint.getsqrtx0x1fqnewhint", get_sqrt_x0x1_fq_new_hint);
    hint_registry.register("myhint.getelementsqrthint", get_element_sqrt_hint);
    hint_registry.register("myhint.gete2sqrthint", get_e2_sqrt_hint);
}
