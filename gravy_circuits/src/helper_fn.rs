use expander_compiler::frontend::*;

pub fn load_circuit_constant<C: Config>(api: &mut API<C>, x: i64) -> Variable{
    if x < 0 {
        let y = api.constant(x.abs() as u32);
        api.neg(y)
    } else {
        api.constant(x.abs() as u32) // For values greater than 100
    }
}