use expander_compiler::frontend::*;

pub fn load_circuit_constant<C: Config>(api: &mut API<C>, x: i64) -> Variable{
    if x < 0 {
        let y = api.constant(x.abs() as u32);
        api.neg(y)
    } else {
        api.constant(x.abs() as u32) // For values greater than 100
    }
}


pub fn four_d_array_to_vec<const K: usize, const L: usize, const M: usize, const N: usize>(
    array: [[[ [Variable; N]; M ]; L]; K]
) -> Vec<Vec<Vec<Vec<Variable>>>> {
    array.iter()
        .map(|matrix_k| 
            matrix_k.iter()
                .map(|matrix_l| 
                    matrix_l.iter()
                        .map(|row_m| 
                            row_m.to_vec()
                        )
                        .collect()
                )
                .collect()
        )
        .collect()
}
