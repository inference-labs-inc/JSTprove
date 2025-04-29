use expander_compiler::frontend::*;
use ethnum::U256;

/// Load in circuit constant given i64, negative values are represented by p-x and positive values are x
pub fn load_circuit_constant<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: i64,
) -> Variable {
    if x < 0 {
        // let y = api.constant(x.abs() as u32);
        let y = api.constant(CircuitField::<C>::from_u256(U256::from(x.abs() as u64)));
        api.neg(y)
    } else {
        // api.constant(x.abs() as u32) // For values greater than 100
        api.constant(CircuitField::<C>::from_u256(U256::from(x.abs() as u64)))
    }
}
/// Convert 4d array to 4d vectors
pub fn four_d_array_to_vec<const K: usize, const L: usize, const M: usize, const N: usize>(
    array: [[[[Variable; N]; M]; L]; K],
) -> Vec<Vec<Vec<Vec<Variable>>>> {
    array
        .iter()
        .map(|matrix_k| {
            matrix_k
                .iter()
                .map(|matrix_l| matrix_l.iter().map(|row_m| row_m.to_vec()).collect())
                .collect()
        })
        .collect()
}
///Read 4d weights vectors of i64 into circuit constants
pub fn read_4d_weights<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    weights_data: &Vec<Vec<Vec<Vec<i64>>>>,
) -> Vec<Vec<Vec<Vec<Variable>>>> {
    let weights: Vec<Vec<Vec<Vec<Variable>>>> = weights_data
        .clone()
        .into_iter()
        .map(|dim1| {
            dim1.into_iter()
                .map(|dim2| {
                    dim2.into_iter()
                        .map(|dim3| {
                            dim3.into_iter()
                                .map(|x| load_circuit_constant(api, x))
                                .collect()
                        })
                        .collect()
                })
                .collect()
        })
        .collect();
    weights
}

/// Read 2d weights vectors of i64 into circuit constants
pub fn read_2d_weights<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    weights_data: &Vec<Vec<i64>>,
) -> Vec<Vec<Variable>> {
    let weights: Vec<Vec<Variable>> = weights_data
        .clone()
        .into_iter()
        .map(|dim1| {
            dim1.into_iter()
                .map(|x| load_circuit_constant(api, x))
                                .collect()
                })
                .collect();
    weights
}

/// Convert 2d array to 2d vectors
pub fn two_d_array_to_vec<const M: usize, const N: usize>(
    matrix: [[Variable; N]; M],
) -> Vec<Vec<Variable>> {
    matrix.iter().map(|row| row.to_vec()).collect()
}