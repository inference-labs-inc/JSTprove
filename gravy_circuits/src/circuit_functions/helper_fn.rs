use expander_compiler::frontend::*;
use ethnum::U256;
use ndarray::{ArrayD, Ix1, Ix2, Ix3, Ix4, IxDyn};

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


pub fn arrayd_to_vec1<T: Clone>(array: ArrayD<T>) -> Vec<T> {
    let array1 = array.into_dimensionality::<Ix1>()
        .expect("Expected 1D array");

    array1.to_vec()
}

pub fn vec1_to_arrayd<T>(v: Vec<T>) -> ArrayD<T> {
    let shape = vec![v.len()];
    ArrayD::from_shape_vec(IxDyn(&shape), v).unwrap()
}

pub fn arrayd_to_vec2<T: Clone>(array: ArrayD<T>) -> Vec<Vec<T>> {
    // Convert to Array2 first
    let array2 = array.into_dimensionality::<Ix2>()
        .expect("Expected 2D array");

    // Now extract rows
    array2
        .outer_iter()
        .map(|row| row.to_vec())
        .collect()
}

pub fn vec2_to_arrayd<T>(v: Vec<Vec<T>>) -> ArrayD<T> {
    let rows = v.len();
    let cols = v[0].len();

    let flat = v.into_iter().flatten().collect();
    ArrayD::from_shape_vec(IxDyn(&[rows, cols]), flat).unwrap()
}

pub fn arrayd_to_vec3<T: Clone>(array: ArrayD<T>) -> Vec<Vec<Vec<T>>> {
    let array3 = array.into_dimensionality::<Ix3>()
        .expect("Expected 3D array");

    array3
        .outer_iter()
        .map(|mat| mat.outer_iter().map(|row| row.to_vec()).collect())
        .collect()
}

pub fn vec3_to_arrayd<T>(v: Vec<Vec<Vec<T>>>) -> ArrayD<T> {
    let d1 = v.len();
    let d2 = v[0].len();
    let d3 = v[0][0].len();

    let flat = v.into_iter().flat_map(|v2| v2.into_iter().flatten()).collect();
    ArrayD::from_shape_vec(IxDyn(&[d1, d2, d3]), flat).unwrap()
}

pub fn arrayd_to_vec4(array: ArrayD<Variable>) -> Vec<Vec<Vec<Vec<Variable>>>> {
    let array4 = array.into_dimensionality::<Ix4>()
        .expect("Expected 4D array");

    let out = array4
        .outer_iter()
        .map(|cube| {
            cube.outer_iter()
                .map(|mat| mat.outer_iter().map(|row| row.to_vec()).collect())
                .collect()
        })
        .collect();
    out
}

pub fn vec4_to_arrayd(v: Vec<Vec<Vec<Vec<Variable>>>>) -> ArrayD<Variable> {
    let d1 = v.len();
    let d2 = v[0].len();
    let d3 = v[0][0].len();
    let d4 = v[0][0][0].len();

    let flat = v.into_iter()
        .flat_map(|v3| v3.into_iter()
            .flat_map(|v2| v2.into_iter()
                .flatten()))
        .collect();

    ArrayD::from_shape_vec(IxDyn(&[d1, d2, d3, d4]), flat).unwrap()
}


pub enum NestedVec<T> {
    D1(Vec<T>),
    D2(Vec<Vec<T>>),
    D3(Vec<Vec<Vec<T>>>),
    D4(Vec<Vec<Vec<Vec<T>>>>),
}

// pub fn arrayd_to_nested_vec<T: Clone>(array: ArrayD<T>) -> NestedVec<T> {
//     match array.ndim() {
//         1 => NestedVec::D1(arrayd_to_vec1(array)),
//         2 => NestedVec::D2(arrayd_to_vec2(array)),
//         3 => NestedVec::D3(arrayd_to_vec3(array)),
//         4 => NestedVec::D4(arrayd_to_vec4(array)),
//         _ => panic!("Unsupported dimensionality"),
//     }
// }

// fn nested_vec_to_arrayd<T>(nested: NestedVec<T>) -> ArrayD<T> {
//     match nested {
//         NestedVec::D1(v) => vec1_to_arrayd(v),
//         NestedVec::D2(v) => vec2_to_arrayd(v),
//         NestedVec::D3(v) => vec3_to_arrayd(v),
//         NestedVec::D4(v) => vec4_to_arrayd(v),
//     }
// }