use std::ops::Neg;

use expander_compiler::frontend::*;
use ethnum::U256;
use gkr_engine::{FieldEngine, GKREngine};
use ndarray::{ArrayD, Ix1, Ix2, Ix3, Ix4, Ix5, IxDyn};
use serde_json::Value;

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
    // panic!("{}", array.ndim());
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

pub fn arrayd_to_vec5(array: ArrayD<Variable>) -> Vec<Vec<Vec<Vec<Vec<Variable>>>>> {
    let array5 = array.into_dimensionality::<Ix5>()
        .expect("Expected 5D array");

    array5
        .outer_iter()
        .map(|d4| {
            d4.outer_iter()
                .map(|d3| {
                    d3.outer_iter()
                        .map(|d2| {
                            d2.outer_iter()
                                .map(|d1| d1.to_vec())
                                .collect()
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

pub fn vec5_to_arrayd(v: Vec<Vec<Vec<Vec<Vec<Variable>>>>>) -> ArrayD<Variable> {
    let d1 = v.len();
    let d2 = v[0].len();
    let d3 = v[0][0].len();
    let d4 = v[0][0][0].len();
    let d5 = v[0][0][0][0].len();

    let flat = v.into_iter()
        .flat_map(|v4| v4.into_iter()
            .flat_map(|v3| v3.into_iter()
                .flat_map(|v2| v2.into_iter()
                    .flatten())))
        .collect();

    ArrayD::from_shape_vec(IxDyn(&[d1, d2, d3, d4, d5]), flat).unwrap()
}

/*
    For scaling functions
*/
pub fn scale_4d_vector(data: &Vec<Vec<Vec<Vec<f64>>>>, x: f64) -> Vec<Vec<Vec<Vec<i64>>>> {
    data.iter()
        .map(|v3| {
            v3.iter()
                .map(|v2| {
                    v2.iter()
                        .map(|v1| {
                            v1.iter()
                                .map(|&val| (val * x) as i64)
                                .collect()
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}




/*
    For witness generation, putting values into the circuit
*/

fn build_1d_vec<C: Config>(shape: [usize; 1]) -> Vec<CircuitField<C>> {
    vec![CircuitField::<C>::zero(); shape[0]]
}

fn build_2d_vec<C: Config>(shape: [usize; 2]) -> Vec<Vec<CircuitField<C>>> {
    vec![vec![CircuitField::<C>::zero(); shape[1]]; shape[0]]
}

fn build_3d_vec<C: Config>(shape: [usize; 3]) -> Vec<Vec<Vec<CircuitField<C>>>> {
    vec![vec![vec![CircuitField::<C>::zero(); shape[2]]; shape[1]]; shape[0]]
}

fn build_4d_vec<C: Config>(shape: [usize; 4]) -> Vec<Vec<Vec<Vec<CircuitField<C>>>>> {
    vec![vec![vec![vec![CircuitField::<C>::zero(); shape[3]]; shape[2]]; shape[1]]; shape[0]]
}

fn build_5d_vec<C: Config>(shape: [usize; 5]) -> Vec<Vec<Vec<Vec<Vec<CircuitField<C>>>>>> {
    vec![vec![vec![vec![vec![CircuitField::<C>::zero(); shape[4]]; shape[3]]; shape[2]]; shape[1]]; shape[0]]
}

fn build_nd_vec<C: Config>(shape: &[usize]) -> Result<AnyDimVec<CircuitField<C>>, String> {
    match shape.len() {
        1 => Ok(AnyDimVec::D1(build_1d_vec::<C>([shape[0]]))),
        2 => Ok(AnyDimVec::D2(build_2d_vec::<C>([shape[0], shape[1]]))),
        3 => Ok(AnyDimVec::D3(build_3d_vec::<C>([shape[0], shape[1], shape[2]]))),
        4 => Ok(AnyDimVec::D4(build_4d_vec::<C>([shape[0], shape[1], shape[2], shape[3]]))),
        5 => Ok(AnyDimVec::D5(build_5d_vec::<C>([shape[0], shape[1], shape[2], shape[3], shape[4]]))),
        _ => Err(format!("Unsupported number of dimensions: {}", shape.len())),
    }
}

#[derive(Debug)]
pub enum AnyDimVec<C> {
    D1(Vec<C>),
    D2(Vec<Vec<C>>),
    D3(Vec<Vec<Vec<C>>>),
    D4(Vec<Vec<Vec<Vec<C>>>>),
    D5(Vec<Vec<Vec<Vec<Vec<C>>>>>),
}


fn flatten_recursive(value: &Value, out: &mut Vec<i64>) {
    match value {
        Value::Number(n) => {
            out.push(n.as_i64().expect("Expected i64 number"));
        }
        Value::Array(arr) => {
            for v in arr {
                flatten_recursive(v, out);
            }
        }
        _ => {
            // let mut file = File::create("foo.txt").unwrap();
            // file.write_all(format!("{:#?}",value).as_bytes()).unwrap();
            panic!("Unexpected non-number value in array {:#?}", value)
            
        },
    }
}

pub fn get_5d_circuit_inputs<C: Config>(
    input: &Value,
    input_shape: &[usize],
) -> Vec<Vec<Vec<Vec<Vec<CircuitField<C>>>>>>{
    let mut flat = Vec::new();
    flatten_recursive(input, &mut flat);

    // Pad the shape with 1s to ensure length 5
    let mut shape = input_shape.to_vec();
    while shape.len() < 5 {
        shape.push(1);
    }

    // Create the ndarray from the flat vector and shape
    let array: ArrayD<i64> = ArrayD::from_shape_vec(IxDyn(&shape), flat)
        .expect("Failed to create ArrayD");

    // Build the nested Vec structure
    let nested = build_nd_vec::<C>(&shape).unwrap();

    match nested {
        AnyDimVec::D5(mut v) => {
            for (idx, &val) in array.indexed_iter() {
                let converted = convert_val_to_field_element::<C>(val);

                let i = idx[0];
                let j = idx[1];
                let k = idx[2];
                let l = idx[3];
                let m = idx[4];

                v[i][j][k][l][m] = converted;
            }
            v
        }
        other => panic!("Expected 5D vector, but got {:?}", other),
    }
}

pub fn get_4d_circuit_inputs<C: Config>(
    input: &Value,
    input_shape: &[usize],
) -> Vec<Vec<Vec<Vec<CircuitField<C>>>>>{
    let mut flat = Vec::new();
    flatten_recursive(input, &mut flat);

    // Pad the shape with 1s to ensure length 5
    let mut shape = input_shape.to_vec();
    while shape.len() < 4 {
        shape.push(1);
    }

    // Create the ndarray from the flat vector and shape
    let array: ArrayD<i64> = ArrayD::from_shape_vec(IxDyn(&shape), flat)
        .expect("Failed to create ArrayD");

    // Build the nested Vec structure
    let nested = build_nd_vec::<C>(&shape).unwrap();

    match nested {
        AnyDimVec::D4(mut v) => {
            for (idx, &val) in array.indexed_iter() {
                let converted = convert_val_to_field_element::<C>(val);

                let i = idx[0];
                let j = idx[1];
                let k = idx[2];
                let l = idx[3];

                v[i][j][k][l] = converted;
            }
            v
        }
        other => panic!("Expected 5D vector, but got {:?}", other),
    }
}

pub fn get_3d_circuit_inputs<C: Config>(
    input: &Value,
    input_shape: &[usize],
) -> Vec<Vec<Vec<CircuitField<C>>>>{
    let mut flat = Vec::new();
    flatten_recursive(input, &mut flat);

    // Pad the shape with 1s to ensure length 5
    let mut shape = input_shape.to_vec();
    while shape.len() < 3 {
        shape.push(1);
    }

    // Create the ndarray from the flat vector and shape
    let array: ArrayD<i64> = ArrayD::from_shape_vec(IxDyn(&shape), flat)
        .expect("Failed to create ArrayD");

    // Build the nested Vec structure
    let nested = build_nd_vec::<C>(&shape).unwrap();

    match nested {
        AnyDimVec::D3(mut v) => {
            for (idx, &val) in array.indexed_iter() {
                let converted = convert_val_to_field_element::<C>(val);

                let i = idx[0];
                let j = idx[1];
                let k = idx[2];

                v[i][j][k] = converted;
            }
            v
        }
        other => panic!("Expected 5D vector, but got {:?}", other),
    }
}


pub fn get_2d_circuit_inputs<C: Config>(
    input: &Value,
    input_shape: &[usize],
) -> Vec<Vec<CircuitField<C>>>{
    let mut flat = Vec::new();
    flatten_recursive(input, &mut flat);

    // Pad the shape with 1s to ensure length 5
    let mut shape = input_shape.to_vec();
    while shape.len() < 2 {
        shape.push(1);
    }

    // Create the ndarray from the flat vector and shape
    let array: ArrayD<i64> = ArrayD::from_shape_vec(IxDyn(&shape), flat)
        .expect("Failed to create ArrayD");

    // Build the nested Vec structure
    let nested = build_nd_vec::<C>(&shape).unwrap();

    match nested {
        AnyDimVec::D2(mut v) => {
            for (idx, &val) in array.indexed_iter() {
                let converted = convert_val_to_field_element::<C>(val);

                let i = idx[0];
                let j = idx[1];

                v[i][j] = converted;
            }
            v
        }
        other => panic!("Expected 5D vector, but got",),
    }
}

pub fn get_1d_circuit_inputs<C: Config>(
    input: &Value,
    input_shape: &[usize],
) -> Vec<CircuitField<C>>{
    let mut flat = Vec::new();
    flatten_recursive(input, &mut flat);

    // Pad the shape with 1s to ensure length 5
    let mut shape = input_shape.to_vec();
    while shape.len() < 1 {
        shape.push(1);
    }

    // Create the ndarray from the flat vector and shape
    let array: ArrayD<i64> = ArrayD::from_shape_vec(IxDyn(&shape), flat)
        .expect("Failed to create ArrayD");

    // Build the nested Vec structure
    let nested = build_nd_vec::<C>(&shape).unwrap();

    match nested {
        AnyDimVec::D1(mut v) => {
            for (idx, &val) in array.indexed_iter() {
                let converted = convert_val_to_field_element::<C>(val);

                let i = idx[0];

                v[i] = converted;
            }
            v
        }
            other => panic!("Expected 1D vector"),

    }
}

// TODO change 64 bits to 128 across the board, or add checks. If more than 64 bits, fail
fn convert_val_to_field_element<C: Config>(val: i64) -> <<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField {
    let converted = if val < 0 {
        CircuitField::<C>::from_u256(U256::from(val.abs() as u64)).neg()
    } else {
        CircuitField::<C>::from_u256(U256::from(val.abs() as u64))
    };
    converted
}

// ─────────────────────────────────────────────────────────────────────────────
// TRAIT: IntoTensor
// ─────────────────────────────────────────────────────────────────────────────

/// Recursively applies a function to all [`Variable`] elements within a nested tensor-like structure.
///
/// This trait enables uniform, elementwise transformations over arbitrarily nested [`Vec`] containers
/// containing [`Variable`]s. It is implemented for:
/// - `Variable` (scalar),
/// - `Vec<Variable>` (1D),
/// - `Vec<Vec<Variable>>` (2D),
/// - and recursively for higher-dimensional tensors.
///
/// This is useful for applying functions like negation, rescaling, or ReLU activation to
/// entire arrays or tensors without manually writing nested loops.
///
/// # Associated Type
/// - [`Output`]: The resulting structure after applying the function to each [`Variable`].
///
/// # Provided Method
/// - `map_elements(f)`: Applies the closure `f` recursively to each [`Variable`] in the structure.
///
/// # Example
/// ```ignore
/// let scalar: Variable = ...;
/// let negated_scalar = scalar.map_elements(|v| api.neg(v));
///
/// let matrix: Vec<Vec<Variable>> = ...;
/// let negated_matrix = matrix.map_elements(|v| api.neg(v));
/// ```
pub trait IntoTensor {
    type Output;

    fn map_elements<F>(self, f: &mut F) -> Self::Output
    where
        F: FnMut(Variable) -> Variable;
}

impl IntoTensor for Variable {
    type Output = Variable;

    fn map_elements<F>(self, f: &mut F) -> Self::Output
    where
        F: FnMut(Variable) -> Variable,
    {
        f(self)
    }
}

impl<T: IntoTensor> IntoTensor for Vec<T> {
    type Output = Vec<T::Output>;

    fn map_elements<F>(self, f: &mut F) -> Self::Output
    where
        F: FnMut(Variable) -> Variable,
    {
        self.into_iter().map(|x| x.map_elements(f)).collect()
    }
}