/// Standard library imports
use std::ops::Neg;

/// External crate imports
use ethnum::U256;
use ndarray::{Array2, ArrayD, Ix1, Ix2, Ix3, Ix4, Ix5, IxDyn, Zip};
use serde_json::Value;

/// ExpanderCompilerCollection and proving framework imports
use expander_compiler::frontend::*;
use gkr_engine::{FieldEngine, GKREngine};

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

/// Convert an ArrayD of i64 values to circuit constants (Variables)
pub fn load_array_constants<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input: &ArrayD<i64>,
) -> ArrayD<Variable> {
    let mut result = ArrayD::default(input.dim());
    Zip::from(&mut result)
        .and(input)
        .for_each(|out_elem, &val| {
            *out_elem = load_circuit_constant(api, val);
        });
    result
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