/// Standard library imports
use std::ops::Neg;

/// External crate imports
use ethnum::U256;
use ndarray::{ArrayD, IxDyn, Zip};
use serde_json::Value;

/// ExpanderCompilerCollection and proving framework imports
use expander_compiler::frontend::*;
use gkr_engine::{FieldEngine, GKREngine};

use crate::circuit_functions::utils::{ArrayConversionError, UtilsError};

/// Load in circuit constant given i64, negative values are represented by p-x and positive values are x
pub fn load_circuit_constant<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: i64,
) -> Variable {
    if x < 0 {
        let y = api.constant(CircuitField::<C>::from_u256(U256::from(x.abs() as u128)));
        api.neg(y)
    } else {
        api.constant(CircuitField::<C>::from_u256(U256::from(x.abs() as u128)))
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
// fn flatten_recursive(value: &Value, out: &mut Vec<i64>) -> Result<(), CircuitInputError> {
//     match value {
//         Value::Number(n) => {
//             if let Some(i) = n.as_i64() {
//                 out.push(i);
//                 Ok(())
//             } else {
//                 Err(CircuitInputError::InvalidNumber { value: value.clone() })
//             }
//         }
//         Value::Array(arr) => {
//             for v in arr {
//                 flatten_recursive(v, out)?; // propagate errors
//             }
//             Ok(())
//         }
//         _ => Err(CircuitInputError::UnexpectedValue { value: value.clone() }),
//     }
// }

fn flatten_recursive(value: &Value, out: &mut Vec<i64>) -> Result<(), UtilsError> {
    match value {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push(i);
                Ok(())
            } else {
                Err(UtilsError::InvalidNumber { value: value.clone() })
            }
        }
        Value::Array(arr) => {
            for v in arr {
                flatten_recursive(v, out)?;
            }
            Ok(())
        }
        _ => Err(UtilsError::InvalidNumber { value: value.clone() }),
            
    }
}
// pub fn get_nd_circuit_inputs<C: Config>(
//     input: &Value,
//     input_shape: &[usize],
// ) -> Result<ArrayD<CircuitField<C>>, CircuitInputError> {
//     // 1) flatten JSON → Vec<i64>
//     let mut flat = Vec::new();
//     flatten_recursive(input, &mut flat)?;

//     // 2) pad to at least 1 dimension
//     let mut shape = input_shape.to_vec();
//     if shape.is_empty() {
//         shape.push(1);
//     }

//     // 3) sanity check
//     let expected: usize = shape.iter().product();
//     if flat.len() != expected {
//         return Err(CircuitInputError::ShapeMismatch {
//             got: flat.len(),
//             expected,
//             shape: shape.clone(),
//         });
//     }

//     // 4) build the i64 ndarray (ShapeError -> CircuitInputError automatically via `From`)
//     let a_i64: ArrayD<i64> = ArrayD::from_shape_vec(IxDyn(&shape), flat)?;

//     // 5) map to your circuit field type
//     Ok(a_i64.mapv(|val| convert_val_to_field_element::<C>(val)))
// }

pub fn get_nd_circuit_inputs<C: Config>(
    input: &Value,
    input_shape: &[usize],
) -> Result<ArrayD<CircuitField<C>>, UtilsError> {
    // 1) flatten JSON → Vec<i64>
    let mut flat = Vec::new();
    flatten_recursive(input, &mut flat)?;

    // 2) pad to at least 1 dimension
    let mut shape = input_shape.to_vec();
    if shape.is_empty() {
        shape.push(1);
    }

    // 3) build the i64 ndarray
    let a_i64: ArrayD<i64> =
        ArrayD::from_shape_vec(IxDyn(&shape), flat).map_err(|err| ArrayConversionError::ShapeError(err))?;

    // 4) map to your circuit field type
    Ok(a_i64.mapv(|val| convert_val_to_field_element::<C>(val)))
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
