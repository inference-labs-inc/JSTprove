/// Standard library imports
use std::ops::Neg;

/// External crate imports
use ethnum::U256;
use ndarray::{ArrayD, IxDyn, Zip};
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
        }
    }
}

pub fn get_nd_circuit_inputs<C: Config>(
    input: &Value,
    input_shape: &[usize],
) -> ArrayD<CircuitField<C>> {
    // 1) flatten JSON → Vec<i64>
    let mut flat = Vec::new();
    flatten_recursive(input, &mut flat);

    // 2) pad to at least 1 dimension
    let mut shape = input_shape.to_vec();
    if shape.is_empty() {
        shape.push(1);
    }

    // 3) sanity check
    let expected: usize = shape.iter().product();
    assert_eq!(
        flat.len(),
        expected,
        "JSON had {} elements but shape {:?} requires {}",
        flat.len(),
        shape,
        expected
    );

    // 4) build the i64 ndarray
    let a_i64: ArrayD<i64> =
        ArrayD::from_shape_vec(IxDyn(&shape), flat).expect("shape/flat-len mismatch");

    // 5) map to your circuit field type
    a_i64.mapv(|val| convert_val_to_field_element::<C>(val))
}

// TODO change 64 bits to 128 across the board, or add checks. If more than 64 bits, fail
fn convert_val_to_field_element<C: Config>(
    val: i64,
) -> <<C as GKREngine>::FieldConfig as FieldEngine>::CircuitField {
    let converted = if val < 0 {
        CircuitField::<C>::from_u256(U256::from(val.abs() as u64)).neg()
    } else {
        CircuitField::<C>::from_u256(U256::from(val.abs() as u64))
    };
    converted
}
