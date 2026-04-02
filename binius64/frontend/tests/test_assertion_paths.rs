// Copyright 2025 Irreducible Inc.
use binius_core::word::Word;
use binius_frontend::CircuitBuilder;

#[test]
fn test_assertion_failure_shows_path() {
    // Create a circuit with nested subcircuits to test path reporting
    let builder = CircuitBuilder::new();

    // Create some nested paths
    let module_a = builder.subcircuit("module_a");
    let submodule = module_a.subcircuit("submodule");

    // Add some wires
    let x = submodule.add_witness();
    let y = submodule.add_witness();

    // Add an assertion that will fail
    submodule.assert_eq("test_assertion", x, y);

    // Build the circuit
    let circuit = builder.build();

    // Create a witness filler
    let mut filler = circuit.new_witness_filler();

    // Set different values to trigger assertion failure
    filler[x] = Word::from_u64(42);
    filler[y] = Word::from_u64(100);

    // Try to populate - this should fail with a nice error message showing the path
    match circuit.populate_wire_witness(&mut filler) {
        Ok(_) => panic!("Circuit should have failed assertion"),
        Err(e) => {
            let error_string = format!("{}", e);
            println!("Error message:\n{}", error_string);

            // Check that the error contains the path
            assert!(
                error_string.contains("module_a.submodule.test_assertion"),
                "Error should contain the path 'module_a.submodule.test_assertion', but got: {}",
                error_string
            );

            // Check that it contains the condition (values are shown in hex)
            assert!(
                error_string.contains("0x000000000000002a")
                    && error_string.contains("0x0000000000000064"),
                "Error should contain the values that failed (42=0x2a, 100=0x64)"
            );
        }
    }
}

#[test]
fn test_multiple_assertion_failures() {
    let builder = CircuitBuilder::new();

    let module = builder.subcircuit("validator");

    // Add multiple assertions that will fail
    let a = module.add_witness();
    let b = module.add_witness();
    let c = module.add_witness();

    module.assert_eq("check_a_equals_b", a, b);
    module.assert_eq("check_b_equals_c", b, c);

    let circuit = builder.build();
    let mut filler = circuit.new_witness_filler();

    // Set all different values
    filler[a] = Word::from_u64(1);
    filler[b] = Word::from_u64(2);
    filler[c] = Word::from_u64(3);

    match circuit.populate_wire_witness(&mut filler) {
        Ok(_) => panic!("Circuit should have failed assertions"),
        Err(e) => {
            let error_string = format!("{}", e);
            println!("Multiple failures:\n{}", error_string);

            // Both assertions should be reported with their paths
            assert!(
                error_string.contains("validator.check_a_equals_b"),
                "Should contain first assertion path"
            );
            assert!(
                error_string.contains("validator.check_b_equals_c"),
                "Should contain second assertion path"
            );
        }
    }
}
