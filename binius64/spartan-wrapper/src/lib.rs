// Copyright 2026 The Binius Developers

//! Spartan wrapper for symbolically executing IOP verifiers to build constraint systems.
//!
//! This crate provides [`IronSpartanBuilderChannel`], an implementation of [`IPVerifierChannel`]
//! that symbolically executes a verifier and records the computation as an IronSpartan constraint
//! system via [`ConstraintBuilder`].
//!
//! [`IPVerifierChannel`]: binius_ip::channel::IPVerifierChannel
//! [`ConstraintBuilder`]: binius_spartan_frontend::circuit_builder::ConstraintBuilder

mod build_elem;
mod channel;

pub use build_elem::{BuildElem, BuildWire};
pub use channel::IronSpartanBuilderChannel;

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use binius_field::{BinaryField128bGhash as B128, Field, arithmetic_traits::InvertOrZero};
    use binius_ip::channel::IPVerifierChannel;
    use binius_spartan_frontend::circuit_builder::ConstraintBuilder;

    use super::*;

    /// Helper to create a BuildWire from a ConstraintBuilder Rc for tests.
    fn alloc_inout_wire(rc: &Rc<std::cell::RefCell<ConstraintBuilder>>) -> BuildElem {
        let wire = rc.borrow_mut().alloc_inout();
        BuildElem::Wire(BuildWire::new(rc, wire))
    }

    #[test]
    fn test_constant_arithmetic() {
        let a = BuildElem::Constant(B128::new(3));
        let b = BuildElem::Constant(B128::new(5));

        // Addition in char 2: 3 + 5 = 3 ^ 5 = 6
        let sum = a.clone() + b.clone();
        assert!(matches!(sum, BuildElem::Constant(c) if c == B128::new(3) + B128::new(5)));

        // Multiplication
        let product = a.clone() * b.clone();
        assert!(matches!(product, BuildElem::Constant(c) if c == B128::new(3) * B128::new(5)));

        // Subtraction equals addition in char 2
        let diff = a - b;
        assert!(matches!(diff, BuildElem::Constant(c) if c == B128::new(3) + B128::new(5)));
    }

    #[test]
    fn test_constant_identity_shortcuts() {
        let rc = Rc::new(std::cell::RefCell::new(ConstraintBuilder::new()));
        let elem = alloc_inout_wire(&rc);

        // Adding zero returns the wire unchanged.
        let result = elem.clone() + BuildElem::Constant(B128::ZERO);
        assert!(matches!(result, BuildElem::Wire(_)));

        // Multiplying by one returns the wire unchanged.
        let result = elem.clone() * BuildElem::Constant(B128::ONE);
        assert!(matches!(result, BuildElem::Wire(_)));

        // Multiplying by zero returns constant zero.
        let result = elem * BuildElem::Constant(B128::ZERO);
        assert!(matches!(result, BuildElem::Constant(c) if c == B128::ZERO));
    }

    #[test]
    fn test_wire_addition_creates_constraint() {
        let rc = Rc::new(std::cell::RefCell::new(ConstraintBuilder::new()));
        let a = alloc_inout_wire(&rc);
        let b = alloc_inout_wire(&rc);

        let _sum = a + b;
        // The addition should produce constraints. After finalization, the zero constraint
        // from addition becomes a mul constraint (multiplied by one).
        let (cs, _layout) = Rc::try_unwrap(rc).unwrap().into_inner().build().finalize();
        assert!(!cs.mul_constraints().is_empty());
    }

    #[test]
    fn test_wire_multiplication_creates_constraint() {
        let rc = Rc::new(std::cell::RefCell::new(ConstraintBuilder::new()));
        let a = alloc_inout_wire(&rc);
        let b = alloc_inout_wire(&rc);

        let _product = a * b;
        let (cs, _layout) = Rc::try_unwrap(rc).unwrap().into_inner().build().finalize();
        // At least 1 mul constraint from a*b.
        assert!(!cs.mul_constraints().is_empty());
    }

    #[test]
    fn test_invert_or_zero_creates_constraints() {
        let rc = Rc::new(std::cell::RefCell::new(ConstraintBuilder::new()));
        let elem = alloc_inout_wire(&rc);

        let _inv = elem.invert_or_zero();
        let (cs, _layout) = Rc::try_unwrap(rc).unwrap().into_inner().build().finalize();
        // InvertOrZero creates: a mul constraint (wire * inv) and a zero constraint
        // (product ^ one), both of which become mul constraints after finalization.
        assert!(cs.mul_constraints().len() >= 2);
    }

    #[test]
    fn test_channel_recv_and_sample() {
        let mut channel = IronSpartanBuilderChannel::new(ConstraintBuilder::new());

        let a = channel.recv_one().unwrap();
        let b = channel.sample();
        let c = channel.recv_array::<3>().unwrap();

        // All should be Wire variants.
        assert!(matches!(a, BuildElem::Wire(_)));
        assert!(matches!(b, BuildElem::Wire(_)));
        for elem in &c {
            assert!(matches!(elem, BuildElem::Wire(_)));
        }
    }

    #[test]
    fn test_channel_assert_zero() {
        let mut channel = IronSpartanBuilderChannel::new(ConstraintBuilder::new());

        // Assert zero on a constant zero should succeed.
        assert!(channel.assert_zero(BuildElem::Constant(B128::ZERO)).is_ok());

        // Assert zero on a nonzero constant should fail.
        assert!(channel.assert_zero(BuildElem::Constant(B128::ONE)).is_err());

        // Assert zero on a wire should succeed (records constraint).
        let wire_elem = channel.recv_one().unwrap();
        assert!(channel.assert_zero(wire_elem).is_ok());
    }

    #[test]
    fn test_verify_iop_builds_constraint_system() {
        use binius_hash::StdCompression;
        use binius_spartan_frontend::{
            circuit_builder::CircuitBuilder, circuits::powers, compiler::compile,
        };
        use binius_spartan_verifier::Verifier;

        // Build a power7 circuit: assert that x^7 = y.
        fn power7_circuit<Builder: CircuitBuilder>(
            builder: &mut Builder,
            x_wire: Builder::Wire,
            y_wire: Builder::Wire,
        ) {
            let powers_vec = powers(builder, x_wire, 7);
            let x7 = powers_vec[6];
            builder.assert_eq(x7, y_wire);
        }

        let mut constraint_builder = ConstraintBuilder::new();
        let x_wire = constraint_builder.alloc_inout();
        let y_wire = constraint_builder.alloc_inout();
        power7_circuit(&mut constraint_builder, x_wire, y_wire);
        let (cs, _layout) = compile(constraint_builder);

        // Set up a real Verifier with the compiled constraint system.
        let log_inv_rate = 1;
        let compression = StdCompression::default();
        let verifier =
            Verifier::<_, binius_hash::StdDigest, _>::setup(cs, log_inv_rate, compression)
                .expect("verifier setup failed");

        let cs = verifier.constraint_system();
        let public_size = 1 << cs.log_public();

        // Create the builder channel and run verify_iop symbolically.
        let mut channel = IronSpartanBuilderChannel::new(ConstraintBuilder::new());

        // Use zero-filled public inputs of the correct length.
        let public = vec![B128::ZERO; public_size];
        verifier
            .verify_iop(&public, &mut channel)
            .expect("symbolic verify_iop failed");

        let builder = channel.finish();

        // Extract the constraint system built by the symbolic execution.
        let (wrapper_cs, _layout) = builder.build().finalize();

        // The symbolic execution should have produced a nontrivial constraint system.
        assert!(wrapper_cs.n_inout() > 0);
        assert!(!wrapper_cs.mul_constraints().is_empty());
    }

    #[test]
    fn test_channel_integration_simple_circuit() {
        // Build a simple circuit: recv two values, multiply them, assert_zero on the
        // difference with a third received value (ie. a * b == c).
        let mut channel = IronSpartanBuilderChannel::new(ConstraintBuilder::new());
        let a = channel.recv_one().unwrap();
        let b = channel.recv_one().unwrap();
        let c = channel.recv_one().unwrap();
        let product = a * b;
        // In char 2, product - c == product + c.
        let diff = product + c;
        channel.assert_zero(diff).unwrap();

        // Finish and extract the constraint system.
        let builder = channel.finish();
        let (cs, _layout) = builder.build().finalize();

        // Should have inout wires from recv + private wires from mul and add.
        assert!(cs.n_inout() >= 3);
        // Should have at least 1 mul constraint from a*b and 1 from the finalized zero
        // constraint.
        assert!(!cs.mul_constraints().is_empty());
    }
}
