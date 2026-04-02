// Copyright 2026 The Binius Developers

//! Symbolic field element types for building constraint systems.

use std::{
    cell::RefCell,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    rc::{Rc, Weak},
};

use binius_field::{
    BinaryField128bGhash as B128, ExtensionField, Field,
    arithmetic_traits::{InvertOrZero, Square},
    field::FieldOps,
};
use binius_spartan_frontend::{
    circuit_builder::{CircuitBuilder, ConstraintBuilder},
    constraint_system::ConstraintWire,
};

/// An opaque wire in the constraint builder, carrying a weak reference to the builder.
///
/// The weak reference allows the channel to reclaim sole ownership of the builder in `finish()`
/// via `Rc::try_unwrap`, even while `BuildElem` values are still alive.
#[derive(Clone)]
pub struct BuildWire {
    builder: Weak<RefCell<ConstraintBuilder>>,
    wire: ConstraintWire,
}

impl BuildWire {
    pub(crate) fn new(builder: &Rc<RefCell<ConstraintBuilder>>, wire: ConstraintWire) -> Self {
        Self {
            builder: Rc::downgrade(builder),
            wire,
        }
    }

    pub(crate) fn wire(&self) -> ConstraintWire {
        self.wire
    }

    fn upgrade(&self) -> Rc<RefCell<ConstraintBuilder>> {
        self.builder
            .upgrade()
            .expect("channel has been consumed by finish()")
    }
}

/// A symbolic field element that is either a known constant or a wire in a constraint system.
#[derive(Clone)]
pub enum BuildElem {
    Constant(B128),
    Wire(BuildWire),
}

impl BuildElem {
    /// Returns the builder Rc if this is a Wire variant.
    fn builder_rc(&self) -> Option<Rc<RefCell<ConstraintBuilder>>> {
        match self {
            BuildElem::Constant(_) => None,
            BuildElem::Wire(w) => Some(w.upgrade()),
        }
    }

    /// Given two BuildElems, return the builder that at least one of them references.
    ///
    /// Panics if both are constants (no builder to resolve).
    fn resolve_builder(a: &BuildElem, b: &BuildElem) -> Rc<RefCell<ConstraintBuilder>> {
        match (a.builder_rc(), b.builder_rc()) {
            (Some(a), Some(b)) => {
                assert!(
                    Rc::ptr_eq(&a, &b),
                    "BuildElem wires reference different ConstraintBuilders"
                );
                b
            }
            (Some(b), None) | (None, Some(b)) => b,
            (None, None) => panic!("cannot resolve builder: both operands are constants"),
        }
    }

    /// Convert this element to a ConstraintWire, allocating a constant wire if necessary.
    fn to_wire(&self, builder: &mut ConstraintBuilder) -> ConstraintWire {
        match self {
            BuildElem::Constant(val) => builder.constant(*val),
            BuildElem::Wire(w) => w.wire,
        }
    }

    fn make_wire(rc: &Rc<RefCell<ConstraintBuilder>>, wire: ConstraintWire) -> Self {
        BuildElem::Wire(BuildWire {
            builder: Rc::downgrade(rc),
            wire,
        })
    }
}

// In characteristic 2, negation is identity.
impl Neg for BuildElem {
    type Output = Self;

    fn neg(self) -> Self {
        self
    }
}

impl Add for BuildElem {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        match (&self, &rhs) {
            (BuildElem::Constant(a), BuildElem::Constant(b)) => BuildElem::Constant(*a + *b),
            _ => {
                if matches!(&self, BuildElem::Constant(c) if *c == B128::ZERO) {
                    return rhs;
                }
                if matches!(&rhs, BuildElem::Constant(c) if *c == B128::ZERO) {
                    return self;
                }
                let rc = BuildElem::resolve_builder(&self, &rhs);
                let mut builder = rc.borrow_mut();
                let a_wire = self.to_wire(&mut builder);
                let b_wire = rhs.to_wire(&mut builder);
                let out = builder.add(a_wire, b_wire);
                BuildElem::make_wire(&rc, out)
            }
        }
    }
}

impl Sub for BuildElem {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        match (&self, &rhs) {
            (BuildElem::Constant(a), BuildElem::Constant(b)) => BuildElem::Constant(*a + *b),
            _ => {
                if matches!(&self, BuildElem::Constant(c) if *c == B128::ZERO) {
                    return rhs;
                }
                if matches!(&rhs, BuildElem::Constant(c) if *c == B128::ZERO) {
                    return self;
                }
                let rc = BuildElem::resolve_builder(&self, &rhs);
                let mut builder = rc.borrow_mut();
                let a_wire = self.to_wire(&mut builder);
                let b_wire = rhs.to_wire(&mut builder);
                let out = builder.sub(a_wire, b_wire);
                BuildElem::make_wire(&rc, out)
            }
        }
    }
}

impl Mul for BuildElem {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        match (&self, &rhs) {
            (BuildElem::Constant(a), BuildElem::Constant(b)) => BuildElem::Constant(*a * *b),
            _ => {
                if matches!(&self, BuildElem::Constant(c) if *c == B128::ZERO) {
                    return BuildElem::Constant(B128::ZERO);
                }
                if matches!(&rhs, BuildElem::Constant(c) if *c == B128::ZERO) {
                    return BuildElem::Constant(B128::ZERO);
                }
                if matches!(&self, BuildElem::Constant(c) if *c == B128::ONE) {
                    return rhs;
                }
                if matches!(&rhs, BuildElem::Constant(c) if *c == B128::ONE) {
                    return self;
                }
                let rc = BuildElem::resolve_builder(&self, &rhs);
                let mut builder = rc.borrow_mut();
                let a_wire = self.to_wire(&mut builder);
                let b_wire = rhs.to_wire(&mut builder);
                let out = builder.mul(a_wire, b_wire);
                BuildElem::make_wire(&rc, out)
            }
        }
    }
}

// By-reference variants: clone and delegate.

impl Add<&BuildElem> for BuildElem {
    type Output = Self;

    fn add(self, rhs: &BuildElem) -> Self {
        self + rhs.clone()
    }
}

impl Sub<&BuildElem> for BuildElem {
    type Output = Self;

    fn sub(self, rhs: &BuildElem) -> Self {
        self - rhs.clone()
    }
}

impl Mul<&BuildElem> for BuildElem {
    type Output = Self;

    fn mul(self, rhs: &BuildElem) -> Self {
        self * rhs.clone()
    }
}

// Assign variants

impl AddAssign for BuildElem {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl SubAssign for BuildElem {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl MulAssign for BuildElem {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl AddAssign<&BuildElem> for BuildElem {
    fn add_assign(&mut self, rhs: &BuildElem) {
        *self = self.clone() + rhs.clone();
    }
}

impl SubAssign<&BuildElem> for BuildElem {
    fn sub_assign(&mut self, rhs: &BuildElem) {
        *self = self.clone() - rhs.clone();
    }
}

impl MulAssign<&BuildElem> for BuildElem {
    fn mul_assign(&mut self, rhs: &BuildElem) {
        *self = self.clone() * rhs.clone();
    }
}

// Sum and Product

impl Sum for BuildElem {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(BuildElem::Constant(B128::ZERO), |acc, x| acc + x)
    }
}

impl<'a> Sum<&'a BuildElem> for BuildElem {
    fn sum<I: Iterator<Item = &'a BuildElem>>(iter: I) -> Self {
        iter.cloned().sum()
    }
}

impl Product for BuildElem {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(BuildElem::Constant(B128::ONE), |acc, x| acc * x)
    }
}

impl<'a> Product<&'a BuildElem> for BuildElem {
    fn product<I: Iterator<Item = &'a BuildElem>>(iter: I) -> Self {
        iter.cloned().product()
    }
}

impl Square for BuildElem {
    fn square(self) -> Self {
        match &self {
            BuildElem::Constant(c) => BuildElem::Constant(c.square()),
            _ => self.clone() * self,
        }
    }
}

impl InvertOrZero for BuildElem {
    fn invert_or_zero(self) -> Self {
        match &self {
            BuildElem::Constant(c) => BuildElem::Constant(c.invert_or_zero()),
            BuildElem::Wire(w) => {
                let rc = w.upgrade();
                let mut builder = rc.borrow_mut();
                let wire = w.wire;

                // Allocate the inverse wire via hint.
                let [inv_wire] = builder.hint([wire], |[x]| [x.invert_or_zero()]);

                // Constrain wire * inverse = one.
                let product = builder.mul(wire, inv_wire);
                let one = builder.constant(B128::ONE);
                builder.assert_eq(product, one);

                BuildElem::make_wire(&rc, inv_wire)
            }
        }
    }
}

impl FieldOps for BuildElem {
    type Scalar = B128;

    fn zero() -> Self {
        BuildElem::Constant(B128::ZERO)
    }

    fn one() -> Self {
        BuildElem::Constant(B128::ONE)
    }

    fn square_transpose<FSub: Field>(_elems: &mut [Self])
    where
        Self::Scalar: ExtensionField<FSub>,
    {
        unimplemented!("square_transpose is not used in symbolic verifier context")
    }
}
