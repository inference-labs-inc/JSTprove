use ark_std::{end_timer, start_timer};
use rand::RngCore;

use crate::Field;

#[allow(clippy::eq_op)]
pub(crate) fn commutativity_tests<F: Field, R: RngCore>(mut rng: R, type_name: String) {
    let _message = format!("commutativity {type_name}");
    let start = start_timer!(|| _message);
    for _ in 0..1000 {
        let a = F::random_unsafe(&mut rng);
        let b = F::random_unsafe(&mut rng);

        assert_eq!(a * b, b * a);
        assert_eq!(a + b, b + a);
    }
    end_timer!(start);
}

pub(crate) fn identity_tests<F: Field, R: RngCore>(mut rng: R, type_name: String) {
    let _message = format!("identity {type_name}");
    let start = start_timer!(|| _message);
    for _ in 0..1000 {
        let a = F::random_unsafe(&mut rng);

        let mut t = a;
        t.add_assign(&F::zero());
        assert_eq!(t, a);

        let mut t = a;
        t.mul_assign(&F::one());
        assert_eq!(t, a);
    }
    end_timer!(start);
}

pub(crate) fn random_multiplication_tests<F: Field, R: RngCore>(mut rng: R, type_name: String) {
    let _message = format!("multiplication {type_name}");
    let start = start_timer!(|| _message);
    for _ in 0..1000 {
        let a = F::random_unsafe(&mut rng);
        let b = F::random_unsafe(&mut rng);
        let c = F::random_unsafe(&mut rng);

        let mut t0 = a; // (a * b) * c
        t0.mul_assign(&b);
        t0.mul_assign(&c);

        let mut t1 = a; // (a * c) * b
        t1.mul_assign(&c);
        t1.mul_assign(&b);

        let mut t2 = b; // (b * c) * a
        t2.mul_assign(&c);
        t2.mul_assign(&a);

        assert_eq!(t0, t1);
        assert_eq!(t1, t2);
        assert_eq!(a.square(), a * a);
    }
    end_timer!(start);
}

pub(crate) fn random_addition_tests<F: Field, R: RngCore>(mut rng: R, type_name: String) {
    let _message = format!("addition {type_name}");
    let start = start_timer!(|| _message);
    for _ in 0..1000 {
        let a = F::random_unsafe(&mut rng);
        let b = F::random_unsafe(&mut rng);
        let c = F::random_unsafe(&mut rng);

        let mut t0 = a; // (a + b) + c
        t0.add_assign(&b);
        t0.add_assign(&c);

        let mut t1 = a; // (a + c) + b
        t1.add_assign(&c);
        t1.add_assign(&b);

        let mut t2 = b; // (b + c) + a
        t2.add_assign(&c);
        t2.add_assign(&a);

        assert_eq!(t0, t1);
        assert_eq!(t1, t2);
    }
    end_timer!(start);
}

pub(crate) fn random_subtraction_tests<F: Field, R: RngCore>(mut rng: R, type_name: String) {
    let _message = format!("subtraction {type_name}");
    let start = start_timer!(|| _message);
    for _ in 0..1000 {
        let a = F::random_unsafe(&mut rng);
        let b = F::random_unsafe(&mut rng);

        let mut t0 = a; // (a - b)
        t0.sub_assign(&b);

        let mut t1 = b; // (b - a)
        t1.sub_assign(&a);

        let mut t2 = t0; // (a - b) + (b - a) = 0
        t2.add_assign(&t1);

        assert!(t2.is_zero());
    }
    end_timer!(start);
}

pub(crate) fn random_negation_tests<F: Field, R: RngCore>(mut rng: R, type_name: String) {
    let _message = format!("negation {type_name}");
    let start = start_timer!(|| _message);
    for _ in 0..1000 {
        let a = F::random_unsafe(&mut rng);
        let mut b = a;
        b = b.neg();
        b.add_assign(&a);
        assert!(b.is_zero());
    }
    end_timer!(start);
}

pub(crate) fn random_doubling_tests<F: Field, R: RngCore>(mut rng: R, type_name: String) {
    let _message = format!("doubling {type_name}");
    let start = start_timer!(|| _message);
    for _ in 0..1000 {
        let mut a = F::random_unsafe(&mut rng);
        let mut b = a;
        a.add_assign(&b);
        b = b.double();

        assert_eq!(a, b);
    }
    end_timer!(start);
}

pub(crate) fn random_squaring_tests<F: Field, R: RngCore>(mut rng: R, type_name: String) {
    let _message = format!("squaring {type_name}");
    let start = start_timer!(|| _message);
    for _ in 0..1000 {
        let mut a = F::random_unsafe(&mut rng);
        let mut b = a;
        a.mul_assign(&b);
        b = b.square();

        assert_eq!(a, b);
    }
    end_timer!(start);
}

pub(crate) fn random_expansion_tests<F: Field, R: RngCore>(mut rng: R, type_name: String) {
    let _message = format!("expansion {type_name}");
    let start = start_timer!(|| _message);
    for _ in 0..1000 {
        // Compare (a + b)(c + d) and (a*c + b*c + a*d + b*d)

        let a = F::random_unsafe(&mut rng);
        let b = F::random_unsafe(&mut rng);
        let c = F::random_unsafe(&mut rng);
        let d = F::random_unsafe(&mut rng);

        let mut t0 = a;
        t0.add_assign(&b);
        let mut t1 = c;
        t1.add_assign(&d);
        t0.mul_assign(&t1);

        let mut t2 = a;
        t2.mul_assign(&c);
        let mut t3 = b;
        t3.mul_assign(&c);
        let mut t4 = a;
        t4.mul_assign(&d);
        let mut t5 = b;
        t5.mul_assign(&d);

        t2.add_assign(&t3);
        t2.add_assign(&t4);
        t2.add_assign(&t5);

        assert_eq!(t0, t2);
    }
    end_timer!(start);
}

pub(crate) fn random_serde_tests<F: Field, R: RngCore>(mut rng: R, type_name: String) {
    let _message = format!("serde {type_name}");
    let start = start_timer!(|| _message);
    for _ in 0..1000 {
        let a = F::random_unsafe(&mut rng);
        let mut buffer = vec![];
        assert!(a.serialize_into(&mut buffer).is_ok());

        let b = F::deserialize_from(buffer.as_slice());
        assert!(b.is_ok());
        let b = b.unwrap();
        assert_eq!(a, b);
    }
    end_timer!(start);
}

pub(crate) fn associativity_tests<F: Field, R: RngCore>(mut rng: R, type_name: String) {
    let _message = format!("associativity {type_name}");
    let start = start_timer!(|| _message);
    for _ in 0..1000 {
        let a = F::random_unsafe(&mut rng);
        let b = F::random_unsafe(&mut rng);
        let c = F::random_unsafe(&mut rng);

        let t0 = (a + b) + c; // (a + b) + c
        let t1 = a + (b + c); // a + (b + c)

        assert_eq!(t0, t1);

        let t0 = a * (b * c);
        let t1 = (a * b) * c;

        assert_eq!(t0, t1);
    }
    end_timer!(start);
}
