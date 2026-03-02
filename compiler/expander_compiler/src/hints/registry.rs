use std::collections::HashMap;

use tiny_keccak::Hasher;

use crate::{field::Field, utils::error::Error};

use super::{stub_impl, BuiltinHintIds};

type HintFn<F> = fn(&[F], &mut [F]) -> Result<(), Error>;

#[derive(Default, Clone)]
pub struct HintRegistry<F: Field> {
    hints: HashMap<usize, HintFn<F>>,
}

pub fn hint_key_to_id(key: &str) -> usize {
    let mut hasher = tiny_keccak::Keccak::v256();
    hasher.update(key.as_bytes());
    let mut hash = [0u8; 32];
    hasher.finalize(&mut hash);

    let res = usize::from_le_bytes(hash[0..8].try_into().unwrap());
    if BuiltinHintIds::from_usize(res).is_some() {
        panic!("Hint id {res} collides with a builtin hint id");
    }
    res
}

impl<F: Field> HintRegistry<F> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn register(&mut self, key: &str, hint: fn(&[F], &mut [F]) -> Result<(), Error>) {
        let id = hint_key_to_id(key);
        if self.hints.contains_key(&id) {
            panic!("Hint with id {id} already exists");
        }
        self.hints.insert(id, hint);
    }
    pub fn call(&self, id: usize, args: &[F], num_outputs: usize) -> Result<Vec<F>, Error> {
        if let Some(hint) = self.hints.get(&id) {
            let mut outputs = vec![F::zero(); num_outputs];
            hint(args, &mut outputs).map(|_| outputs)
        } else {
            panic!("Hint with id {id} not found");
        }
    }
}

#[derive(Default)]
pub struct EmptyHintCaller;

impl EmptyHintCaller {
    pub fn new() -> Self {
        Self
    }
}
pub struct StubHintCaller;

pub trait HintCaller<F: Field>: Sync + 'static {
    fn call(&self, id: usize, args: &[F], num_outputs: usize) -> Result<Vec<F>, Error>;
}

impl<F: Field + 'static> HintCaller<F> for HintRegistry<F> {
    fn call(&self, id: usize, args: &[F], num_outputs: usize) -> Result<Vec<F>, Error> {
        self.call(id, args, num_outputs)
    }
}

impl<F: Field> HintCaller<F> for EmptyHintCaller {
    fn call(&self, id: usize, _: &[F], _: usize) -> Result<Vec<F>, Error> {
        Err(Error::UserError(format!("hint with id {id} not found")))
    }
}

impl<F: Field> HintCaller<F> for StubHintCaller {
    fn call(&self, id: usize, args: &[F], num_outputs: usize) -> Result<Vec<F>, Error> {
        Ok(stub_impl(id, &args.to_vec(), num_outputs))
    }
}
