use std::fmt::Display;

use tiny_keccak::Hasher;

use super::builder::{ensure_variable_valid, Variable};

pub trait JoinVecVariables {
    fn join_vec_variables(&self, res: &mut Vec<Variable>, structure: &mut Vec<usize>);
}

impl JoinVecVariables for Variable {
    fn join_vec_variables(&self, res: &mut Vec<Variable>, _structure: &mut Vec<usize>) {
        ensure_variable_valid(*self);
        res.push(*self);
    }
}

impl<T: JoinVecVariables> JoinVecVariables for Vec<T> {
    fn join_vec_variables(&self, res: &mut Vec<Variable>, structure: &mut Vec<usize>) {
        structure.push(self.len());
        for item in self {
            item.join_vec_variables(res, structure);
        }
    }
}

impl<T: JoinVecVariables> JoinVecVariables for &T {
    fn join_vec_variables(&self, res: &mut Vec<Variable>, structure: &mut Vec<usize>) {
        (*self).join_vec_variables(res, structure);
    }
}

pub trait RebuildVecVariables {
    fn rebuild_vec_variables(s: &mut &[Variable], structure: &mut &[usize]) -> Self;
}

impl RebuildVecVariables for Variable {
    fn rebuild_vec_variables(s: &mut &[Variable], _structure: &mut &[usize]) -> Self {
        let res = s[0];
        *s = &s[1..];
        res
    }
}

impl<T: RebuildVecVariables> RebuildVecVariables for Vec<T> {
    fn rebuild_vec_variables(s: &mut &[Variable], structure: &mut &[usize]) -> Self {
        let len = structure[0];
        *structure = &structure[1..];
        let mut res = Vec::with_capacity(len);
        for _ in 0..len {
            res.push(T::rebuild_vec_variables(s, structure));
        }
        res
    }
}

pub trait HashStructureAndPrimitive {
    fn hash_structure_and_primitive(&self, hasher: &mut impl Hasher);
}

impl HashStructureAndPrimitive for Variable {
    fn hash_structure_and_primitive(&self, _hasher: &mut impl Hasher) {}
}

trait Primitive: Display {}
impl Primitive for u64 {}
impl Primitive for &u64 {}
impl Primitive for u32 {}
impl Primitive for &u32 {}
impl Primitive for usize {}
impl Primitive for &usize {}
impl Primitive for i64 {}
impl Primitive for &i64 {}
impl Primitive for i32 {}
impl Primitive for &i32 {}
impl Primitive for isize {}
impl Primitive for &isize {}

impl<T: Primitive> HashStructureAndPrimitive for T {
    fn hash_structure_and_primitive(&self, hasher: &mut impl Hasher) {
        let s = self.to_string();
        hasher.update(s.len().to_string().as_bytes());
        hasher.update(b",");
        hasher.update(s.as_bytes());
    }
}

impl<T: HashStructureAndPrimitive> HashStructureAndPrimitive for Vec<T> {
    fn hash_structure_and_primitive(&self, hasher: &mut impl Hasher) {
        hasher.update(self.len().to_string().as_bytes());
        hasher.update(b";");
        for item in self {
            item.hash_structure_and_primitive(hasher);
        }
    }
}

impl HashStructureAndPrimitive for &Variable {
    fn hash_structure_and_primitive(&self, hasher: &mut impl Hasher) {
        (*self).hash_structure_and_primitive(hasher);
    }
}

impl<T: HashStructureAndPrimitive> HashStructureAndPrimitive for &Vec<T> {
    fn hash_structure_and_primitive(&self, hasher: &mut impl Hasher) {
        (*self).hash_structure_and_primitive(hasher);
    }
}
