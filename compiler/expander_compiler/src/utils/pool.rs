use std::{collections::HashMap, hash::Hash};

#[derive(Default, Clone)]
pub struct Pool<V> {
    vec: Vec<V>,
    map: HashMap<V, usize>,
}

impl<V> Pool<V>
where
    V: Hash + Eq + Clone,
{
    pub fn new() -> Self {
        Pool {
            vec: Vec::new(),
            map: HashMap::new(),
        }
    }

    pub fn add(&mut self, v: &V) -> usize {
        if let Some(&idx) = self.map.get(v) {
            return idx;
        }
        let idx = self.vec.len();
        self.vec.push(v.clone());
        self.map.insert(v.clone(), idx);
        idx
    }

    pub fn get_idx(&self, val: &V) -> usize {
        *self.map.get(val).expect("Pool value does not exist")
    }

    pub fn try_get_idx(&self, val: &V) -> Option<usize> {
        self.map.get(val).cloned()
    }

    pub fn get(&self, idx: usize) -> &V {
        self.vec.get(idx).expect("Pool index out of bounds")
    }

    pub fn vec(&self) -> &Vec<V> {
        &self.vec
    }

    pub fn map(&self) -> &HashMap<V, usize> {
        &self.map
    }

    pub fn len(&self) -> usize {
        self.vec.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vec.is_empty()
    }
}
