pub const EMPTY: usize = usize::MAX >> 9;

#[derive(Clone, Debug)]
pub struct InputMapping {
    next_size: usize,
    mapping: Vec<usize>,
}

impl InputMapping {
    pub fn new(next_size: usize, mapping: Vec<usize>) -> Self {
        InputMapping { next_size, mapping }
    }

    pub fn new_identity(next_size: usize) -> Self {
        InputMapping {
            next_size,
            mapping: (0..next_size).collect(),
        }
    }

    pub fn cur_size(&self) -> usize {
        self.mapping.len()
    }

    pub fn next_size(&self) -> usize {
        self.next_size
    }

    pub fn map(&self, pos: usize) -> usize {
        self.mapping[pos]
    }

    pub fn map_inputs<T: Default + Clone>(&self, inputs: &[T]) -> Vec<T> {
        assert_eq!(inputs.len(), self.mapping.len());
        let mut new_inputs = vec![T::default(); self.next_size];
        for i in 0..inputs.len() {
            if self.mapping[i] != EMPTY {
                new_inputs[self.mapping[i]] = inputs[i].clone();
            }
        }
        new_inputs
    }

    pub fn mapping(&self) -> &Vec<usize> {
        &self.mapping
    }

    pub fn validate(&self) -> bool {
        let mut used = vec![false; self.next_size];
        for &m in &self.mapping {
            if m != EMPTY {
                if m >= self.next_size {
                    return false;
                }
                if used[m] {
                    return false;
                }
                used[m] = true;
            }
        }
        for &u in &used {
            if !u {
                return false;
            }
        }
        true
    }

    pub fn compose(&self, other: &InputMapping) -> InputMapping {
        let mut new_mapping = Vec::new();
        for i in 0..self.mapping.len() {
            if self.mapping[i] == EMPTY {
                new_mapping.push(EMPTY);
            } else {
                new_mapping.push(other.mapping[self.mapping[i]]);
            }
        }
        InputMapping {
            next_size: other.next_size,
            mapping: new_mapping,
        }
    }

    pub fn compose_in_place(&mut self, other: &InputMapping) {
        for i in 0..self.mapping.len() {
            if self.mapping[i] != EMPTY {
                self.mapping[i] = other.mapping[self.mapping[i]];
            }
        }
        self.next_size = other.next_size;
    }
}
