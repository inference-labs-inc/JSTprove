use frontend::layouter::builder::NodeRef;
use shared_types::Field;

use crate::padding::{next_power_of_two, num_vars_for};

#[derive(Clone, Debug)]
pub struct ShapedMLE<F: Field> {
    pub node: NodeRef<F>,
    pub shape: Vec<usize>,
    pub padded_total: usize,
    pub num_vars: usize,
}

impl<F: Field> ShapedMLE<F> {
    pub fn new(node: NodeRef<F>, shape: Vec<usize>) -> Self {
        let total: usize = shape.iter().product();
        let padded_total = next_power_of_two(total);
        let num_vars = num_vars_for(total);
        Self {
            node,
            shape,
            padded_total,
            num_vars,
        }
    }

    pub fn total_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn flat_index(&self, coords: &[usize]) -> usize {
        assert_eq!(coords.len(), self.shape.len());
        let mut idx = 0;
        let mut stride = 1;
        for i in (0..self.shape.len()).rev() {
            idx += coords[i] * stride;
            stride *= self.shape[i];
        }
        idx
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let new_total: usize = new_shape.iter().product();
        assert_eq!(
            self.total_elements(),
            new_total,
            "reshape: total elements must match"
        );
        Self {
            node: self.node.clone(),
            shape: new_shape,
            padded_total: self.padded_total,
            num_vars: self.num_vars,
        }
    }

    pub fn with_node(&self, node: NodeRef<F>) -> Self {
        Self {
            node,
            shape: self.shape.clone(),
            padded_total: self.padded_total,
            num_vars: self.num_vars,
        }
    }
}

pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

pub fn flat_to_coords(flat: usize, shape: &[usize]) -> Vec<usize> {
    let strides = compute_strides(shape);
    let mut coords = vec![0usize; shape.len()];
    let mut remaining = flat;
    for i in 0..shape.len() {
        coords[i] = remaining / strides[i];
        remaining %= strides[i];
    }
    coords
}
