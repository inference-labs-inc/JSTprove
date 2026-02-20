use std::collections::HashMap;

use super::graph::OpType;

pub fn get_scale_plan(op: OpType) -> HashMap<usize, usize> {
    match op {
        OpType::Conv => {
            let mut m = HashMap::new();
            m.insert(1, 1);
            m.insert(2, 2);
            m
        }
        OpType::Gemm => {
            let mut m = HashMap::new();
            m.insert(1, 1);
            m.insert(2, 2);
            m
        }
        OpType::BatchNormalization => {
            let mut m = HashMap::new();
            m.insert(1, 1);
            m.insert(2, 2);
            m
        }
        _ => HashMap::new(),
    }
}
