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
        OpType::LayerNormalization => {
            let mut m = HashMap::new();
            m.insert(1, 1); // Scale/gamma quantised at α¹
            m.insert(2, 2); // B/beta quantised at α²
            m
        }
        OpType::GridSample => {
            let mut m = HashMap::new();
            m.insert(1, 1); // grid normalised coordinates quantised at α¹
            m
        }
        OpType::MatMul => {
            // If input[1] is a constant weight, quantise at α¹.
            let mut m = HashMap::new();
            m.insert(1, 1);
            m
        }
        OpType::ConvTranspose => {
            // Same as Conv: weights at α¹, biases at α².
            let mut m = HashMap::new();
            m.insert(1, 1);
            m.insert(2, 2);
            m
        }
        _ => HashMap::new(),
    }
}
