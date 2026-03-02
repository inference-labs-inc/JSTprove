use serdes::ExpSerde;

use crate::{circuit::input_mapping::InputMapping, utils::misc::next_power_of_two};

pub type Shape = Vec<usize>;
pub type Axes = Vec<usize>;
/*
Bit order definition:
Suppose bit_order = [a_0, a_1, a_2, ...]
Then when we read the i-th position, where i = sum(b_j * 2^j), b_j = 0 or 1,
we will read the j-th position, where j = sum(b_j * 2^(a_j)).
*/
pub type BitOrder = Vec<usize>;

pub fn shape_prepend(shape: &Shape, x: usize) -> Shape {
    let mut shape = shape.clone();
    shape.insert(0, x);
    shape
}

#[derive(Debug, Clone, ExpSerde)]
pub struct ShapeHistory {
    vec_len: usize,
    entries: Vec<Entry>,
}

#[derive(Debug, Clone, ExpSerde)]
struct Entry {
    shape: Shape,
    axes: Option<Axes>,
}

impl Entry {
    fn minimize(&self, keep_first_dim: bool) -> Self {
        let axes = match &self.axes {
            None => {
                let new_shape = if !keep_first_dim || self.shape.len() <= 1 {
                    vec![shape_vec_len(&self.shape)]
                } else {
                    vec![self.shape[0], shape_vec_len(&self.shape[1..])]
                };
                return Entry {
                    shape: new_shape,
                    axes: None,
                };
            }
            Some(axes) => axes,
        };
        let mut merge_shape = vec![false; self.shape.len()];
        for (i, (&a, &b)) in axes.iter().zip(axes.iter().skip(1)).enumerate() {
            if a + 1 == b && (!keep_first_dim || i > 0) {
                merge_shape[b] = true;
            }
        }
        let mut new_shape_id = vec![0; self.shape.len()];
        let mut new_shape = Vec::with_capacity(self.shape.len());
        for (i, &s) in self.shape.iter().enumerate() {
            if !merge_shape[i] {
                new_shape_id[i] = new_shape.len();
                new_shape.push(s);
            } else {
                *new_shape.last_mut().unwrap() *= s;
            }
        }
        let mut new_axes = Vec::with_capacity(axes.len());
        for &a in axes {
            if !merge_shape[a] {
                new_axes.push(new_shape_id[a]);
            }
        }
        Entry {
            shape: new_shape,
            axes: Some(new_axes),
        }
    }
    fn transposed_shape(&self) -> Shape {
        match &self.axes {
            None => self.shape.clone(),
            Some(axes) => axes.iter().map(|&a| self.shape[a]).collect(),
        }
    }
    fn transpose_shape(&self, shape: &[(usize, usize)]) -> Vec<(usize, usize)> {
        if self.axes.is_none() {
            return shape.to_vec();
        }
        let mut segments = vec![];
        let mut cur_prod = 1;
        let mut target = 1;
        let mut self_shape_iter = self.shape.iter();
        for &x in shape.iter() {
            if cur_prod == target {
                cur_prod = x.0;
                target = *self_shape_iter.next().unwrap();
                segments.push(vec![x]);
            } else {
                cur_prod *= x.0;
                segments.last_mut().unwrap().push(x);
            }
        }
        assert_eq!(cur_prod, target);
        assert_eq!(self_shape_iter.next(), None);
        let mut res = Vec::with_capacity(shape.len());
        for i in self.axes.as_ref().unwrap() {
            res.extend(segments[*i].iter());
        }
        res
    }
    fn undo_transpose_shape_products(&self, products: &[usize]) -> Vec<usize> {
        if self.axes.is_none() {
            return products.to_vec();
        }
        let ts = self.transposed_shape();
        let mut segments_in_ts = vec![Vec::new(); ts.len()];
        let mut cur_ts_prod = 1;
        let mut cur_ts_idx = 0;
        for &x in products.iter().skip(1) {
            segments_in_ts[self.axes.as_ref().unwrap()[cur_ts_idx]].push(x / cur_ts_prod);
            if x == cur_ts_prod * ts[cur_ts_idx] {
                cur_ts_prod = x;
                cur_ts_idx += 1;
            }
        }
        let mut res = Vec::with_capacity(products.len() + 1);
        res.push(1);
        let mut cur_prod = 1;
        for (x, &shape_i) in segments_in_ts.iter().zip(self.shape.iter()) {
            for &y in x.iter() {
                res.push(cur_prod * y);
            }
            cur_prod *= shape_i;
        }
        res
    }
}

pub trait Reshape {
    fn reshape(&self, new_shape: &[usize]) -> Self;
}

pub trait Transpose {
    fn transpose(&self, axes: &[usize]) -> Self;
}

pub fn shape_vec_len(shape: &[usize]) -> usize {
    shape.iter().product()
}

pub fn shape_vec_padded_len(shape: &[usize]) -> usize {
    shape.iter().map(|&x| next_power_of_two(x)).product()
}

pub fn prefix_products(shape: &[usize]) -> Vec<usize> {
    let mut products = Vec::with_capacity(shape.len() + 1);
    let mut product = 1;
    products.push(1);
    for &dim in shape {
        product *= dim;
        products.push(product);
    }
    products
}

pub fn prefix_products_to_shape(products: &[usize]) -> Vec<usize> {
    let mut shape = Vec::with_capacity(products.len() - 1);
    for i in 1..products.len() {
        shape.push(products[i] / products[i - 1]);
    }
    shape
}

pub fn merge_shape_products(a: &[usize], b: &[usize]) -> Vec<usize> {
    assert_eq!(a[0], 1);
    assert_eq!(b[0], 1);
    assert_eq!(a.last().unwrap(), b.last().unwrap());
    let mut all: Vec<usize> = a.iter().chain(b.iter()).cloned().collect();
    all.sort();
    all.dedup();
    for (x, y) in all.iter().zip(all.iter().skip(1)) {
        assert_eq!(y % x, 0, "Detected illegal shape operation");
    }
    all
}

pub fn keep_shape_products_until(shape: &[usize], x: usize) -> Vec<usize> {
    let p = shape.iter().position(|&y| y == x).unwrap();
    shape[..=p].to_vec()
}

pub fn keep_shape_until(shape: &[usize], x: usize) -> Vec<usize> {
    let mut p = 1;
    if x == 1 {
        return Vec::new();
    }
    for (i, &y) in shape.iter().enumerate() {
        p *= y;
        if p == x {
            return shape[..=i].to_vec();
        }
    }
    unreachable!()
}

pub fn keep_shape_since(shape: &[usize], x: usize) -> Vec<usize> {
    let mut p = 1;
    if x == 1 {
        return shape.to_vec();
    }
    for (i, &y) in shape.iter().enumerate() {
        p *= y;
        if p == x {
            return shape[i + 1..].to_vec();
        }
    }
    unreachable!()
}

pub fn shape_padded_mapping(shape: &[usize]) -> InputMapping {
    let mut cur = vec![0];
    let mut step = 1;
    for &len in shape.iter().rev() {
        let mut new = cur.clone();
        for i in 1..len {
            new.extend(cur.iter().map(|&y| y + i * step));
        }
        step *= next_power_of_two(len);
        cur = new;
    }
    InputMapping::new(step, cur)
}

impl ShapeHistory {
    pub fn new(initial_shape: Shape) -> Self {
        Self {
            vec_len: shape_vec_len(&initial_shape),
            entries: vec![Entry {
                shape: initial_shape,
                axes: None,
            }],
        }
    }

    // Suppose we need to ensure that the current shape is legal
    // This function returns a list of dimension lengths where the initial vector must be split
    // split_first_dim: first dimension of current shape will be split
    pub fn get_initial_split_list(&self, split_first_dim: bool) -> Vec<usize> {
        let last_entry = self.entries.last().unwrap().minimize(split_first_dim);
        let mut split_list = prefix_products(&last_entry.shape);
        for e in self.entries.iter().rev().skip(1) {
            let e = e.minimize(false);
            assert!(e.axes.is_some());
            let et = e.transposed_shape();
            let merged_split_list = merge_shape_products(&split_list, &prefix_products(&et));
            split_list = e.undo_transpose_shape_products(&merged_split_list);
        }
        split_list
    }

    pub fn get_transposed_shape_and_bit_order(&self, shape: &[usize]) -> (Shape, BitOrder) {
        let mut cur = None;
        let initial_shape = || {
            shape
                .iter()
                .enumerate()
                .map(|(i, &x)| (x, i))
                .collect::<Vec<_>>()
        };
        for e in self.entries.iter() {
            cur = if e.axes.as_ref().is_none() {
                cur
            } else if cur.is_none() {
                Some(e.transpose_shape(&initial_shape()))
            } else {
                Some(e.transpose_shape(&cur.unwrap()))
            };
        }
        let new_shape_and_id = match cur {
            None => initial_shape(),
            Some(transposed_shape) => transposed_shape,
        };
        let bit_len = shape
            .iter()
            .map(|&x| next_power_of_two(x).trailing_zeros() as usize)
            .collect::<Vec<_>>();
        let mut bit_start = vec![0];
        for &x in bit_len.iter().rev() {
            bit_start.push(bit_start.last().unwrap() + x);
        }
        let mut bit_order = Vec::new();
        for &x in new_shape_and_id.iter().rev() {
            let n = bit_len[x.1];
            let k = bit_start[bit_len.len() - x.1 - 1];
            for i in 0..n {
                bit_order.push(k + i);
            }
        }
        (
            new_shape_and_id.iter().map(|&(x, _)| x).collect(),
            bit_order,
        )
    }

    pub fn shape(&self) -> Shape {
        let last_entry = self.entries.last().unwrap();
        match &last_entry.axes {
            None => last_entry.shape.clone(),
            Some(axes) => axes.iter().map(|&a| last_entry.shape[a]).collect(),
        }
    }

    pub fn permute_vec<T: Default + Clone>(&self, s: &[T]) -> Vec<T> {
        let mut idx = None;
        for e in self.entries.iter() {
            if e.axes.is_none() {
                break;
            }
            let mut ts = e.shape.clone();
            ts.reverse();
            let ts_mul = prefix_products(&ts);
            let mut cur = vec![0];
            for x in e.axes.as_ref().unwrap().iter().rev() {
                let len = ts[ts.len() - x - 1];
                let step = ts_mul[ts.len() - x - 1];
                let mut new = cur.clone();
                for i in 1..len {
                    new.extend(cur.iter().map(|&y| y + i * step));
                }
                cur = new;
            }
            if idx.is_none() {
                idx = Some(cur);
            } else {
                let t = idx.unwrap();
                assert_eq!(t.len(), cur.len());
                idx = Some(cur.iter().map(|&x| t[x]).collect());
            }
        }
        match idx {
            None => s.to_vec(),
            Some(idx) => idx.iter().map(|&x| s[x].clone()).collect(),
        }
    }
}

impl Reshape for ShapeHistory {
    fn reshape(&self, new_shape: &[usize]) -> Self {
        if shape_vec_len(new_shape) != self.vec_len {
            panic!("Reshape to a shape with different vector length is not allowed, expected {}, got {}", self.vec_len, shape_vec_len(new_shape));
        }
        let mut res = self.clone();
        if res.entries.last().unwrap().axes.is_none() {
            res.entries.last_mut().unwrap().shape = new_shape.to_vec();
        } else {
            res.entries.push(Entry {
                shape: new_shape.to_vec(),
                axes: None,
            });
        }
        res
    }
}

impl Transpose for ShapeHistory {
    fn transpose(&self, axes: &[usize]) -> Self {
        if axes.len() != self.entries.last().unwrap().shape.len() {
            panic!(
                "Transpose axes length must match the shape length, expected {}, got {}",
                self.entries.last().unwrap().shape.len(),
                axes.len()
            );
        }
        let mut res = self.clone();
        match &res.entries.last().unwrap().axes {
            None => {
                res.entries.last_mut().unwrap().axes = Some(axes.to_vec());
            }
            Some(cur_axes) => {
                let new_axes: Vec<usize> = axes.iter().map(|&a| cur_axes[a]).collect();
                res.entries.last_mut().unwrap().axes = Some(new_axes);
            }
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entry_minimize() {
        let entry = Entry {
            shape: vec![2, 3, 4],
            axes: None,
        };
        let minimized = entry.minimize(true);
        assert_eq!(minimized.shape, vec![2, 12]);
        assert_eq!(minimized.axes, None);
        let minimized = entry.minimize(false);
        assert_eq!(minimized.shape, vec![24]);
        assert_eq!(minimized.axes, None);

        let entry = Entry {
            shape: vec![2, 3, 4],
            axes: Some(vec![0, 1, 2]),
        };
        let minimized = entry.minimize(true);
        assert_eq!(minimized.shape, vec![2, 12]);
        assert_eq!(minimized.axes, Some(vec![0, 1]));
        let minimized = entry.minimize(false);
        assert_eq!(minimized.shape, vec![24]);
        assert_eq!(minimized.axes, Some(vec![0]));

        let entry = Entry {
            shape: vec![2, 3, 4],
            axes: Some(vec![2, 0, 1]),
        };
        let minimized = entry.minimize(true);
        assert_eq!(minimized.shape, vec![6, 4]);
        assert_eq!(minimized.axes, Some(vec![1, 0]));
        let minimized = entry.minimize(false);
        assert_eq!(minimized.shape, vec![6, 4]);
        assert_eq!(minimized.axes, Some(vec![1, 0]));

        let entry = Entry {
            shape: vec![2, 3, 5, 7],
            axes: Some(vec![2, 3, 0, 1]),
        };
        let minimized = entry.minimize(true);
        assert_eq!(minimized.shape, vec![6, 5, 7]);
        assert_eq!(minimized.axes, Some(vec![1, 2, 0]));
        let minimized = entry.minimize(false);
        assert_eq!(minimized.shape, vec![6, 35]);
        assert_eq!(minimized.axes, Some(vec![1, 0]));
    }

    #[test]
    fn test_undo_transpose_shape_products() {
        let entry = Entry {
            shape: vec![4, 4, 4, 4, 4],
            axes: None,
        };
        let products = vec![1, 2, 4, 16, 32, 64, 256, 512, 1024];
        let result = entry.undo_transpose_shape_products(&products);
        assert_eq!(result, products);

        let entry = Entry {
            shape: vec![4, 4, 4, 4, 4],
            axes: Some(vec![0, 1, 2, 3, 4]),
        };
        let products = vec![1, 2, 4, 16, 32, 64, 256, 512, 1024];
        let result = entry.undo_transpose_shape_products(&products);
        assert_eq!(result, products);

        let entry = Entry {
            shape: vec![4, 4, 4, 4, 4],
            axes: Some(vec![2, 4, 1, 0, 3]),
        };
        let products = vec![1, 2, 4, 16, 32, 64, 256, 512, 1024];
        let result = entry.undo_transpose_shape_products(&products);
        assert_eq!(result, vec![1, 4, 8, 16, 32, 64, 128, 256, 1024]);
    }

    #[test]
    fn test_merge_shape_products() {
        let a = vec![1, 2, 4, 16];
        let b = vec![1, 2, 8, 16];
        let merged = merge_shape_products(&a, &b);
        assert_eq!(merged, vec![1, 2, 4, 8, 16]);
    }

    #[test]
    #[should_panic]
    fn test_merge_shape_products_invalid() {
        let a = vec![1, 2, 4, 12];
        let b = vec![1, 3, 6, 12];
        merge_shape_products(&a, &b);
    }

    #[test]
    fn test_get_initial_split_list() {
        let sh = ShapeHistory::new(vec![16, 9]);
        let sh = sh.reshape(&[9, 16]);
        assert_eq!(sh.get_initial_split_list(false), vec![1, 144]);
        assert_eq!(sh.get_initial_split_list(true), vec![1, 9, 144]);
        let sh = sh.reshape(&[3, 16, 3]);
        assert_eq!(sh.get_initial_split_list(true), vec![1, 3, 144]);
        let sh = sh.reshape(&[2, 2, 2, 2, 3, 3]);
        let sh = sh.transpose(&[1, 0, 2, 3, 4, 5]);
        assert_eq!(sh.get_initial_split_list(false), vec![1, 2, 4, 144]);
        assert_eq!(sh.get_initial_split_list(true), vec![1, 2, 4, 144]);
        let sh = sh.reshape(&[16, 9]);
        assert_eq!(sh.get_initial_split_list(false), vec![1, 2, 4, 144]);
        assert_eq!(sh.get_initial_split_list(true), vec![1, 2, 4, 16, 144]);
        let sh = sh.transpose(&[1, 0]);
        assert_eq!(sh.get_initial_split_list(false), vec![1, 2, 4, 16, 144]);
        assert_eq!(sh.get_initial_split_list(true), vec![1, 2, 4, 16, 144]);
        let sh = sh.reshape(&[3, 3, 16]);
        assert_eq!(sh.get_initial_split_list(false), vec![1, 2, 4, 16, 144]);
        assert_eq!(sh.get_initial_split_list(true), vec![1, 2, 4, 16, 48, 144]);
    }

    #[test]
    fn test_get_initial_split_list_invalid() {
        let sh = ShapeHistory::new(vec![16, 9]);
        let sh = sh.transpose(&[1, 0]);
        let sh = sh.reshape(&[16, 9]);
        sh.get_initial_split_list(false);
        assert!(std::panic::catch_unwind(|| {
            sh.get_initial_split_list(true);
        })
        .is_err());
    }

    #[test]
    fn test_get_transposed_shape_and_bit_order() {
        let sh = ShapeHistory::new(vec![125, 125]);
        let sh = sh.transpose(&[1, 0]);
        assert_eq!(
            sh.get_transposed_shape_and_bit_order(&[5, 5, 5, 5, 5, 5]),
            (
                vec![5, 5, 5, 5, 5, 5],
                vec![9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8]
            )
        );
        assert_eq!(
            sh.get_transposed_shape_and_bit_order(&[5, 5, 5, 25, 5]),
            (
                vec![25, 5, 5, 5, 5],
                vec![8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 1, 2, 3, 4, 5, 6, 7]
            )
        );
    }

    #[test]
    fn test_permute_vec() {
        let sh = ShapeHistory::new(vec![2, 3, 2]);
        assert_eq!(
            sh.permute_vec(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        );
        let sh = sh.transpose(&[2, 0, 1]);
        assert_eq!(
            sh.permute_vec(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
            vec![1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12]
        );
        let sh = sh.reshape(&[2, 3, 2]);
        assert_eq!(
            sh.permute_vec(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
            vec![1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12]
        );
        let sh = sh.transpose(&[2, 0, 1]);
        assert_eq!(
            sh.permute_vec(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
            vec![1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
        );
    }
}
