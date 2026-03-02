use arith::SimdField;

use crate::field::FieldRaw;

pub trait VecShaped<T: Clone + Default> {
    fn flatten_shaped(&self, to: &mut Vec<T>) -> Vec<usize>;
    fn unflatten_shaped<'a>(&mut self, s: &'a [T], shape: &[usize]) -> &'a [T];
}

impl<T: FieldRaw> VecShaped<T> for T {
    fn flatten_shaped(&self, to: &mut Vec<T>) -> Vec<usize> {
        to.push(*self);
        vec![]
    }
    fn unflatten_shaped<'a>(&mut self, s: &'a [T], shape: &[usize]) -> &'a [T] {
        if !shape.is_empty() {
            panic!("Shape dimension mismatch in unflatten");
        }
        if s.is_empty() {
            panic!("Shape mismatch in unflatten");
        }
        *self = s[0];
        &s[1..]
    }
}

impl<T, V> VecShaped<T> for Vec<V>
where
    T: Clone + Default,
    V: VecShaped<T> + Default + Clone,
{
    fn flatten_shaped(&self, to: &mut Vec<T>) -> Vec<usize> {
        if self.is_empty() {
            panic!("Empty vector is not allowed for flatten");
        }
        let sub_shape = self[0].flatten_shaped(to);
        for v in self.iter().skip(1) {
            let cur_shape = v.flatten_shaped(to);
            if cur_shape != sub_shape {
                panic!("Shape mismatch in flatten");
            }
        }
        let mut shape = sub_shape;
        shape.push(self.len());
        shape
    }
    fn unflatten_shaped<'a>(&mut self, mut s: &'a [T], shape: &[usize]) -> &'a [T] {
        if shape.is_empty() {
            panic!("Shape dimension mismatch in unflatten");
        }
        let cur_len = shape[0];
        *self = vec![V::default(); cur_len];
        let sub_shape = &shape[1..];
        for v in self.iter_mut() {
            s = v.unflatten_shaped(s, sub_shape);
        }
        s
    }
}

pub fn flatten_shaped<T: FieldRaw, V: VecShaped<T>>(v: &V) -> (Vec<T>, Vec<usize>) {
    let mut to = Vec::new();
    let shape = v.flatten_shaped(&mut to).into_iter().rev().collect();
    (to, shape)
}

pub fn unflatten_shaped<T: FieldRaw, V: VecShaped<T> + Default>(mut s: &[T], shape: &[usize]) -> V {
    let mut v = V::default();
    s = v.unflatten_shaped(s, shape);
    if !s.is_empty() {
        panic!("Shape mismatch in unflatten");
    }
    v
}

// Auto pack simd

pub fn flatten_shaped_pack_simd<F: FieldRaw, V: VecShaped<F>, SimdF: SimdField<Scalar = F>>(
    v: &V,
) -> (Vec<SimdF>, Vec<usize>) {
    let mut to = Vec::new();
    let shape: Vec<usize> = v.flatten_shaped(&mut to).into_iter().rev().collect();
    assert_eq!(shape.first(), Some(&SimdF::PACK_SIZE));
    let mut to_simd = Vec::new();
    let len = to.len() / SimdF::PACK_SIZE;
    for i in 0..len {
        let mut vals = Vec::with_capacity(SimdF::PACK_SIZE);
        for j in 0..SimdF::PACK_SIZE {
            vals.push(to[j * len + i]);
        }
        to_simd.push(SimdF::pack(&vals));
    }
    (to_simd, shape[1..].to_vec())
}

pub fn unflatten_shaped_unpack_simd<
    F: FieldRaw,
    V: VecShaped<F> + Default,
    SimdF: SimdField<Scalar = F>,
>(
    s: &[SimdF],
    shape: &[usize],
) -> V {
    let mut v = V::default();
    let mut s2 = vec![F::default(); s.len() * SimdF::PACK_SIZE];
    let len = s.len();
    for i in 0..len {
        let vals = s[i].unpack();
        for j in 0..SimdF::PACK_SIZE {
            s2[j * len + i] = vals[j];
        }
    }
    let mut new_shape = vec![SimdF::PACK_SIZE];
    new_shape.extend_from_slice(shape);
    let res = v.unflatten_shaped(&s2, &new_shape);
    if !res.is_empty() {
        panic!("Shape mismatch in unflatten");
    }
    v
}
