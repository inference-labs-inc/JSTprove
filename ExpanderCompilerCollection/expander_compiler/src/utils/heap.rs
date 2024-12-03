// Handwritten binary min-heap with custom comparator

use std::cmp::Ordering;

pub fn push<F: Fn(usize, usize) -> Ordering>(s: &mut Vec<usize>, x: usize, cmp: F) {
    s.push(x);
    let mut i = s.len() - 1;
    while i > 0 {
        let p = (i - 1) / 2;
        if cmp(s[i], s[p]) == Ordering::Less {
            s.swap(i, p);
            i = p;
        } else {
            break;
        }
    }
}

pub fn pop<F: Fn(usize, usize) -> Ordering>(s: &mut Vec<usize>, cmp: F) -> Option<usize> {
    if s.is_empty() {
        return None;
    }
    let ret = Some(s[0]);
    if s.len() == 1 {
        s.pop();
        return ret;
    }
    s[0] = s.pop().unwrap();
    let mut i = 0;
    while 2 * i + 1 < s.len() {
        let mut j = 2 * i + 1;
        if j + 1 < s.len() && cmp(s[j + 1], s[j]) == Ordering::Less {
            j += 1;
        }
        if cmp(s[j], s[i]) == Ordering::Less {
            s.swap(i, j);
            i = j;
        } else {
            break;
        }
    }
    ret
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use std::collections::BinaryHeap;

    #[test]
    fn test_heap() {
        let mut my_heap = vec![];
        let mut std_heap = BinaryHeap::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        for i in 0..100000 {
            let op = if i < 50000 {
                rng.gen_range(0..2)
            } else {
                rng.gen_range(0..3) % 2
            };
            if op == 0 {
                let x = rng.gen_range(0..100000);
                push(&mut my_heap, x, |a, b| b.cmp(&a));
                std_heap.push(x);
            } else {
                let x = pop(&mut my_heap, |a, b| b.cmp(&a));
                let y = std_heap.pop();
                assert_eq!(x, y);
            }
        }
    }
}
