pub fn bucket_sort<T: Ord + Clone + Default, F: Fn(&T) -> usize>(
    arr: Vec<T>,
    f: F,
    num_buckets: usize,
) -> (Vec<T>, Vec<usize>) {
    let n = arr.len();
    let m = num_buckets;
    let mut cnt = vec![0; m];
    for i in 0..n {
        cnt[f(&arr[i])] += 1;
    }
    for i in 1..m {
        cnt[i] += cnt[i - 1];
    }
    let mut res = vec![T::default(); n];
    for i in (0..n).rev() {
        let p = f(&arr[i]);
        cnt[p] -= 1;
        res[cnt[p]] = arr[i].clone();
    }
    (res, cnt)
}
