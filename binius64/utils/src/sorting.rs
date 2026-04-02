// Copyright 2024-2025 Irreducible Inc.

use itertools::Itertools;

/// Returns whether the given values are sorted in ascending order.
pub fn is_sorted_ascending<T: PartialOrd + Clone>(values: impl Iterator<Item = T>) -> bool {
    !values.tuple_windows().any(|(a, b)| a > b)
}

/// Stable sorts a collection of objects based on a key function and an optional descending flag.
///
/// This function takes a collection of objects, sorts them stably based on the value returned by a
/// key function, and can optionally use descending sort order. It returns a tuple containing a
/// vector of the original indices of the objects and a vector of the sorted objects.
///
/// # Arguments
///
/// * `objs`: An iterable collection of objects to be sorted.
/// * `key`: A function that takes a reference to an object and returns a `usize` value used for
///   sorting.
/// * `descending`: A boolean flag indicating whether to sort in descending order.
///
/// # Returns
///
/// A tuple where the first element is a vector of the original indices of the objects and the
/// second element is a vector of the sorted objects.
///
/// # Note
///
/// This function uses stable sorting to ensure consistency in ordering objects that compare as
/// equal.
pub fn stable_sort<T>(
    objs: impl IntoIterator<Item = T>,
    key: impl Fn(&T) -> usize,
    descending: bool,
) -> (Vec<usize>, Vec<T>) {
    let mut indexed_objs = objs.into_iter().enumerate().collect::<Vec<_>>();
    // NOTE: Important to use stable sorting for prover-verifier consistency!
    if descending {
        indexed_objs.sort_by(|a, b| key(&b.1).cmp(&key(&a.1)));
    } else {
        indexed_objs.sort_by(|a, b| key(&a.1).cmp(&key(&b.1)));
    }
    indexed_objs.into_iter().unzip()
}

/// Restores the original order of a collection of objects based on their original indices.
///
/// This function takes a collection of objects and their corresponding original indices, and
/// returns a vector of the objects sorted back to their original order.
///
/// # Arguments
///
/// - `original_indices`: An iterable collection of the original indices of the objects.
/// - `sorted_objs`: An iterable collection of the objects that have been sorted.
///
/// # Returns
///
/// A vector of objects restored to their original order based on the provided indices.
///
/// # Note
///
/// This function assumes that the length of `original_indices` and `sorted_objs` are the same, and
/// that `original_indices` contains unique values representing valid indices.
pub fn unsort<T>(
    original_indices: impl IntoIterator<Item = usize>,
    sorted_objs: impl IntoIterator<Item = T>,
) -> Vec<T> {
    let mut temp = original_indices
        .into_iter()
        .zip(sorted_objs)
        .collect::<Vec<_>>();
    temp.sort_by_key(|(i, _)| *i);
    temp.into_iter().map(|(_, obj)| obj).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stable_sort() {
        let items = vec![
            ("apple", 3),
            ("banana", 2),
            ("cherry", 3),
            ("date", 1),
            ("elderberry", 2),
        ];

        // Sort by the second element of the tuple (the number) in ascending order
        let key = |item: &(&str, usize)| item.1;

        let (indices_asc, sorted_items_asc) = stable_sort(items.clone(), key, false);
        assert_eq!(indices_asc, vec![3, 1, 4, 0, 2]); // Expected original indices
        assert_eq!(
            sorted_items_asc,
            vec![
                ("date", 1),
                ("banana", 2),
                ("elderberry", 2),
                ("apple", 3),
                ("cherry", 3)
            ]
        );

        // Sort by the second element of the tuple (the number) in descending order
        let (indices_desc, sorted_items_desc) = stable_sort(items, key, true);
        assert_eq!(indices_desc, vec![0, 2, 1, 4, 3]); // Expected original indices
        assert_eq!(
            sorted_items_desc,
            vec![
                ("apple", 3),
                ("cherry", 3),
                ("banana", 2),
                ("elderberry", 2),
                ("date", 1)
            ]
        );
    }
}
