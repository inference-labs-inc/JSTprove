// Copyright 2024-2025 Irreducible Inc.

/// An index mapping positive integer IDs to optional values.
#[derive(Debug, Clone)]
pub struct SparseIndex<T> {
    entries: Vec<Option<T>>,
}

impl<T> SparseIndex<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let mut entries = Vec::with_capacity(capacity);
        entries.resize_with(capacity, || None);
        Self { entries }
    }

    pub fn entry(&mut self, id: usize) -> Entry<'_, T> {
        if self.entries.len() <= id {
            self.entries.resize_with(id + 1, || None);
        }
        Entry(&mut self.entries[id])
    }

    pub fn get(&self, id: usize) -> Option<&T> {
        self.entries.get(id)?.as_ref()
    }

    pub fn get_mut(&mut self, id: usize) -> Option<&mut T> {
        self.entries.get_mut(id)?.as_mut()
    }

    pub fn set(&mut self, id: usize, val: T) {
        if self.entries.len() <= id {
            self.entries.resize_with(id + 1, || None);
        }
        self.entries[id] = Some(val);
    }

    pub fn contains_key(&self, id: usize) -> bool {
        self.entries.get(id).map(|x| x.is_some()).unwrap_or(false)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.iter().all(|v| v.is_none())
    }

    pub fn len(&self) -> usize {
        self.entries.iter().flatten().count()
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (usize, &T)> {
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.as_ref().map(|v| (i, v)))
    }

    pub fn iter_mut(&mut self) -> impl DoubleEndedIterator<Item = (usize, &mut T)> {
        self.entries
            .iter_mut()
            .enumerate()
            .filter_map(|(i, v)| v.as_mut().map(|v| (i, v)))
    }

    pub fn keys(&self) -> impl DoubleEndedIterator<Item = usize> + '_ {
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.as_ref().map(|_| i))
    }

    pub fn values(&self) -> impl DoubleEndedIterator<Item = &T> {
        self.entries.iter().flatten()
    }

    pub fn values_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut T> {
        self.entries.iter_mut().flatten()
    }
}

impl<T> Default for SparseIndex<T> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

impl<T> IntoIterator for SparseIndex<T> {
    type Item = (usize, T);
    type IntoIter = std::iter::FilterMap<
        std::iter::Enumerate<std::vec::IntoIter<Option<T>>>,
        fn((usize, Option<T>)) -> Option<(usize, T)>,
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.entries
            .into_iter()
            .enumerate()
            .filter_map(|(i, v)| v.map(|v| (i, v)))
    }
}

impl<T> std::iter::FromIterator<(usize, T)> for SparseIndex<T> {
    fn from_iter<I: IntoIterator<Item = (usize, T)>>(iter: I) -> Self {
        let mut index = Self::new();
        for (i, v) in iter {
            index.set(i, v);
        }
        index
    }
}

impl<T> std::ops::Index<usize> for SparseIndex<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).unwrap()
    }
}

impl<T> std::ops::IndexMut<usize> for SparseIndex<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).unwrap()
    }
}

pub struct Entry<'a, V: 'a>(&'a mut Option<V>);

impl<'a, V: 'a> Entry<'a, V> {
    pub fn or_insert(self, default: V) -> &'a mut V {
        self.0.get_or_insert(default)
    }

    pub fn or_insert_with(self, f: impl FnOnce() -> V) -> &'a mut V {
        self.0.get_or_insert_with(f)
    }
}
