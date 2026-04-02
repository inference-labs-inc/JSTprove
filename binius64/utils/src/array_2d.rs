// Copyright 2024-2025 Irreducible Inc.

use std::ops::{AddAssign, Deref, DerefMut, Index, IndexMut};

use bytemuck::{Zeroable, allocation::zeroed_vec};

/// 2D array with row-major layout.
#[derive(Debug)]
pub struct Array2D<T, Data: Deref<Target = [T]> = Vec<T>> {
    data: Data,
    rows: usize,
    cols: usize,
}

impl<T: Default + Clone> Array2D<T> {
    /// Create a new 2D array of the given size initialized with default values.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![T::default(); rows * cols],
            rows,
            cols,
        }
    }

    /// Create a new 2D array of the given size initialized with zeroes.
    pub fn zeroes(rows: usize, cols: usize) -> Self
    where
        T: Zeroable,
    {
        Self {
            data: zeroed_vec(rows * cols),
            rows,
            cols,
        }
    }
}

impl<T, Data: Deref<Target = [T]>> Array2D<T, Data> {
    /// Returns the number of rows in the array.
    pub fn rows(&self) -> usize {
        self.data.len() / self.cols
    }

    /// Returns the number of columns in the array.
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the row at the given index.
    pub fn get_row(&self, i: usize) -> &[T] {
        let start = i * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Returns an iterator over the rows of the array.
    pub fn iter_rows(&self) -> impl Iterator<Item = &[T]> {
        (0..self.rows).map(move |i| self.get_row(i))
    }

    /// Return the element at the given row and column without bounds checking.
    /// # Safety
    /// The caller must ensure that `i` and `j` are less than the number of rows and columns
    /// respectively.
    pub unsafe fn get_unchecked(&self, i: usize, j: usize) -> &T {
        unsafe { self.data.get_unchecked(i * self.cols + j) }
    }

    /// View of the array in a different shape, underlying elements stay the same.
    pub fn reshape(&self, rows: usize, cols: usize) -> Option<Array2D<T, &[T]>> {
        if rows * cols != self.data.len() {
            return None;
        }

        Some(Array2D {
            data: &self.data,
            rows,
            cols,
        })
    }
}

impl<T, Data: DerefMut<Target = [T]>> Array2D<T, Data> {
    /// Returns the mutable row at the given index.
    pub fn get_row_mut(&mut self, i: usize) -> &mut [T] {
        let start = i * self.cols;
        &mut self.data[start..start + self.cols]
    }

    /// Return the mutable element at the given row and column without bounds checking.
    /// # Safety
    /// The caller must ensure that `i` and `j` are less than the number of rows and columns
    /// respectively.
    pub unsafe fn get_unchecked_mut(&mut self, i: usize, j: usize) -> &mut T {
        unsafe { self.data.get_unchecked_mut(i * self.cols + j) }
    }

    /// Mutable view of the array in a different shape, underlying elements stay the same.
    pub fn reshape_mut(&mut self, rows: usize, cols: usize) -> Option<Array2D<T, &mut [T]>> {
        if rows * cols != self.data.len() {
            return None;
        }

        Some(Array2D {
            data: self.data.deref_mut(),
            rows,
            cols,
        })
    }
}

impl<T, Data: Deref<Target = [T]>> Index<(usize, usize)> for Array2D<T, Data> {
    type Output = T;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.data[i * self.cols + j]
    }
}

impl<T, Data: DerefMut<Target = [T]>> IndexMut<(usize, usize)> for Array2D<T, Data> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        &mut self.data[i * self.cols + j]
    }
}

impl<T: Default + Clone + AddAssign, Data: Deref<Target = [T]>> Array2D<T, Data> {
    /// Returns the sum of the elements in each row.
    pub fn sum_rows(&self) -> Vec<T> {
        let mut sum = vec![T::default(); self.cols];

        for row in self.iter_rows() {
            for (i, elem) in row.iter().enumerate() {
                sum[i] += elem.clone();
            }
        }

        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_set() {
        let mut arr = Array2D::new(2, 3);
        arr[(0, 0)] = 1;
        arr[(0, 1)] = 2;
        arr[(0, 2)] = 3;
        arr[(1, 0)] = 4;
        arr[(1, 1)] = 5;
        arr[(1, 2)] = 6;

        assert_eq!(arr[(0, 0)], 1);
        assert_eq!(arr[(0, 1)], 2);
        assert_eq!(arr[(0, 2)], 3);
        assert_eq!(arr[(1, 0)], 4);
        assert_eq!(arr[(1, 1)], 5);
        assert_eq!(arr[(1, 2)], 6);
    }

    #[test]
    fn test_unchecked_access() {
        let mut arr = Array2D::new(2, 3);
        unsafe {
            *arr.get_unchecked_mut(0, 0) = 1;
            *arr.get_unchecked_mut(0, 1) = 2;
            *arr.get_unchecked_mut(0, 2) = 3;
            *arr.get_unchecked_mut(1, 0) = 4;
            *arr.get_unchecked_mut(1, 1) = 5;
            *arr.get_unchecked_mut(1, 2) = 6;
        }

        unsafe {
            assert_eq!(*arr.get_unchecked(0, 0), 1);
            assert_eq!(*arr.get_unchecked(0, 1), 2);
            assert_eq!(*arr.get_unchecked(0, 2), 3);
            assert_eq!(*arr.get_unchecked(1, 0), 4);
            assert_eq!(*arr.get_unchecked(1, 1), 5);
            assert_eq!(*arr.get_unchecked(1, 2), 6);
        }
    }

    #[test]
    fn test_get_row() {
        let mut arr = Array2D::new(2, 3);
        arr[(0, 0)] = 1;
        arr[(0, 1)] = 2;
        arr[(0, 2)] = 3;
        arr[(1, 0)] = 4;
        arr[(1, 1)] = 5;
        arr[(1, 2)] = 6;

        assert_eq!(arr.get_row(0), &[1, 2, 3]);
        assert_eq!(arr.get_row_mut(0), &mut [1, 2, 3]);
        assert_eq!(arr.get_row(1), &[4, 5, 6]);
        assert_eq!(arr.get_row_mut(1), &mut [4, 5, 6]);
    }

    #[test]
    fn test_sum_rows() {
        let mut arr = Array2D::new(2, 3);
        arr[(0, 0)] = 1;
        arr[(0, 1)] = 2;
        arr[(0, 2)] = 3;
        arr[(1, 0)] = 4;
        arr[(1, 1)] = 5;
        arr[(1, 2)] = 6;

        assert_eq!(arr.sum_rows(), vec![5, 7, 9]);
    }
}
