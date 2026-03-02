/// macro to implement common arithmetic of field types
#[macro_export]
macro_rules! field_common {
    ($field:ident) => {
        impl Sub<&$field> for $field {
            type Output = $field;

            #[inline]
            fn sub(self, rhs: &$field) -> $field {
                self.sub(*rhs)
            }
        }

        impl Sub<$field> for $field {
            type Output = $field;

            #[inline]
            fn sub(self, rhs: $field) -> $field {
                sub_internal(&self, &rhs)
            }
        }

        impl SubAssign for $field {
            #[inline]
            fn sub_assign(&mut self, rhs: $field) {
                *self = (*self).sub(rhs)
            }
        }

        impl SubAssign<&$field> for $field {
            #[inline]
            fn sub_assign(&mut self, rhs: &$field) {
                *self = (*self).sub(rhs)
            }
        }

        // ========================
        // additions
        // ========================

        impl Add<&$field> for $field {
            type Output = $field;

            #[inline]
            fn add(self, rhs: &$field) -> $field {
                self.add(*rhs)
            }
        }

        impl Add<$field> for $field {
            type Output = $field;

            #[inline]
            fn add(self, rhs: $field) -> $field {
                add_internal(&self, &rhs)
            }
        }

        impl AddAssign for $field {
            #[inline]
            fn add_assign(&mut self, rhs: $field) {
                *self = (*self).add(rhs)
            }
        }

        impl<'b> AddAssign<&'b $field> for $field {
            #[inline]
            fn add_assign(&mut self, rhs: &'b $field) {
                *self = (*self).add(rhs)
            }
        }

        impl<T> Sum<T> for $field
        where
            T: core::borrow::Borrow<Self>,
        {
            fn sum<I>(iter: I) -> Self
            where
                I: Iterator<Item = T>,
            {
                iter.fold(Self::ZERO, |acc, item| acc + item.borrow())
            }
        }

        // ========================
        // multiplications
        // ========================
        impl Mul<$field> for $field {
            type Output = $field;

            #[inline]
            fn mul(self, rhs: $field) -> $field {
                mul_internal(&self, &rhs)
            }
        }

        impl<'b> Mul<&'b $field> for $field {
            type Output = $field;

            #[inline]
            fn mul(self, rhs: &'b $field) -> $field {
                self.mul(*rhs)
            }
        }

        impl Mul<$field> for &$field {
            type Output = $field;

            #[inline(always)]
            fn mul(self, rhs: $field) -> $field {
                *self * rhs
            }
        }

        impl Mul<&$field> for &$field {
            type Output = $field;

            #[inline(always)]
            fn mul(self, rhs: &$field) -> $field {
                *self * *rhs
            }
        }

        impl MulAssign for $field {
            #[inline]
            fn mul_assign(&mut self, rhs: $field) {
                *self = self.clone().mul(rhs)
            }
        }

        impl<'b> MulAssign<&'b $field> for $field {
            #[inline]
            fn mul_assign(&mut self, rhs: &'b $field) {
                *self = self.clone().mul(rhs)
            }
        }

        impl<T> Product<T> for $field
        where
            T: core::borrow::Borrow<Self>,
        {
            fn product<I: Iterator<Item = T>>(iter: I) -> Self {
                iter.fold(Self::one(), |acc, item| acc * item.borrow())
            }
        }
    };
}

#[macro_export]
macro_rules! rep_field_common {
    ($field:ident <const N: usize>) => {
        impl<const N: usize> Sub<&$field<N>> for $field<N> {
            type Output = $field<N>;

            #[inline]
            fn sub(self, rhs: &$field<N>) -> $field<N> {
                self.sub(*rhs)
            }
        }

        impl<const N: usize> Sub<$field<N>> for $field<N> {
            type Output = $field<N>;

            #[inline]
            fn sub(self, rhs: $field<N>) -> $field<N> {
                (&self).sub_internal(&rhs)
            }
        }

        impl<const N: usize> SubAssign for $field<N> {
            #[inline]
            fn sub_assign(&mut self, rhs: $field<N>) {
                *self = (*self).sub(rhs)
            }
        }

        impl<const N: usize> SubAssign<&$field<N>> for $field<N> {
            #[inline]
            fn sub_assign(&mut self, rhs: &$field<N>) {
                *self = (*self).sub(rhs)
            }
        }

        // ========================
        // additions
        // ========================

        impl<const N: usize> Add<&$field<N>> for $field<N> {
            type Output = $field<N>;

            #[inline]
            fn add(self, rhs: &$field<N>) -> $field<N> {
                self.add(*rhs)
            }
        }

        impl<const N: usize> Add<$field<N>> for $field<N> {
            type Output = $field<N>;

            #[inline]
            fn add(self, rhs: $field<N>) -> $field<N> {
                (&self).add_internal(&rhs)
            }
        }

        impl<const N: usize> AddAssign for $field<N> {
            #[inline]
            fn add_assign(&mut self, rhs: $field<N>) {
                *self = (*self).add(rhs)
            }
        }

        impl<'b, const N: usize> AddAssign<&'b $field<N>> for $field<N> {
            #[inline]
            fn add_assign(&mut self, rhs: &'b $field<N>) {
                *self = (*self).add(rhs)
            }
        }

        impl<T, const N: usize> Sum<T> for $field<N>
        where
            T: core::borrow::Borrow<Self>,
        {
            fn sum<I>(iter: I) -> Self
            where
                I: Iterator<Item = T>,
            {
                iter.fold(Self::ZERO, |acc, item| acc + item.borrow())
            }
        }

        impl<const N: usize> Neg for $field<N> {
            type Output = $field<N>;

            #[inline]
            fn neg(self) -> $field<N> {
                Self::ZERO - self
            }
        }

        // ========================
        // multiplications
        // ========================
        impl<const N: usize> Mul<$field<N>> for $field<N> {
            type Output = $field<N>;

            #[inline]
            fn mul(self, rhs: $field<N>) -> $field<N> {
                (&self).mul_internal(&rhs)
            }
        }

        impl<'b, const N: usize> Mul<&'b $field<N>> for $field<N> {
            type Output = $field<N>;

            #[inline]
            fn mul(self, rhs: &'b $field<N>) -> $field<N> {
                self.mul(*rhs)
            }
        }

        impl<const N: usize> Mul<$field<N>> for &$field<N> {
            type Output = $field<N>;

            #[inline(always)]
            fn mul(self, rhs: $field<N>) -> $field<N> {
                *self * rhs
            }
        }

        impl<const N: usize> Mul<&$field<N>> for &$field<N> {
            type Output = $field<N>;

            #[inline(always)]
            fn mul(self, rhs: &$field<N>) -> $field<N> {
                *self * *rhs
            }
        }

        impl<const N: usize> MulAssign for $field<N> {
            #[inline]
            fn mul_assign(&mut self, rhs: $field<N>) {
                *self = self.mul(&rhs)
            }
        }

        impl<'b, const N: usize> MulAssign<&'b $field<N>> for $field<N> {
            #[inline]
            fn mul_assign(&mut self, rhs: &'b $field<N>) {
                *self = self.mul(rhs)
            }
        }

        impl<T, const N: usize> Product<T> for $field<N>
        where
            T: core::borrow::Borrow<Self>,
        {
            fn product<I: Iterator<Item = T>>(iter: I) -> Self {
                iter.fold(Self::one(), |acc, item| acc * item.borrow())
            }
        }
    };
}
