// Copyright 2024-2025 Irreducible Inc.

#[macro_export]
macro_rules! impl_packed_field_display {
    ($name:ident) => {
        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{{")?;
                let mut iter = self.iter();
                if let Some(scalar) = iter.next() {
                    write!(f, "{}", scalar)?;
                }
                for scalar in iter {
                    write!(f, " {}", scalar)?;
                }
                write!(f, "}}")
            }
        }
    };
}
