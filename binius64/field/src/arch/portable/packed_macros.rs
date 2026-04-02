// Copyright 2024-2025 Irreducible Inc.

macro_rules! define_packed_binary_fields {
    (
        underlier: $underlier:ident,
        packed_fields: [
            $(
                packed_field {
                    name: $name:ident,
                    scalar: $scalar:ident,
                    mul: ($($mul:tt)*),
                    square: ($($square:tt)*),
                    invert: ($($invert:tt)*),
                    mul_alpha: ($($mul_alpha:tt)*),
                    transform: ($($transform:tt)*),
                }
            ),* $(,)?
        ]
    ) => {
        $(
            define_packed_binary_field!(
                $name,
                $crate::$scalar,
                $underlier,
                ($($mul)*),
                ($($square)*),
                ($($invert)*),
                ($($mul_alpha)*),
                ($($transform)*)
            );
        )*
    };
}

macro_rules! define_packed_binary_field {
	(
		$name:ident, $scalar:path, $underlier:ident,
		($($mul:tt)*),
		($($square:tt)*),
		($($invert:tt)*),
		($($mul_alpha:tt)*),
		($($transform:tt)*)
	) => {
		// Define packed field types
		pub type $name = $crate::arch::PackedPrimitiveType<$underlier, $scalar>;

		// Define serialization and deserialization
		impl_serialize_deserialize_for_packed_binary_field!($name);

		// Define multiplication
		impl_strategy!(impl_mul_with       $name, ($($mul)*));

		// Define square
		impl_strategy!(impl_square_with    $name, ($($square)*));

		// Define invert
		impl_strategy!(impl_invert_with    $name, ($($invert)*));

		// Define multiply by alpha
		impl_strategy!(impl_mul_alpha_with $name, ($($mul_alpha)*));

		// Define linear transformations
		//impl_transformation!($name, ($($transform)*));
	};
}

macro_rules! impl_serialize_deserialize_for_packed_binary_field {
    ($bin_type:ty) => {
        impl binius_utils::SerializeBytes for $bin_type {
            fn serialize(
                &self,
                write_buf: impl binius_utils::bytes::BufMut,
            ) -> Result<(), binius_utils::SerializationError> {
                self.0.serialize(write_buf)
            }
        }

        impl binius_utils::DeserializeBytes for $bin_type {
            fn deserialize(
                read_buf: impl binius_utils::bytes::Buf,
            ) -> Result<Self, binius_utils::SerializationError> {
                Ok(Self(
                    binius_utils::DeserializeBytes::deserialize(read_buf)?,
                    std::marker::PhantomData,
                ))
            }
        }
    };
}

pub(crate) use define_packed_binary_field;
pub(crate) use define_packed_binary_fields;
pub(crate) use impl_serialize_deserialize_for_packed_binary_field;

pub(crate) use crate::arithmetic_traits::{
    impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with,
};

pub(crate) mod portable_macros {
    macro_rules! impl_strategy {
		($impl_macro:ident $name:ident, (None)) => {};
		// gfni condition: strategy types are in $crate::arch
		($impl_macro:ident $name:ident, (if gfni $strategy:tt else $fallback:tt)) => {
			cfg_if! {
				if #[cfg(all(target_arch = "x86_64", target_feature = "sse2", target_feature = "gfni"))] {
					$impl_macro!($name @ $crate::arch::$strategy);
				} else {
					$impl_macro!($name @ $crate::arch::$fallback);
				}
			}
		};
		// gfni_x86 condition: bigger types are re-exported at $crate root
		($impl_macro:ident $name:ident, (if gfni_x86 $bigger:tt else $fallback:tt)) => {
			cfg_if! {
				if #[cfg(all(target_arch = "x86_64", target_feature = "sse2", target_feature = "gfni"))] {
					$impl_macro!($name => $crate::$bigger);
				} else {
					$impl_macro!($name @ $crate::arch::$fallback);
				}
			}
		};
		// Path to strategy in caller's scope
		($impl_macro:ident $name:ident, ($strategy:path)) => {
			$impl_macro!($name @ $strategy);
		};
	}

    pub(crate) use impl_strategy;
}
