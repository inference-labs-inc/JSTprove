// Copyright 2025 Irreducible Inc.

use std::mem::MaybeUninit;

/// Turn a given slice of `T` into a slice of `MaybeUninit<T>`.
///
/// # Panics
///
/// This function is going to panic in case `T` has a destructor.
pub fn slice_uninit_mut<T>(slice: &mut [T]) -> &mut [MaybeUninit<T>] {
    assert!(!std::mem::needs_drop::<T>());
    unsafe {
        // SAFETY:
        //
        // The `slice` is a valid mutable reference, so:
        // - Its pointer is non-null and properly aligned
        // - It points to `len` consecutive properly initialized values of type T
        // - The memory region is valid for reads and writes
        // - The memory belongs to a single allocated object
        // - The total size is no larger than isize::MAX and doesn't wrap around the address space
        //
        // By casting the pointer to `*mut MaybeUninit<T>`, we're essentially forgetting that
        // the values are initialized. This is safe because:
        // 1. We've asserted that T doesn't have a destructor, so no cleanup is needed,
        // 2. `MaybeUninit<T>` has the same memory layout as T,
        // 3. We maintain the same length.
        //
        // The returned slice takes over the lifetime of the input slice making the lifetime
        // correct.
        std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut MaybeUninit<T>, slice.len())
    }
}

/// This can be removed when MaybeUninit::slice_assume_init_mut is stabilized
/// <https://github.com/rust-lang/rust/issues/63569>
///
/// # Safety
///
/// It is up to the caller to guarantee that the `MaybeUninit<T>` elements
/// really are in an initialized state.
/// Calling this when the content is not yet fully initialized causes undefined behavior.
///
/// See [`assume_init_mut`] for more details and examples.
///
/// [`assume_init_mut`]: MaybeUninit::assume_init_mut
pub const unsafe fn slice_assume_init_mut<T>(slice: &mut [MaybeUninit<T>]) -> &mut [T] {
    unsafe { std::mem::transmute(slice) }
}

/// This can be removed when MaybeUninit::slice_assume_init_ref is stabilized
/// <https://github.com/rust-lang/rust/issues/63569>
///
/// # Safety
///
/// It is up to the caller to guarantee that the `MaybeUninit<T>` elements
/// really are in an initialized state.
/// Calling this when the content is not yet fully initialized causes undefined behavior.
///
/// See [`assume_init_ref`] for more details and examples.
///
/// [`assume_init_ref`]: MaybeUninit::assume_init_ref
pub const unsafe fn slice_assume_init_ref<T>(slice: &[MaybeUninit<T>]) -> &[T] {
    unsafe { std::mem::transmute(slice) }
}
