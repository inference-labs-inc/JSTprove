use std::ptr::copy_nonoverlapping;

/// Trait for types that can be serialized to and from shared memory
/// Compare to traditional deserialization, this trait deserializes
/// 'in place' as much as possible, without allocating new memory.
/// For example, a `Vec<T>` will only allocate memory for the pointer and length.
///
/// We assume all types have a minimum alignment of 8 bytes, which is the case for most types in
/// Rust. Be careful when using types with different alignments, as this may lead to undefined
/// behavior. For example, avx512 type `__m512i` is aligned to 64 bytes, and `__m256i` is aligned to
/// 32 bytes.
pub trait MPISharedMemory {
    /// The serialization size of the type in bytes.
    fn bytes_size(&self) -> usize;

    /// Write the value to the shared memory pointed to by `ptr`.
    /// The pointer is advanced by the size of the value.
    /// # Safety
    /// The caller must ensure that the pointer is valid and has enough space allocated.
    fn to_memory(&self, ptr: &mut *mut u8);

    /// Read the value from the shared memory pointed to by `ptr`.
    /// The pointer is advanced by the size of the value.
    ///
    /// Depending on the type, this may involve multiple objects across processes
    /// controlling the same memory. By default, only the root process will write to the memory,
    /// but this is not enforced by the trait.
    ///
    /// Q: Why not simply use Arc<Mutex<T>>?
    /// A: Because the memory is shared across processes, not threads.
    fn new_from_memory(ptr: &mut *mut u8) -> Self;

    /// Since the memory is shared across processes, the drop of the value
    /// may cause multiple deallocations of the same memory.
    /// This function is used to discard the control of the shared memory
    /// before the value is dropped.
    fn discard_control_of_shared_mem(self);
}

impl MPISharedMemory for usize {
    fn bytes_size(&self) -> usize {
        8
    }

    fn to_memory(&self, ptr: &mut *mut u8) {
        unsafe {
            (*ptr as *mut usize).write(*self);
            *ptr = ptr.add(8);
        }
    }

    fn new_from_memory(ptr: &mut *mut u8) -> Self {
        unsafe {
            let ret = (*ptr as *mut usize).read();
            *ptr = ptr.add(8);
            ret
        }
    }

    fn discard_control_of_shared_mem(self) {}
}

impl<T: Copy> MPISharedMemory for Vec<T> {
    fn bytes_size(&self) -> usize {
        let alignment = std::mem::align_of::<T>();
        std::cmp::max(self.len().bytes_size(), alignment) + self.len() * std::mem::size_of::<T>()
    }

    fn to_memory(&self, ptr: &mut *mut u8) {
        unsafe {
            let len = self.len();
            len.to_memory(ptr);
            align_ptr(ptr, std::mem::align_of::<T>());
            copy_nonoverlapping(self.as_ptr(), *ptr as *mut T, len);
            *ptr = ptr.add(len * std::mem::size_of::<T>());
        }
    }

    fn new_from_memory(ptr: &mut *mut u8) -> Self {
        unsafe {
            let len = usize::new_from_memory(ptr);
            align_ptr(ptr, std::mem::align_of::<T>());
            let ret = Vec::<T>::from_raw_parts(*ptr as *mut T, len, len);
            *ptr = ptr.add(len * std::mem::size_of::<T>());
            ret
        }
    }

    fn discard_control_of_shared_mem(self) {
        self.leak();
    }
}

impl<T1: MPISharedMemory, T2: MPISharedMemory> MPISharedMemory for (T1, T2) {
    fn bytes_size(&self) -> usize {
        self.0.bytes_size() + self.1.bytes_size()
    }

    fn to_memory(&self, ptr: &mut *mut u8) {
        self.0.to_memory(ptr);
        self.1.to_memory(ptr);
    }

    fn new_from_memory(ptr: &mut *mut u8) -> Self {
        let t1 = T1::new_from_memory(ptr);
        let t2 = T2::new_from_memory(ptr);
        (t1, t2)
    }

    fn discard_control_of_shared_mem(self) {
        self.0.discard_control_of_shared_mem();
        self.1.discard_control_of_shared_mem();
    }
}

pub fn align_ptr(ptr: &mut *mut u8, align: usize) {
    let addr = *ptr as usize;
    let aligned = (addr + align - 1) & !(align - 1);
    *ptr = aligned as *mut u8;
}
