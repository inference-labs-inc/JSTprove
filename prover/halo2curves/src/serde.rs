use std::io::{self, Read, Write};

/// Trait for converting raw bytes to/from the internal representation of a type.
/// For example, field elements are represented in Montgomery form and serialized/deserialized without Montgomery reduction.
pub trait SerdeObject: Sized {
    /// The purpose of unchecked functions is to read the internal memory representation
    /// of a type from bytes as quickly as possible. No sanitization checks are performed
    /// to ensure the bytes represent a valid object. As such this function should only be
    /// used internally as an extension of machine memory. It should not be used to deserialize
    /// externally provided data.
    fn from_raw_bytes_unchecked(bytes: &[u8]) -> Self;
    fn from_raw_bytes(bytes: &[u8]) -> Option<Self>;

    fn to_raw_bytes(&self) -> Vec<u8>;

    /// The purpose of unchecked functions is to read the internal memory representation
    /// of a type from disk as quickly as possible. No sanitization checks are performed
    /// to ensure the bytes represent a valid object. This function should only be used
    /// internally when some machine state cannot be kept in memory (e.g., between runs)
    /// and needs to be reloaded as quickly as possible.
    fn read_raw_unchecked<R: Read>(reader: &mut R) -> Self;
    fn read_raw<R: Read>(reader: &mut R) -> io::Result<Self>;

    fn write_raw<W: Write>(&self, writer: &mut W) -> io::Result<()>;
}
