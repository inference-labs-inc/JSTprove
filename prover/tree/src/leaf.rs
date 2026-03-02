use std::fmt;
use std::fmt::{Debug, Display};

use serdes::ExpSerde;
use tiny_keccak::{Hasher, Keccak};

use crate::Node;

/// Each leaf should have 64 bytes or 512 bits
pub const LEAF_BYTES: usize = 64;

/// Each leaf hash should have 32 bytes
pub const LEAF_HASH_BYTES: usize = 32;

/// Represents a leaf in the Merkle tree, containing 64 bytes of data stored in a BabyBearx16.
#[derive(Debug, Copy, Clone, PartialEq, Eq, ExpSerde)]
pub struct Leaf {
    pub data: [u8; LEAF_BYTES],
}

impl Default for Leaf {
    fn default() -> Self {
        Self {
            data: [0u8; LEAF_BYTES],
        }
    }
}

impl Display for Leaf {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Display the first and last byte of the leaf data for brevity
        write!(f, "leaf: 0x{:02x?}", self.data)
    }
}

impl Leaf {
    /// Creates a new Leaf with the given data.
    pub fn new(data: [u8; LEAF_BYTES]) -> Self {
        Self { data }
    }

    pub fn leaf_hash(&self) -> Node {
        let mut hasher = Keccak::v256();
        hasher.update(&self.data);
        let mut res = [0u8; LEAF_HASH_BYTES];

        hasher.finalize(&mut res);

        Node { data: res }
    }
}
