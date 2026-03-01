use std::{fmt, fmt::Display};

use serdes::ExpSerde;
use tiny_keccak::{Hasher, Keccak};

use crate::LEAF_HASH_BYTES;

/// A node in the Merkle tree, representing 32 bytes of data.
#[derive(Debug, Copy, Clone, PartialEq, Default, ExpSerde)]
pub struct Node {
    pub(crate) data: [u8; LEAF_HASH_BYTES],
}

impl Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Display the first and last byte of the node for brevity
        write!(f, "node: 0x{:02x?}...{:02x?}", self.data[0], self.data[31])
    }
}

impl Node {
    /// Creates a new Node with the given data.
    pub fn new(data: [u8; LEAF_HASH_BYTES]) -> Self {
        Self { data }
    }

    /// Computes the hash of two child nodes to create a parent node.
    ///
    /// This function uses Keccak for hashing and takes the first 32 bytes of the result.
    ///
    /// # Arguments
    ///
    /// * `left` - The left child node
    /// * `right` - The right child node
    ///
    /// # Returns
    ///
    /// A new Node containing the hash of the two input nodes.
    #[inline]
    pub fn node_hash(left: &Node, right: &Node) -> Node {
        let mut hasher = Keccak::v256();
        hasher.update(&left.data);
        hasher.update(&right.data);

        let mut result = [0u8; LEAF_HASH_BYTES];
        hasher.finalize(&mut result);

        Node { data: result }
    }

    /// Returns the data of the node as a slice of bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }
}
