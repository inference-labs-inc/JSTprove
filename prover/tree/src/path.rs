use std::fmt;
use std::fmt::{Debug, Display};

use ark_std::{end_timer, start_timer};
use serdes::ExpSerde;

use crate::{
    common_ancestor, convert_index_to_last_level, is_left_child, parent_index, Leaf, Node, Tree,
};

/// Represents a path in the Merkle tree, used for proving membership.
#[derive(Clone, Debug, PartialEq, Default, ExpSerde)]
pub struct Path {
    pub leaf: Leaf,
    pub(crate) path_nodes: Vec<Node>,
    pub index: usize,
}

impl Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "leaf index: {}", self.index)?;

        let position_list = self.position_list();
        for ((i, node), is_right_node) in self.path_nodes.iter().enumerate().zip(position_list) {
            writeln!(
                f,
                "{i}-th node, is right node {is_right_node}, sibling: {node}",
            )?;
        }

        Ok(())
    }
}

impl Path {
    /// Computes the position of on-path nodes in the Merkle tree.
    ///
    /// This function converts the leaf index to a boolean array in big-endian form,
    /// where `true` indicates a right child and `false` indicates a left child.
    #[inline]
    fn position_list(&'_ self) -> impl '_ + Iterator<Item = bool> {
        (0..self.path_nodes.len() + 1).map(move |i| ((self.index >> i) & 1) != 0)
    }

    /// Verifies the path against a given root and leaf.
    ///
    /// # Arguments
    ///
    /// * `root` - The root node of the Merkle tree
    /// * `leaf` - The leaf node to verify
    /// * `hasher` - The Poseidon hash parameters
    ///
    /// # Returns
    ///
    /// `true` if the path is valid, `false` otherwise.
    #[inline]
    pub fn verify(&self, root: &Node) -> bool {
        let timer = start_timer!(|| "path verify");

        let position_list = self.position_list().collect::<Vec<_>>();
        // let leaf_node = self.leaf.leaf_hash(hasher);
        let leaf_node = self.leaf.leaf_hash();
        let mut current_node = leaf_node;

        // Traverse the path from leaf to root
        for (i, node) in self.path_nodes.iter().rev().enumerate() {
            if position_list[i] {
                current_node = Node::node_hash(node, &current_node)
            } else {
                current_node = Node::node_hash(&current_node, node)
            };
        }

        end_timer!(timer);
        if current_node != *root {
            println!("path does not match the root");
            false
        } else {
            true
        }
    }

    #[inline]
    pub fn root(&self) -> Node {
        let position_list = self.position_list().collect::<Vec<_>>();
        // let leaf_node = self.leaf.leaf_hash(hasher);
        let leaf_node = self.leaf.leaf_hash();
        let mut current_node = leaf_node;

        // Traverse the path from leaf to root
        for (i, node) in self.path_nodes.iter().rev().enumerate() {
            if position_list[i] {
                current_node = Node::node_hash(node, &current_node)
            } else {
                current_node = Node::node_hash(&current_node, node)
            };
        }

        current_node
    }

    /// Return the leaf of the path
    #[inline]
    pub fn leaf(&self) -> &Leaf {
        &self.leaf
    }
}

#[derive(Clone, Debug, PartialEq, Default, ExpSerde)]
pub struct RangePath {
    pub leaves: Vec<Leaf>,
    pub(crate) path_nodes: Vec<Node>,
    pub left: usize,
    pub right: usize,
}

impl Display for RangePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "leaf index range: [{}, {}]", self.left, self.right)?;

        let position_list = self.position_list();
        for ((i, node), is_right_node) in self.path_nodes.iter().enumerate().zip(position_list) {
            writeln!(
                f,
                "{i}-th node, is right node {is_right_node}, sibling: {node}"
            )?;
        }

        Ok(())
    }
}

impl RangePath {
    #[inline]
    fn position_list(&'_ self) -> impl '_ + Iterator<Item = bool> {
        let common_ancestor = common_ancestor(self.left, self.right);

        (0..self.path_nodes.len() + 1).map(move |i| ((common_ancestor >> i) & 1) != 0)
    }

    #[inline]
    pub fn root(&self) -> Node {
        let sub_tree = Tree::new_with_leaves(self.leaves.clone());

        let tree_height = sub_tree.height() + self.path_nodes.len();
        let mut current_node = sub_tree.root();

        drop(sub_tree);

        let left_index_in_tree = convert_index_to_last_level(self.left, tree_height);
        let right_index_in_tree = convert_index_to_last_level(self.right, tree_height);

        let mut current_node_index = common_ancestor(left_index_in_tree, right_index_in_tree);

        self.path_nodes.iter().rev().for_each(|node| {
            if is_left_child(current_node_index) {
                current_node = Node::node_hash(&current_node, node)
            } else {
                current_node = Node::node_hash(node, &current_node)
            }

            current_node_index = parent_index(current_node_index).unwrap();
        });

        current_node
    }

    #[inline]
    pub fn verify(&self, root: &Node) -> bool {
        self.root() == *root
    }
}
