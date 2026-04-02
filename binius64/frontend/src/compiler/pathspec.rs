// Copyright 2025 Irreducible Inc.
use cranelift_entity::PrimaryMap;

/// A designator of a path within a circuit.
///
/// Compact, only 32-bit.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PathSpec(u32);
cranelift_entity::entity_impl!(PathSpec);

struct Node {
    name: String,
    parent: PathSpec,
}

/// A tree that holds paths within a circuit.
pub struct PathSpecTree {
    root: PathSpec,
    nodes: PrimaryMap<PathSpec, Node>,
}

impl PathSpecTree {
    /// Creates a new empty tree.
    pub fn new() -> Self {
        let mut nodes = PrimaryMap::new();
        let root = nodes.push(Node {
            name: String::new(),
            parent: PathSpec(0),
        });
        Self { root, nodes }
    }

    /// Extend the tree with a new branch that stems from the given `parent` and has a certain
    /// `name`.
    pub fn extend(&mut self, parent: PathSpec, name: impl Into<String>) -> PathSpec {
        self.nodes.push(Node {
            name: name.into(),
            parent,
        })
    }

    /// Writes a string representation of the given path spec into a given string buffer.
    ///
    /// Note that the string buffer is not reset, which allows appending to the existing contents
    /// of the string but poses a string of mangling of the string.
    pub fn stringify(&self, ls: PathSpec, out: &mut String) {
        fn stringify_rec(
            root: PathSpec,
            nodes: &PrimaryMap<PathSpec, Node>,
            ls: PathSpec,
            out: &mut String,
        ) {
            if ls == root {
                return;
            }
            stringify_rec(root, nodes, nodes[ls].parent, out);
            out.push('.');
            out.push_str(&nodes[ls].name);
        }
        stringify_rec(self.root, &self.nodes, ls, out);
    }

    /// Returns the parent of the given path or null if `root` was supplied.
    pub fn parent(&self, path: PathSpec) -> Option<PathSpec> {
        if path == self.root {
            return None;
        }
        Some(self.nodes[path].parent)
    }

    /// Returns the root of the tree.
    pub fn root(&self) -> PathSpec {
        self.root
    }
}

impl Default for PathSpecTree {
    fn default() -> Self {
        Self::new()
    }
}
