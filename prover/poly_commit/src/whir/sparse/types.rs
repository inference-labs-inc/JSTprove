//! Data structures for the sparse-MLE WHIR commitment.
//!
//! The naming here mirrors Spartan §7.2 wherever possible:
//!
//! * `row` is the address vector for the z-axis (output index)
//! * `col_x` and `col_y` are the address vectors for the two input axes
//! * `val` is the gate coefficient at each non-zero position
//! * `nnz` is the number of non-zero entries in the dense MLE
//!
//! For 2-arity gates (e.g. `add(z, x)` with one input axis) we set
//! `col_y` to the all-zero vector and `n_y = 0`; the construction
//! degenerates cleanly to the 2-axis Spartan case.

use arith::{ExtensionField, Field};
use serdes::{ExpSerde, SerdeError, SerdeResult};
use tree::Node;

use super::memcheck::AddrTimestamps;

/// Maximum number of non-zero gate triples we will accept inside a
/// single sparse polynomial. The bound is generous (≥ 1B entries) and
/// only exists to bound deserialization work; bundles built by the
/// honest setup phase carry far smaller polynomials.
pub const SPARSE_MLE_MAX_NNZ: usize = 1usize << 30;

/// Maximum log-domain length for any single address axis.
///
/// Capped at 32 so cell addresses fit in `u32`. This lets the
/// generic memory-checking code use the `Field: From<u32>` lift in
/// `arith::Field`'s trait bounds without needing a wider conversion.
/// 2^32 cells per axis is far above any feasible layer size — the
/// largest production layers we currently see are on the order of
/// 2^25 gates.
pub const SPARSE_MLE_MAX_LOG_DOMAIN: usize = 32;

/// Number of input axes a sparse wiring polynomial spans.
///
/// `Two` is the standard `add(z, x)` selector with one input wire and
/// `Three` is the `mul(z, x, y)` selector with two input wires. The
/// arity is recorded explicitly in the commitment so the verifier can
/// route to the right number of memory-checking sumchecks without
/// having to inspect the polynomial.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SparseArity {
    /// Two-axis (`add(z, x)`-style) wiring selector. Default since
    /// it is the simpler protocol case and only needs the y-axis
    /// machinery to be opted into.
    #[default]
    Two,
    /// Three-axis (`mul(z, x, y)`-style) wiring selector.
    Three,
}

impl SparseArity {
    /// Number of address axes (i.e. number of memory-checking tracks
    /// the open / verify routines must drive).
    #[must_use]
    pub const fn num_input_axes(self) -> usize {
        match self {
            Self::Two => 1,
            Self::Three => 2,
        }
    }
}

impl ExpSerde for SparseArity {
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        let tag: u8 = match self {
            Self::Two => 0,
            Self::Three => 1,
        };
        tag.serialize_into(&mut writer)
    }

    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        let tag = u8::deserialize_from(&mut reader)?;
        match tag {
            0 => Ok(Self::Two),
            1 => Ok(Self::Three),
            _ => Err(SerdeError::DeserializeError),
        }
    }
}

/// Sparse representation of a 3-arity multilinear polynomial.
///
/// `M(z, x, y) = Σ_k val[k] · ẽq(row[k], z) · ẽq(col_x[k], x) · ẽq(col_y[k], y)`
///
/// The `row`, `col_x`, `col_y` vectors are little-endian indices into
/// the hypercubes `{0,1}^n_z`, `{0,1}^n_x`, `{0,1}^n_y` respectively.
/// `val[k]` lives in the base field.
#[derive(Debug, Clone)]
pub struct SparseMle3<F: Field> {
    pub n_z: usize,
    pub n_x: usize,
    pub n_y: usize,
    pub arity: SparseArity,
    pub row: Vec<usize>,
    pub col_x: Vec<usize>,
    pub col_y: Vec<usize>,
    pub val: Vec<F>,
}

impl<F: Field> SparseMle3<F> {
    /// Number of non-zero entries.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.val.len()
    }

    /// Total variable count of the dense MLE this sparse rep encodes.
    #[must_use]
    pub fn total_vars(&self) -> usize {
        self.n_z + self.n_x + self.n_y
    }

    /// Validate the structural invariants of the sparse representation.
    ///
    /// # Errors
    /// Returns [`SparseMleError::DimensionMismatch`] if the four
    /// constituent vectors have inconsistent lengths,
    /// [`SparseMleError::AddressOutOfRange`] if any address exceeds
    /// the declared axis size, [`SparseMleError::NnzExceeded`] if
    /// `nnz` exceeds [`SPARSE_MLE_MAX_NNZ`], and
    /// [`SparseMleError::DomainTooLarge`] if any axis is wider than
    /// [`SPARSE_MLE_MAX_LOG_DOMAIN`].
    pub fn validate(&self) -> Result<(), SparseMleError> {
        if self.row.len() != self.val.len()
            || self.col_x.len() != self.val.len()
            || self.col_y.len() != self.val.len()
        {
            return Err(SparseMleError::DimensionMismatch {
                row: self.row.len(),
                col_x: self.col_x.len(),
                col_y: self.col_y.len(),
                val: self.val.len(),
            });
        }
        if self.val.len() > SPARSE_MLE_MAX_NNZ {
            return Err(SparseMleError::NnzExceeded(self.val.len()));
        }
        if self.n_z > SPARSE_MLE_MAX_LOG_DOMAIN
            || self.n_x > SPARSE_MLE_MAX_LOG_DOMAIN
            || self.n_y > SPARSE_MLE_MAX_LOG_DOMAIN
        {
            return Err(SparseMleError::DomainTooLarge {
                n_z: self.n_z,
                n_x: self.n_x,
                n_y: self.n_y,
            });
        }
        let m_z = 1usize.checked_shl(self.n_z as u32).unwrap_or(0);
        let m_x = 1usize.checked_shl(self.n_x as u32).unwrap_or(0);
        let m_y = if self.arity == SparseArity::Two {
            1
        } else {
            1usize.checked_shl(self.n_y as u32).unwrap_or(0)
        };
        for k in 0..self.val.len() {
            if self.row[k] >= m_z {
                return Err(SparseMleError::AddressOutOfRange {
                    axis: AddressAxis::Z,
                    index: k,
                    addr: self.row[k],
                    bound: m_z,
                });
            }
            if self.col_x[k] >= m_x {
                return Err(SparseMleError::AddressOutOfRange {
                    axis: AddressAxis::X,
                    index: k,
                    addr: self.col_x[k],
                    bound: m_x,
                });
            }
            if self.arity == SparseArity::Two {
                if self.col_y[k] != 0 {
                    return Err(SparseMleError::AddressOutOfRange {
                        axis: AddressAxis::Y,
                        index: k,
                        addr: self.col_y[k],
                        bound: 1,
                    });
                }
            } else if self.col_y[k] >= m_y {
                return Err(SparseMleError::AddressOutOfRange {
                    axis: AddressAxis::Y,
                    index: k,
                    addr: self.col_y[k],
                    bound: m_y,
                });
            }
        }
        Ok(())
    }

    /// Naive evaluation of `M(z, x, y)` from the sparse representation.
    ///
    /// Used by tests as the soundness oracle. Cost is `O(nnz · log m)`
    /// per call where `m` is the largest axis size, so it must not be
    /// used inside the protocol's hot path.
    #[must_use]
    pub fn evaluate<E: ExtensionField<BaseField = F>>(&self, z: &[E], x: &[E], y: &[E]) -> E {
        assert_eq!(z.len(), self.n_z, "z length mismatch");
        assert_eq!(x.len(), self.n_x, "x length mismatch");
        if self.arity == SparseArity::Three {
            assert_eq!(y.len(), self.n_y, "y length mismatch");
        }
        let mut acc = E::ZERO;
        for k in 0..self.val.len() {
            let e_z = eval_eq_at_index(z, self.row[k]);
            let e_x = eval_eq_at_index(x, self.col_x[k]);
            let e_y = if self.arity == SparseArity::Two {
                E::ONE
            } else {
                eval_eq_at_index(y, self.col_y[k])
            };
            acc += e_z * e_x * e_y * E::from(self.val[k]);
        }
        acc
    }
}

/// Address axis identifier used in error reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressAxis {
    Z,
    X,
    Y,
}

/// Errors raised by sparse-MLE construction and validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseMleError {
    DimensionMismatch {
        row: usize,
        col_x: usize,
        col_y: usize,
        val: usize,
    },
    AddressOutOfRange {
        axis: AddressAxis,
        index: usize,
        addr: usize,
        bound: usize,
    },
    NnzExceeded(usize),
    DomainTooLarge {
        n_z: usize,
        n_x: usize,
        n_y: usize,
    },
}

impl std::fmt::Display for SparseMleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch {
                row,
                col_x,
                col_y,
                val,
            } => write!(
                f,
                "sparse MLE dimension mismatch: row={row}, col_x={col_x}, col_y={col_y}, val={val}"
            ),
            Self::AddressOutOfRange {
                axis,
                index,
                addr,
                bound,
            } => write!(
                f,
                "sparse MLE address out of range on axis {axis:?}: entry {index} addr={addr} >= bound {bound}"
            ),
            Self::NnzExceeded(n) => write!(
                f,
                "sparse MLE nnz {n} exceeds maximum {SPARSE_MLE_MAX_NNZ}"
            ),
            Self::DomainTooLarge { n_z, n_x, n_y } => write!(
                f,
                "sparse MLE log-domain too large: n_z={n_z}, n_x={n_x}, n_y={n_y}, max {SPARSE_MLE_MAX_LOG_DOMAIN}"
            ),
        }
    }
}

impl std::error::Error for SparseMleError {}

/// Public commitment to a sparse multilinear polynomial.
///
/// Following Spartan §7.2.3 optimization (5), the constituent vectors
/// (`row`, `col_x`, `col_y`, `val`) and the memory-checking timestamps
/// are concatenated into a single dense polynomial whose multilinear
/// extension is committed with one WHIR call. The Merkle root of that
/// commitment is `batched_root`; the `aux` fields record the segment
/// dimensions the verifier needs to address into the batched layout.
#[derive(Debug, Clone)]
pub struct SparseMle3Commitment {
    pub arity: SparseArity,
    pub n_z: usize,
    pub n_x: usize,
    pub n_y: usize,
    pub nnz: usize,
    pub log_nnz: usize,
    pub batched_root: Node,
    /// Total number of variables in the batched dense polynomial that
    /// `batched_root` commits to. Verifier uses this to bound the WHIR
    /// challenge length.
    pub batched_num_vars: usize,
}

impl ExpSerde for SparseMle3Commitment {
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        self.arity.serialize_into(&mut writer)?;
        (self.n_z as u64).serialize_into(&mut writer)?;
        (self.n_x as u64).serialize_into(&mut writer)?;
        (self.n_y as u64).serialize_into(&mut writer)?;
        (self.nnz as u64).serialize_into(&mut writer)?;
        (self.log_nnz as u64).serialize_into(&mut writer)?;
        (self.batched_num_vars as u64).serialize_into(&mut writer)?;
        self.batched_root.serialize_into(&mut writer)?;
        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        let arity = SparseArity::deserialize_from(&mut reader)?;
        let n_z = u64::deserialize_from(&mut reader)? as usize;
        let n_x = u64::deserialize_from(&mut reader)? as usize;
        let n_y = u64::deserialize_from(&mut reader)? as usize;
        let nnz = u64::deserialize_from(&mut reader)? as usize;
        let log_nnz = u64::deserialize_from(&mut reader)? as usize;
        let batched_num_vars = u64::deserialize_from(&mut reader)? as usize;
        let batched_root = Node::deserialize_from(&mut reader)?;
        if n_z > SPARSE_MLE_MAX_LOG_DOMAIN
            || n_x > SPARSE_MLE_MAX_LOG_DOMAIN
            || n_y > SPARSE_MLE_MAX_LOG_DOMAIN
            || nnz > SPARSE_MLE_MAX_NNZ
            || log_nnz > SPARSE_MLE_MAX_LOG_DOMAIN
        {
            return Err(SerdeError::DeserializeError);
        }
        // Structural invariants: nnz must be a power of two,
        // log_nnz must match, and arity Two requires n_y == 0.
        if nnz != 0 && (nnz & (nnz - 1)) != 0 {
            return Err(SerdeError::DeserializeError);
        }
        if nnz != 0 && log_nnz != nnz.trailing_zeros() as usize {
            return Err(SerdeError::DeserializeError);
        }
        if arity == SparseArity::Two && n_y != 0 {
            return Err(SerdeError::DeserializeError);
        }
        Ok(Self {
            arity,
            n_z,
            n_x,
            n_y,
            nnz,
            log_nnz,
            batched_root,
            batched_num_vars,
        })
    }
}

/// Witness data the prover keeps after committing.
///
/// Holds the dense layout that backs `SparseMle3Commitment.batched_root`
/// together with the per-axis timestamps that the open routine needs
/// to drive the offline-memory-checking sumcheck. Erased into the
/// generic [`crate::whir::WhirScratchPad`] for caching across calls.
#[derive(Debug, Clone)]
pub struct SparseMleScratchPad<F: Field> {
    pub arity: SparseArity,
    pub n_z: usize,
    pub n_x: usize,
    pub n_y: usize,
    pub nnz: usize,
    pub log_nnz: usize,
    pub row: Vec<usize>,
    pub col_x: Vec<usize>,
    pub col_y: Vec<usize>,
    pub val: Vec<F>,
    pub ts_z: AddrTimestamps<F>,
    pub ts_x: AddrTimestamps<F>,
    pub ts_y: Option<AddrTimestamps<F>>,
}

/// Opening proof for a sparse-MLE WHIR commitment.
///
/// Concretely contains:
/// * the eval-claim sumcheck transcript reducing
///   `v = Σ_k val[k]·e_z[k]·e_x[k]·e_y[k]` to a final claim about
///   four / three constituent evaluations at a random point in `{0,1}^log_nnz`,
/// * a grand-product / multiset-equality argument transcript per
///   address axis (one for `z`, `x`, and — if arity = 3 — `y`),
/// * the WHIR opening of the batched constituent polynomial at the
///   sumcheck random point.
///
/// The concrete inner types are introduced in the open / verify
/// modules; this struct is intentionally a thin opaque container at
/// the moment so callers can plumb it through bundle code without a
/// dependency on the (still-evolving) protocol layout.
#[derive(Debug, Clone, Default)]
pub struct SparseMle3Opening<F: ExtensionField> {
    pub claimed_eval: F,
    /// Inner protocol bytes; the layout is documented in `open.rs`
    /// and consumed by `verify.rs`. Held as bytes here so the type
    /// stays serdes-stable across protocol revisions.
    pub protocol_bytes: Vec<u8>,
}

impl<F: ExtensionField> ExpSerde for SparseMle3Opening<F> {
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        self.claimed_eval.serialize_into(&mut writer)?;
        (self.protocol_bytes.len() as u64).serialize_into(&mut writer)?;
        writer
            .write_all(&self.protocol_bytes)
            .map_err(|_| SerdeError::DeserializeError)?;
        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        // Cap on protocol-bytes payload to bound deserialization work.
        // 256 MiB is well above any honest opening for the layer sizes
        // we expect; the cap exists purely to reject pathological inputs.
        const MAX_PROTOCOL_BYTES: usize = 1 << 28;
        let claimed_eval = F::deserialize_from(&mut reader)?;
        let len = u64::deserialize_from(&mut reader)? as usize;
        if len > MAX_PROTOCOL_BYTES {
            return Err(SerdeError::DeserializeError);
        }
        let mut protocol_bytes = vec![0u8; len];
        reader
            .read_exact(&mut protocol_bytes)
            .map_err(|_| SerdeError::DeserializeError)?;
        Ok(Self {
            claimed_eval,
            protocol_bytes,
        })
    }
}

/// Evaluate `ẽq(addr_bits, r)` where `addr_bits` is the binary
/// expansion of `addr` in `r.len()` little-endian variables. This is
/// the closed-form `Π_i ((1-r_i) + r_i·b_i + ... )` written without
/// allocating an intermediate eq table.
///
/// Pre-condition: `addr < 2^r.len()`.
#[inline]
pub(crate) fn eval_eq_at_index<E: Field>(r: &[E], addr: usize) -> E {
    let mut acc = E::ONE;
    for (i, ri) in r.iter().enumerate() {
        let bit = (addr >> i) & 1;
        if bit == 1 {
            acc *= *ri;
        } else {
            acc *= E::ONE - *ri;
        }
    }
    acc
}
