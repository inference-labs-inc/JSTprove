use serde::{Deserialize, Serialize};

use crate::runner::errors::RunError;

/// Base field of a compiled circuit. This is a strict subset of the
/// information required to verify a proof — it identifies the modulus
/// the circuit is built over but says nothing about the polynomial
/// commitment scheme, extension degree, or transcript hash.
///
/// Used as the return type of [`Field::detect_from_circuit_bytes`],
/// which can recover the base field from a serialized circuit's
/// embedded modulus but cannot determine the rest of the proof
/// configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Field {
    Bn254,
    Goldilocks,
}

/// Complete proof system configuration for a compiled circuit.
///
/// Each variant corresponds 1:1 with an [`expander_compiler::frontend::Config`]
/// implementation in the underlying prover and carries the same numeric
/// `CONFIG_ID`. Variants encode the full
/// (field, extension degree, transcript, PCS) tuple — there is no
/// implicit composition. Adding a new prover config requires adding a
/// new variant here and a new arm everywhere `ProofConfig` is matched,
/// keeping the routing tables exhaustively typed.
///
/// Versioning is per-variant: see [`ProofConfig::current_version`]. The
/// version stamped at compile time is recorded in the bundle manifest;
/// loaders refuse to operate on bundles whose stamped version does not
/// match the current spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProofConfig {
    /// BN254 base field with the SHA2 transcript and Raw PCS.
    /// Maps to `BN254Config = BN254ConfigSha2Raw` (`CONFIG_ID = 2`).
    Bn254Raw,
    /// Goldilocks base field, x1 SIMD lane, SHA2 transcript, Raw PCS.
    /// Maps to `GoldilocksConfig = Goldilocksx1ConfigSha2Raw`
    /// (`CONFIG_ID = 4`).
    GoldilocksRaw,
    /// Goldilocks base field with the Basefold PCS.
    /// Maps to `GoldilocksBasefoldConfig = Goldilocksx1ConfigSha2Basefold`
    /// (`CONFIG_ID = 8`).
    GoldilocksBasefold,
    /// Goldilocks degree-2 extension with the Basefold PCS.
    /// Maps to `GoldilocksExt2BasefoldConfig = GoldilocksExt2x1ConfigSha2Basefold`
    /// (`CONFIG_ID = 9`).
    GoldilocksExt2Basefold,
    /// Goldilocks degree-3 extension with the WHIR PCS.
    /// Maps to `GoldilocksWhirConfig = GoldilocksExt3x1ConfigSha2Whir`
    /// (`CONFIG_ID = 10`).
    GoldilocksExt3Whir,
    /// Goldilocks degree-4 extension with the WHIR PCS, providing the
    /// post-quantum security margin used by the PQ profile.
    /// Maps to `GoldilocksWhirPQConfig = GoldilocksExt4x1ConfigSha2Whir`
    /// (`CONFIG_ID = 11`).
    GoldilocksExt4Whir,
}

/// Errors that can occur when constructing or interpreting a
/// [`ProofConfig`].
#[derive(Debug, Clone)]
pub enum ProofConfigError {
    UnknownConfigId(usize),
    UnknownName(String),
    VersionMismatch {
        config: ProofConfig,
        stored: u32,
        current: u32,
    },
}

impl std::fmt::Display for ProofConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownConfigId(id) => write!(f, "unknown proof config id: {id}"),
            Self::UnknownName(s) => write!(f, "unknown proof config: {s}"),
            Self::VersionMismatch {
                config,
                stored,
                current,
            } => write!(
                f,
                "{config} version mismatch: bundle was compiled with v{stored} but current code supports v{current}"
            ),
        }
    }
}

impl std::error::Error for ProofConfigError {}

impl ProofConfig {
    /// Stable numeric identifier matching the underlying prover
    /// config's `CONFIG_ID`. Embedded in serialized bundles to permit
    /// forward-compatible recognition without depending on the variant
    /// name.
    #[must_use]
    pub const fn config_id(self) -> usize {
        match self {
            Self::Bn254Raw => 2,
            Self::GoldilocksRaw => 4,
            Self::GoldilocksBasefold => 8,
            Self::GoldilocksExt2Basefold => 9,
            Self::GoldilocksExt3Whir => 10,
            Self::GoldilocksExt4Whir => 11,
        }
    }

    /// Look up a `ProofConfig` from its numeric identifier. Used when
    /// reading bundles produced by older or third-party tooling.
    ///
    /// # Errors
    /// Returns [`ProofConfigError::UnknownConfigId`] if the id does not
    /// match any known variant.
    pub fn from_config_id(id: usize) -> Result<Self, ProofConfigError> {
        match id {
            2 => Ok(Self::Bn254Raw),
            4 => Ok(Self::GoldilocksRaw),
            8 => Ok(Self::GoldilocksBasefold),
            9 => Ok(Self::GoldilocksExt2Basefold),
            10 => Ok(Self::GoldilocksExt3Whir),
            11 => Ok(Self::GoldilocksExt4Whir),
            _ => Err(ProofConfigError::UnknownConfigId(id)),
        }
    }

    /// Current spec version for this proof config. Bumped when the
    /// underlying prover changes the wire format, transcript domain
    /// separators, or proof structure for a given variant in a way
    /// that breaks compatibility with previously compiled bundles.
    ///
    /// Each variant is enumerated explicitly so that adding a new
    /// variant forces a deliberate per-variant version assignment.
    /// All current variants start at version 1.
    #[must_use]
    #[allow(clippy::match_same_arms)]
    pub const fn current_version(self) -> u32 {
        match self {
            Self::Bn254Raw => 1,
            Self::GoldilocksRaw => 1,
            Self::GoldilocksBasefold => 1,
            Self::GoldilocksExt2Basefold => 1,
            Self::GoldilocksExt3Whir => 1,
            Self::GoldilocksExt4Whir => 1,
        }
    }

    /// Base field of this proof config. Useful for callers that want
    /// to verify a detected field matches the expected proof config.
    #[must_use]
    pub const fn field(self) -> Field {
        match self {
            Self::Bn254Raw => Field::Bn254,
            Self::GoldilocksRaw
            | Self::GoldilocksBasefold
            | Self::GoldilocksExt2Basefold
            | Self::GoldilocksExt3Whir
            | Self::GoldilocksExt4Whir => Field::Goldilocks,
        }
    }
}

impl Default for ProofConfig {
    fn default() -> Self {
        Self::Bn254Raw
    }
}

impl std::fmt::Display for ProofConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Bn254Raw => "bn254_raw",
            Self::GoldilocksRaw => "goldilocks_raw",
            Self::GoldilocksBasefold => "goldilocks_basefold",
            Self::GoldilocksExt2Basefold => "goldilocks_ext2_basefold",
            Self::GoldilocksExt3Whir => "goldilocks_ext3_whir",
            Self::GoldilocksExt4Whir => "goldilocks_ext4_whir",
        };
        write!(f, "{name}")
    }
}

impl std::str::FromStr for ProofConfig {
    type Err = ProofConfigError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "bn254_raw" | "bn254" => Ok(Self::Bn254Raw),
            "goldilocks_raw" => Ok(Self::GoldilocksRaw),
            "goldilocks_basefold" | "goldilocks" => Ok(Self::GoldilocksBasefold),
            "goldilocks_ext2_basefold" | "goldilocks_ext2" => Ok(Self::GoldilocksExt2Basefold),
            "goldilocks_ext3_whir" | "goldilocks_whir" => Ok(Self::GoldilocksExt3Whir),
            "goldilocks_ext4_whir" | "goldilocks_whir_pq" => Ok(Self::GoldilocksExt4Whir),
            other => Err(ProofConfigError::UnknownName(other.to_string())),
        }
    }
}

/// A `ProofConfig` paired with the version that was current at compile
/// time. Stored in the bundle manifest so loaders can detect bundles
/// produced by an incompatible prover release.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StampedProofConfig {
    pub config: ProofConfig,
    pub version: u32,
}

impl StampedProofConfig {
    /// Stamp a config with the current version of the local prover.
    #[must_use]
    pub fn current(config: ProofConfig) -> Self {
        Self {
            config,
            version: config.current_version(),
        }
    }

    /// Verify that the stamped version matches what the current code
    /// can handle.
    ///
    /// # Errors
    /// Returns [`ProofConfigError::VersionMismatch`] if the stamped
    /// version differs from the current spec version.
    pub fn ensure_current(&self) -> Result<(), ProofConfigError> {
        let current = self.config.current_version();
        if self.version == current {
            Ok(())
        } else {
            Err(ProofConfigError::VersionMismatch {
                config: self.config,
                stored: self.version,
                current,
            })
        }
    }
}

impl Field {
    /// Detect the base field of a serialized circuit by inspecting the
    /// embedded modulus header. The detected field is unambiguous —
    /// every supported field has a unique modulus — but it does not
    /// determine the proof config (PCS, extension degree, etc.).
    /// Callers that need the full proof config must read it from the
    /// bundle manifest.
    ///
    /// # Errors
    /// Returns [`RunError::Deserialize`] if the bytes cannot be
    /// decompressed, are too short to contain a header, or encode an
    /// unknown field.
    pub fn detect_from_circuit_bytes(circuit_bytes: &[u8]) -> Result<Self, RunError> {
        use expander_compiler::frontend::{
            BN254Config, CircuitField, FieldArith, GoldilocksConfig,
        };
        use expander_compiler::serdes::ExpSerde;
        use jstprove_io::auto_decompress_bytes;
        use std::io::Cursor;
        use std::mem::size_of;

        const MAGIC_SIZE: usize = size_of::<usize>();
        // U256 serializes as 4 little-endian u64 limbs.
        const MODULUS_SIZE: usize = 4 * size_of::<u64>();
        const HEADER_SIZE: usize = MAGIC_SIZE + MODULUS_SIZE;

        let data = auto_decompress_bytes(circuit_bytes)
            .map_err(|e| RunError::Deserialize(format!("circuit decompress: {e:?}")))?;

        if data.len() < HEADER_SIZE {
            return Err(RunError::Deserialize(format!(
                "circuit bytes too short for header: {} < {HEADER_SIZE}",
                data.len()
            )));
        }

        let mut cursor = Cursor::new(&data[MAGIC_SIZE..HEADER_SIZE]);
        let modulus = ethnum::U256::deserialize_from(&mut cursor)
            .map_err(|e| RunError::Deserialize(format!("modulus: {e:?}")))?;

        let bn254_modulus = <CircuitField<BN254Config> as FieldArith>::MODULUS;
        let goldilocks_modulus = <CircuitField<GoldilocksConfig> as FieldArith>::MODULUS;

        if modulus == bn254_modulus {
            Ok(Self::Bn254)
        } else if modulus == goldilocks_modulus {
            Ok(Self::Goldilocks)
        } else {
            Err(RunError::Deserialize(format!(
                "unknown circuit field modulus: {modulus:?}"
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_id_round_trip() {
        for variant in [
            ProofConfig::Bn254Raw,
            ProofConfig::GoldilocksRaw,
            ProofConfig::GoldilocksBasefold,
            ProofConfig::GoldilocksExt2Basefold,
            ProofConfig::GoldilocksExt3Whir,
            ProofConfig::GoldilocksExt4Whir,
        ] {
            let id = variant.config_id();
            let parsed = ProofConfig::from_config_id(id).unwrap();
            assert_eq!(parsed, variant);
        }
    }

    #[test]
    fn from_unknown_config_id_errors() {
        assert!(matches!(
            ProofConfig::from_config_id(999),
            Err(ProofConfigError::UnknownConfigId(999))
        ));
    }

    #[test]
    fn display_parse_round_trip() {
        for variant in [
            ProofConfig::Bn254Raw,
            ProofConfig::GoldilocksRaw,
            ProofConfig::GoldilocksBasefold,
            ProofConfig::GoldilocksExt2Basefold,
            ProofConfig::GoldilocksExt3Whir,
            ProofConfig::GoldilocksExt4Whir,
        ] {
            let s = variant.to_string();
            let parsed: ProofConfig = s.parse().unwrap();
            assert_eq!(parsed, variant);
        }
    }

    #[test]
    fn parses_legacy_aliases() {
        assert_eq!(
            "bn254".parse::<ProofConfig>().unwrap(),
            ProofConfig::Bn254Raw
        );
        assert_eq!(
            "goldilocks".parse::<ProofConfig>().unwrap(),
            ProofConfig::GoldilocksBasefold
        );
        assert_eq!(
            "goldilocks_whir".parse::<ProofConfig>().unwrap(),
            ProofConfig::GoldilocksExt3Whir
        );
        assert_eq!(
            "goldilocks_whir_pq".parse::<ProofConfig>().unwrap(),
            ProofConfig::GoldilocksExt4Whir
        );
    }

    #[test]
    fn parse_unknown_errors() {
        assert!("nonsense".parse::<ProofConfig>().is_err());
    }

    #[test]
    fn field_classification() {
        assert_eq!(ProofConfig::Bn254Raw.field(), Field::Bn254);
        for variant in [
            ProofConfig::GoldilocksRaw,
            ProofConfig::GoldilocksBasefold,
            ProofConfig::GoldilocksExt2Basefold,
            ProofConfig::GoldilocksExt3Whir,
            ProofConfig::GoldilocksExt4Whir,
        ] {
            assert_eq!(variant.field(), Field::Goldilocks);
        }
    }

    #[test]
    fn current_version_starts_at_one() {
        for variant in [
            ProofConfig::Bn254Raw,
            ProofConfig::GoldilocksRaw,
            ProofConfig::GoldilocksBasefold,
            ProofConfig::GoldilocksExt2Basefold,
            ProofConfig::GoldilocksExt3Whir,
            ProofConfig::GoldilocksExt4Whir,
        ] {
            assert_eq!(variant.current_version(), 1);
        }
    }

    #[test]
    fn stamped_current_validates() {
        let stamped = StampedProofConfig::current(ProofConfig::GoldilocksExt4Whir);
        assert_eq!(stamped.config, ProofConfig::GoldilocksExt4Whir);
        assert_eq!(stamped.version, 1);
        stamped.ensure_current().unwrap();
    }

    #[test]
    fn stamped_version_mismatch_errors() {
        let stamped = StampedProofConfig {
            config: ProofConfig::GoldilocksExt4Whir,
            version: 999,
        };
        assert!(matches!(
            stamped.ensure_current(),
            Err(ProofConfigError::VersionMismatch {
                stored: 999,
                current: 1,
                ..
            })
        ));
    }

    mod field_detection {
        use super::*;
        use expander_compiler::frontend::{
            BN254Config, CircuitField, FieldArith, GoldilocksConfig,
        };
        use expander_compiler::serdes::ExpSerde;

        fn header_with_modulus(modulus: ethnum::U256) -> Vec<u8> {
            let mut buf = vec![0u8; 8];
            modulus.serialize_into(&mut buf).unwrap();
            buf
        }

        #[test]
        fn detects_bn254() {
            let bytes = header_with_modulus(<CircuitField<BN254Config> as FieldArith>::MODULUS);
            assert_eq!(
                Field::detect_from_circuit_bytes(&bytes).unwrap(),
                Field::Bn254
            );
        }

        #[test]
        fn detects_goldilocks() {
            let bytes =
                header_with_modulus(<CircuitField<GoldilocksConfig> as FieldArith>::MODULUS);
            assert_eq!(
                Field::detect_from_circuit_bytes(&bytes).unwrap(),
                Field::Goldilocks
            );
        }

        #[test]
        fn rejects_short_input() {
            let err = Field::detect_from_circuit_bytes(&[0u8; 10]).unwrap_err();
            assert!(matches!(err, RunError::Deserialize(_)));
        }

        #[test]
        fn rejects_unknown_modulus() {
            let bogus = ethnum::U256::from_words(0xdead_beef_u128, 0xcafe_babe_u128);
            let bytes = header_with_modulus(bogus);
            let err = Field::detect_from_circuit_bytes(&bytes).unwrap_err();
            match err {
                RunError::Deserialize(msg) => {
                    assert!(msg.contains("unknown circuit field modulus"), "{msg}")
                }
                other => panic!("expected Deserialize, got {other:?}"),
            }
        }

        #[test]
        fn rejects_corrupt_compression() {
            // zstd magic followed by truncated payload
            let bytes = vec![0x28, 0xb5, 0x2f, 0xfd, 0x00, 0x00];
            let err = Field::detect_from_circuit_bytes(&bytes).unwrap_err();
            assert!(matches!(err, RunError::Deserialize(_)));
        }
    }
}
