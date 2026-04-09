#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum Curve {
    #[default]
    #[serde(alias = "BN254")]
    Bn254,
    #[serde(alias = "Goldilocks")]
    Goldilocks,
    #[serde(
        rename = "goldilocks_basefold",
        alias = "GoldilocksBasefold",
        alias = "goldilocksbasefold"
    )]
    GoldilocksBasefold,
    #[serde(
        rename = "goldilocks_ext2",
        alias = "GoldilocksExt2",
        alias = "goldilocksext2"
    )]
    GoldilocksExt2,
    #[serde(
        rename = "goldilocks_whir",
        alias = "GoldilocksWhir",
        alias = "goldilockswhir"
    )]
    GoldilocksWhir,
    #[serde(
        rename = "goldilocks_whir_pq",
        alias = "GoldilocksWhirPQ",
        alias = "goldilockswhirpq"
    )]
    GoldilocksWhirPQ,
}

impl Curve {
    #[must_use]
    pub fn is_field_compatible(&self, other: &Curve) -> bool {
        self.base_field_id() == other.base_field_id()
    }

    fn base_field_id(self) -> u8 {
        match self {
            Self::Bn254 => 0,
            Self::Goldilocks
            | Self::GoldilocksBasefold
            | Self::GoldilocksWhir
            | Self::GoldilocksWhirPQ => 1,
            Self::GoldilocksExt2 => 2,
        }
    }

    /// Canonical default curve variant for a given base field identifier.
    ///
    /// When detecting a curve from serialized circuit bytes alone, only the
    /// base field can be determined — multiple curve variants may share the
    /// same field but differ in PCS. This returns a sensible default for
    /// callers that lack additional context (e.g., bundles missing the
    /// `curve` manifest field).
    #[must_use]
    fn default_for_base_field(base_field_id: u8) -> Option<Self> {
        match base_field_id {
            0 => Some(Self::Bn254),
            1 => Some(Self::GoldilocksWhirPQ),
            2 => Some(Self::GoldilocksExt2),
            _ => None,
        }
    }

    /// Detect the base field of a serialized circuit by inspecting the
    /// embedded field modulus. Returns the canonical default curve variant
    /// for the detected field.
    ///
    /// This is a best-effort primitive intended as a fallback when bundle
    /// metadata does not carry an explicit curve. Callers that know the
    /// specific curve variant (e.g., from `CircuitParams.curve`) should
    /// prefer that.
    ///
    /// # Errors
    /// Returns `RunError::Deserialize` if the bytes cannot be decompressed,
    /// are too short to contain a modulus, or encode an unknown field.
    pub fn detect_base_field(
        circuit_bytes: &[u8],
    ) -> Result<Self, crate::runner::errors::RunError> {
        use crate::runner::errors::RunError;
        use expander_compiler::frontend::{
            BN254Config, CircuitField, FieldArith, GoldilocksConfig, GoldilocksExt2BasefoldConfig,
        };
        use expander_compiler::serdes::ExpSerde;
        use jstprove_io::auto_decompress_bytes;
        use std::io::Cursor;

        const MAGIC_SIZE: usize = 8;
        const MODULUS_SIZE: usize = 32;
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
        let goldilocks_ext2_modulus =
            <CircuitField<GoldilocksExt2BasefoldConfig> as FieldArith>::MODULUS;

        if modulus == bn254_modulus {
            return Self::default_for_base_field(0)
                .ok_or_else(|| RunError::Deserialize("unmapped base field".into()));
        }
        if modulus == goldilocks_modulus {
            return Self::default_for_base_field(1)
                .ok_or_else(|| RunError::Deserialize("unmapped base field".into()));
        }
        if modulus == goldilocks_ext2_modulus {
            return Self::default_for_base_field(2)
                .ok_or_else(|| RunError::Deserialize("unmapped base field".into()));
        }

        Err(RunError::Deserialize(format!(
            "unknown circuit field modulus: {modulus:?}"
        )))
    }
}

impl std::fmt::Display for Curve {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bn254 => write!(f, "bn254"),
            Self::Goldilocks => write!(f, "goldilocks"),
            Self::GoldilocksBasefold => write!(f, "goldilocks_basefold"),
            Self::GoldilocksExt2 => write!(f, "goldilocks_ext2"),
            Self::GoldilocksWhir => write!(f, "goldilocks_whir"),
            Self::GoldilocksWhirPQ => write!(f, "goldilocks_whir_pq"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CurveParseError(String);

impl std::fmt::Display for CurveParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unknown curve: {}", self.0)
    }
}

impl std::error::Error for CurveParseError {}

impl std::str::FromStr for Curve {
    type Err = CurveParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "bn254" => Ok(Self::Bn254),
            "goldilocks" => Ok(Self::Goldilocks),
            "goldilocks_basefold" => Ok(Self::GoldilocksBasefold),
            "goldilocks_ext2" => Ok(Self::GoldilocksExt2),
            "goldilocks_whir" => Ok(Self::GoldilocksWhir),
            "goldilocks_whir_pq" => Ok(Self::GoldilocksWhirPQ),
            other => Err(CurveParseError(other.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_bn254() {
        assert_eq!(Curve::default(), Curve::Bn254);
    }

    #[test]
    fn round_trip_display_parse() {
        for variant in [
            Curve::Bn254,
            Curve::Goldilocks,
            Curve::GoldilocksBasefold,
            Curve::GoldilocksExt2,
            Curve::GoldilocksWhir,
            Curve::GoldilocksWhirPQ,
        ] {
            let s = variant.to_string();
            let parsed: Curve = s.parse().unwrap();
            assert_eq!(parsed, variant);
        }
    }

    #[test]
    fn parse_unknown_errors() {
        assert!("unknown".parse::<Curve>().is_err());
    }
}
