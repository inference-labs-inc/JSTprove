use crate::circuit_functions::layers::LayerKind;

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum ProofSystem {
    #[default]
    Expander,
    Remainder,
    DirectBuilder,
}

impl ProofSystem {
    #[must_use]
    pub fn is_remainder(&self) -> bool {
        matches!(self, Self::Remainder)
    }

    #[must_use]
    pub fn is_direct_builder(&self) -> bool {
        matches!(self, Self::DirectBuilder)
    }

    #[must_use]
    pub fn supported_ops(&self) -> &'static [&'static str] {
        match self {
            Self::Expander | Self::DirectBuilder => LayerKind::SUPPORTED_OP_NAMES,
            Self::Remainder => Self::REMAINDER_OPS,
        }
    }

    const REMAINDER_OPS: &[&str] = &[
        "Add",
        "BatchNormalization",
        "Conv",
        "Flatten",
        "Gemm",
        "MaxPool",
        "ReLU",
        "Reshape",
        "Squeeze",
        "Sub",
        "Unsqueeze",
    ];
}

impl std::fmt::Display for ProofSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Expander => write!(f, "expander"),
            Self::Remainder => write!(f, "remainder"),
            Self::DirectBuilder => write!(f, "directbuilder"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProofSystemParseError(String);

impl std::fmt::Display for ProofSystemParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unknown proof system: {}", self.0)
    }
}

impl std::error::Error for ProofSystemParseError {}

impl std::str::FromStr for ProofSystem {
    type Err = ProofSystemParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "expander" => Ok(Self::Expander),
            "remainder" => Ok(Self::Remainder),
            "directbuilder" => Ok(Self::DirectBuilder),
            other => Err(ProofSystemParseError(other.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remainder_ops_subset_of_expander() {
        let expander = ProofSystem::Expander.supported_ops();
        for &name in ProofSystem::Remainder.supported_ops() {
            assert!(
                expander.contains(&name),
                "Remainder op {name:?} not in Expander list"
            );
        }
    }

    #[test]
    fn direct_builder_roundtrip_display_parse() {
        let ps = ProofSystem::DirectBuilder;
        let s = ps.to_string();
        assert_eq!(s, "directbuilder");
        let parsed: ProofSystem = s.parse().unwrap();
        assert!(parsed.is_direct_builder());
    }

    #[test]
    fn direct_builder_is_not_remainder() {
        assert!(!ProofSystem::DirectBuilder.is_remainder());
    }

    #[test]
    fn direct_builder_supported_ops_match_expander() {
        let expander = ProofSystem::Expander.supported_ops();
        let direct = ProofSystem::DirectBuilder.supported_ops();
        assert_eq!(expander, direct);
    }

    #[test]
    fn parse_unknown_variant_errors() {
        assert!("unknown".parse::<ProofSystem>().is_err());
    }
}
