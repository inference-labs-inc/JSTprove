use crate::circuit_functions::layers::LayerKind;

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum ProofSystem {
    #[default]
    Expander,
}

impl ProofSystem {
    #[must_use]
    pub fn supported_ops(&self) -> &'static [&'static str] {
        match self {
            Self::Expander => LayerKind::SUPPORTED_OP_NAMES,
        }
    }
}

impl std::fmt::Display for ProofSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Expander => write!(f, "expander"),
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
            other => Err(ProofSystemParseError(other.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_unknown_variant_errors() {
        assert!("unknown".parse::<ProofSystem>().is_err());
        assert!("remainder".parse::<ProofSystem>().is_err());
    }
}
