use crate::circuit_functions::layers::LayerKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProofSystem {
    Expander,
    Remainder,
}

impl ProofSystem {
    #[must_use]
    pub fn supported_ops(&self) -> &'static [&'static str] {
        match self {
            Self::Expander => LayerKind::SUPPORTED_OP_NAMES,
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
}
