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
    #[serde(alias = "GoldilocksBasefold", alias = "goldilocks_basefold")]
    GoldilocksBasefold,
    #[serde(alias = "M31", alias = "mersenne31")]
    M31,
}

impl std::fmt::Display for Curve {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bn254 => write!(f, "bn254"),
            Self::Goldilocks => write!(f, "goldilocks"),
            Self::GoldilocksBasefold => write!(f, "goldilocks_basefold"),
            Self::M31 => write!(f, "m31"),
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
            "m31" | "mersenne31" => Ok(Self::M31),
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
            Curve::M31,
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
