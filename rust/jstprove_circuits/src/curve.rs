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
}

impl Curve {
    #[must_use]
    pub fn is_field_compatible(&self, other: &Curve) -> bool {
        self.base_field_id() == other.base_field_id()
    }

    fn base_field_id(self) -> u8 {
        match self {
            Self::Bn254 => 0,
            Self::Goldilocks | Self::GoldilocksBasefold | Self::GoldilocksWhir => 1,
            Self::GoldilocksExt2 => 2,
        }
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
