use ethnum::U256;
use serdes::{ExpSerde, SerdeResult};

#[derive(Clone, Debug, Default)]
pub struct WhirSRS;

impl ExpSerde for WhirSRS {
    fn serialize_into<W: std::io::Write>(&self, _writer: W) -> SerdeResult<()> {
        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(_reader: R) -> SerdeResult<Self> {
        Ok(Self)
    }
}

impl gkr_engine::StructuredReferenceString for WhirSRS {
    type PKey = WhirSRS;
    type VKey = WhirSRS;

    fn into_keys(self) -> (Self::PKey, Self::VKey) {
        (WhirSRS, WhirSRS)
    }
}

#[derive(Clone, Debug, Default)]
pub struct WhirCommitment {
    pub num_vars: usize,
}

impl ExpSerde for WhirCommitment {
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        (self.num_vars as u64).serialize_into(&mut writer)?;
        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        let num_vars = u64::deserialize_from(&mut reader)? as usize;
        Ok(Self { num_vars })
    }
}

#[derive(Clone, Debug)]
pub struct WhirOpening {
    pub proof: whir::transcript::Proof,
}

impl Default for WhirOpening {
    fn default() -> Self {
        Self {
            proof: whir::transcript::Proof {
                narg_string: Vec::new(),
                hints: Vec::new(),
                #[cfg(debug_assertions)]
                pattern: Vec::new(),
            },
        }
    }
}

impl ExpSerde for WhirOpening {
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        let nlen = U256::from(self.proof.narg_string.len() as u64);
        nlen.serialize_into(&mut writer)?;
        writer.write_all(&self.proof.narg_string)?;

        let hlen = U256::from(self.proof.hints.len() as u64);
        hlen.serialize_into(&mut writer)?;
        writer.write_all(&self.proof.hints)?;

        #[cfg(debug_assertions)]
        {
            let pattern_json = serde_json::to_vec(&self.proof.pattern).unwrap_or_default();
            let plen = U256::from(pattern_json.len() as u64);
            plen.serialize_into(&mut writer)?;
            writer.write_all(&pattern_json)?;
        }

        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        let nlen = U256::deserialize_from(&mut reader)?.as_usize();
        let mut narg = vec![0u8; nlen];
        reader.read_exact(&mut narg)?;

        let hlen = U256::deserialize_from(&mut reader)?.as_usize();
        let mut hints = vec![0u8; hlen];
        reader.read_exact(&mut hints)?;

        #[cfg(debug_assertions)]
        let pattern = {
            let plen = U256::deserialize_from(&mut reader)?.as_usize();
            let mut pattern_json = vec![0u8; plen];
            reader.read_exact(&mut pattern_json)?;
            serde_json::from_slice(&pattern_json).unwrap_or_default()
        };

        Ok(Self {
            proof: whir::transcript::Proof {
                narg_string: narg,
                hints,
                #[cfg(debug_assertions)]
                pattern,
            },
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct WhirScratchPad {
    pub cached_vector: Option<Vec<u64>>,
}

impl ExpSerde for WhirScratchPad {
    fn serialize_into<W: std::io::Write>(&self, _writer: W) -> SerdeResult<()> {
        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(_reader: R) -> SerdeResult<Self> {
        Ok(Self::default())
    }
}
