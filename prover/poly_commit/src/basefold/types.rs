use arith::ExtensionField;
use ethnum::U256;
use serdes::{ExpSerde, SerdeResult};
use tree::{Node, Path};

pub const BASEFOLD_NUM_QUERIES: usize = 100;
pub const RATE_LOG: usize = 1;

#[derive(Clone, Debug, Default)]
pub struct BasefoldSRS;

impl ExpSerde for BasefoldSRS {
    fn serialize_into<W: std::io::Write>(&self, _writer: W) -> SerdeResult<()> {
        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(_reader: R) -> SerdeResult<Self> {
        Ok(Self)
    }
}

impl gkr_engine::StructuredReferenceString for BasefoldSRS {
    type PKey = BasefoldSRS;
    type VKey = BasefoldSRS;

    fn into_keys(self) -> (Self::PKey, Self::VKey) {
        (BasefoldSRS, BasefoldSRS)
    }
}

#[derive(Clone, Debug, Default)]
pub struct BasefoldCommitment {
    pub root: Node,
    pub num_vars: usize,
}

impl ExpSerde for BasefoldCommitment {
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        self.root.serialize_into(&mut writer)?;
        (self.num_vars as u64).serialize_into(&mut writer)?;
        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        let root = Node::deserialize_from(&mut reader)?;
        let num_vars = u64::deserialize_from(&mut reader)? as usize;
        Ok(Self { root, num_vars })
    }
}

#[derive(Clone, Debug, Default)]
pub struct SumcheckRoundMessage<F: ExtensionField> {
    pub eval_at_1: F,
    pub eval_at_2: F,
}

impl<F: ExtensionField> ExpSerde for SumcheckRoundMessage<F> {
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        self.eval_at_1.serialize_into(&mut writer)?;
        self.eval_at_2.serialize_into(&mut writer)?;
        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        let eval_at_1 = F::deserialize_from(&mut reader)?;
        let eval_at_2 = F::deserialize_from(&mut reader)?;
        Ok(Self {
            eval_at_1,
            eval_at_2,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct BasefoldOpening<F: ExtensionField> {
    pub round_commitments: Vec<Node>,
    pub sumcheck_messages: Vec<SumcheckRoundMessage<F>>,
    pub final_poly: Vec<F>,
    pub query_proofs: Vec<FriQueryProof<F>>,
}

impl<F: ExtensionField> ExpSerde for BasefoldOpening<F> {
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        let len = U256::from(self.round_commitments.len() as u64);
        len.serialize_into(&mut writer)?;
        for node in &self.round_commitments {
            node.serialize_into(&mut writer)?;
        }

        let slen = U256::from(self.sumcheck_messages.len() as u64);
        slen.serialize_into(&mut writer)?;
        for msg in &self.sumcheck_messages {
            msg.serialize_into(&mut writer)?;
        }

        let flen = U256::from(self.final_poly.len() as u64);
        flen.serialize_into(&mut writer)?;
        for f in &self.final_poly {
            f.serialize_into(&mut writer)?;
        }

        let qlen = U256::from(self.query_proofs.len() as u64);
        qlen.serialize_into(&mut writer)?;
        for qp in &self.query_proofs {
            qp.serialize_into(&mut writer)?;
        }

        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        let len = U256::deserialize_from(&mut reader)?.as_usize();
        let mut round_commitments = Vec::with_capacity(len);
        for _ in 0..len {
            round_commitments.push(Node::deserialize_from(&mut reader)?);
        }

        let slen = U256::deserialize_from(&mut reader)?.as_usize();
        let mut sumcheck_messages = Vec::with_capacity(slen);
        for _ in 0..slen {
            sumcheck_messages.push(SumcheckRoundMessage::deserialize_from(&mut reader)?);
        }

        let flen = U256::deserialize_from(&mut reader)?.as_usize();
        let mut final_poly = Vec::with_capacity(flen);
        for _ in 0..flen {
            final_poly.push(F::deserialize_from(&mut reader)?);
        }

        let qlen = U256::deserialize_from(&mut reader)?.as_usize();
        let mut query_proofs = Vec::with_capacity(qlen);
        for _ in 0..qlen {
            query_proofs.push(FriQueryProof::deserialize_from(&mut reader)?);
        }

        Ok(Self {
            round_commitments,
            sumcheck_messages,
            final_poly,
            query_proofs,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct FriQueryProof<F: ExtensionField> {
    pub initial_leaf_proof: Path,
    pub initial_sibling_proof: Path,
    pub round_proofs: Vec<FriRoundProof<F>>,
}

impl<F: ExtensionField> ExpSerde for FriQueryProof<F> {
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        self.initial_leaf_proof.serialize_into(&mut writer)?;
        self.initial_sibling_proof.serialize_into(&mut writer)?;
        let len = U256::from(self.round_proofs.len() as u64);
        len.serialize_into(&mut writer)?;
        for rp in &self.round_proofs {
            rp.serialize_into(&mut writer)?;
        }
        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        let initial_leaf_proof = Path::deserialize_from(&mut reader)?;
        let initial_sibling_proof = Path::deserialize_from(&mut reader)?;
        let len = U256::deserialize_from(&mut reader)?.as_usize();
        let mut round_proofs = Vec::with_capacity(len);
        for _ in 0..len {
            round_proofs.push(FriRoundProof::deserialize_from(&mut reader)?);
        }
        Ok(Self {
            initial_leaf_proof,
            initial_sibling_proof,
            round_proofs,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct FriRoundProof<F: ExtensionField> {
    pub leaf_proof: Path,
    pub sibling_proof: Path,
    pub leaf_values: Vec<F>,
    pub sibling_values: Vec<F>,
}

impl<F: ExtensionField> ExpSerde for FriRoundProof<F> {
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        self.leaf_proof.serialize_into(&mut writer)?;
        self.sibling_proof.serialize_into(&mut writer)?;
        let llen = U256::from(self.leaf_values.len() as u64);
        llen.serialize_into(&mut writer)?;
        for v in &self.leaf_values {
            v.serialize_into(&mut writer)?;
        }
        let slen = U256::from(self.sibling_values.len() as u64);
        slen.serialize_into(&mut writer)?;
        for v in &self.sibling_values {
            v.serialize_into(&mut writer)?;
        }
        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        let leaf_proof = Path::deserialize_from(&mut reader)?;
        let sibling_proof = Path::deserialize_from(&mut reader)?;
        let llen = U256::deserialize_from(&mut reader)?.as_usize();
        let mut leaf_values = Vec::with_capacity(llen);
        for _ in 0..llen {
            leaf_values.push(F::deserialize_from(&mut reader)?);
        }
        let slen = U256::deserialize_from(&mut reader)?.as_usize();
        let mut sibling_values = Vec::with_capacity(slen);
        for _ in 0..slen {
            sibling_values.push(F::deserialize_from(&mut reader)?);
        }
        Ok(Self {
            leaf_proof,
            sibling_proof,
            leaf_values,
            sibling_values,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct BasefoldScratchPad;

impl ExpSerde for BasefoldScratchPad {
    fn serialize_into<W: std::io::Write>(&self, _writer: W) -> SerdeResult<()> {
        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(_reader: R) -> SerdeResult<Self> {
        Ok(Self)
    }
}
