use std::sync::Arc;

use arith::ExtensionField;
use serdes::{ExpSerde, SerdeResult};
use tree::{Node, Path, Tree};

use super::parameters::WHIR_RATE_LOG;

pub const fn whir_rate_log() -> usize {
    WHIR_RATE_LOG
}

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
    pub root: Node,
    pub num_vars: usize,
}

impl ExpSerde for WhirCommitment {
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
pub struct WhirRoundQueryProof {
    pub leaf_proof: Path,
}

impl ExpSerde for WhirRoundQueryProof {
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        self.leaf_proof.serialize_into(&mut writer)?;
        Ok(())
    }
    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        let leaf_proof = Path::deserialize_from(&mut reader)?;
        Ok(Self { leaf_proof })
    }
}

#[derive(Clone, Debug, Default)]
pub struct WhirOpening<F: ExtensionField> {
    pub round_commitments: Vec<Node>,
    pub sumcheck_messages: Vec<crate::basefold::SumcheckRoundMessage<F>>,
    pub ood_evaluations: Vec<Vec<F>>,
    pub pow_nonces: Vec<u64>,
    pub final_poly: Vec<F>,
    pub round_query_proofs: Vec<Vec<WhirRoundQueryProof>>,
}

impl<F: ExtensionField> ExpSerde for WhirOpening<F> {
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        (self.round_commitments.len() as u64).serialize_into(&mut writer)?;
        for n in &self.round_commitments {
            n.serialize_into(&mut writer)?;
        }
        (self.sumcheck_messages.len() as u64).serialize_into(&mut writer)?;
        for m in &self.sumcheck_messages {
            m.serialize_into(&mut writer)?;
        }
        (self.ood_evaluations.len() as u64).serialize_into(&mut writer)?;
        for evals in &self.ood_evaluations {
            (evals.len() as u64).serialize_into(&mut writer)?;
            for e in evals {
                e.serialize_into(&mut writer)?;
            }
        }
        (self.pow_nonces.len() as u64).serialize_into(&mut writer)?;
        for &n in &self.pow_nonces {
            n.serialize_into(&mut writer)?;
        }
        (self.final_poly.len() as u64).serialize_into(&mut writer)?;
        for f in &self.final_poly {
            f.serialize_into(&mut writer)?;
        }
        (self.round_query_proofs.len() as u64).serialize_into(&mut writer)?;
        for round_proofs in &self.round_query_proofs {
            (round_proofs.len() as u64).serialize_into(&mut writer)?;
            for qp in round_proofs {
                qp.serialize_into(&mut writer)?;
            }
        }
        Ok(())
    }
    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        const MAX_ROUNDS: usize = 64;
        const MAX_SUMCHECK: usize = 256;
        const MAX_QUERIES_PER_ROUND: usize = 1 << 16;
        const MAX_FINAL_POLY: usize = 1 << 20;

        let rc_len = u64::deserialize_from(&mut reader)? as usize;
        if rc_len > MAX_ROUNDS {
            return Err(serdes::SerdeError::DeserializeError);
        }
        let mut round_commitments = Vec::with_capacity(rc_len);
        for _ in 0..rc_len {
            round_commitments.push(Node::deserialize_from(&mut reader)?);
        }
        let sm_len = u64::deserialize_from(&mut reader)? as usize;
        if sm_len > MAX_SUMCHECK {
            return Err(serdes::SerdeError::DeserializeError);
        }
        let mut sumcheck_messages = Vec::with_capacity(sm_len);
        for _ in 0..sm_len {
            sumcheck_messages.push(crate::basefold::SumcheckRoundMessage::deserialize_from(
                &mut reader,
            )?);
        }
        let ood_len = u64::deserialize_from(&mut reader)? as usize;
        if ood_len > MAX_ROUNDS {
            return Err(serdes::SerdeError::DeserializeError);
        }
        let mut ood_evaluations = Vec::with_capacity(ood_len);
        for _ in 0..ood_len {
            let e_len = u64::deserialize_from(&mut reader)? as usize;
            if e_len > MAX_SUMCHECK {
                return Err(serdes::SerdeError::DeserializeError);
            }
            let mut evals = Vec::with_capacity(e_len);
            for _ in 0..e_len {
                evals.push(F::deserialize_from(&mut reader)?);
            }
            ood_evaluations.push(evals);
        }
        let pn_len = u64::deserialize_from(&mut reader)? as usize;
        if pn_len > MAX_ROUNDS {
            return Err(serdes::SerdeError::DeserializeError);
        }
        let mut pow_nonces = Vec::with_capacity(pn_len);
        for _ in 0..pn_len {
            pow_nonces.push(u64::deserialize_from(&mut reader)?);
        }
        let fp_len = u64::deserialize_from(&mut reader)? as usize;
        if fp_len > MAX_FINAL_POLY {
            return Err(serdes::SerdeError::DeserializeError);
        }
        let mut final_poly = Vec::with_capacity(fp_len);
        for _ in 0..fp_len {
            final_poly.push(F::deserialize_from(&mut reader)?);
        }
        let rq_len = u64::deserialize_from(&mut reader)? as usize;
        if rq_len > MAX_ROUNDS {
            return Err(serdes::SerdeError::DeserializeError);
        }
        let mut round_query_proofs = Vec::with_capacity(rq_len);
        for _ in 0..rq_len {
            let q_len = u64::deserialize_from(&mut reader)? as usize;
            if q_len > MAX_QUERIES_PER_ROUND {
                return Err(serdes::SerdeError::DeserializeError);
            }
            let mut proofs = Vec::with_capacity(q_len);
            for _ in 0..q_len {
                proofs.push(WhirRoundQueryProof::deserialize_from(&mut reader)?);
            }
            round_query_proofs.push(proofs);
        }
        Ok(Self {
            round_commitments,
            sumcheck_messages,
            ood_evaluations,
            pow_nonces,
            final_poly,
            round_query_proofs,
        })
    }
}

struct ErasedCodeword(Box<dyn std::any::Any + Send + Sync>);

impl std::fmt::Debug for ErasedCodeword {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ErasedCodeword")
    }
}

pub struct WhirScratchPad {
    commit_cache: Option<Arc<(Tree, ErasedCodeword, usize)>>,
}

impl Clone for WhirScratchPad {
    fn clone(&self) -> Self {
        Self {
            commit_cache: self.commit_cache.clone(),
        }
    }
}

impl std::fmt::Debug for WhirScratchPad {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WhirScratchPad")
            .field("cached", &self.commit_cache.is_some())
            .finish()
    }
}

impl Default for WhirScratchPad {
    fn default() -> Self {
        Self { commit_cache: None }
    }
}

impl WhirScratchPad {
    pub(crate) fn store_commit<F: Send + Sync + 'static>(
        &mut self,
        tree: Tree,
        codeword: Vec<F>,
        num_evals: usize,
    ) {
        self.commit_cache = Some(Arc::new((
            tree,
            ErasedCodeword(Box::new(codeword)),
            num_evals,
        )));
    }

    pub(crate) fn get_commit<F: Send + Sync + 'static>(
        &self,
        num_evals: usize,
    ) -> Option<(&Tree, &Vec<F>)> {
        let arc = self.commit_cache.as_ref()?;
        if arc.2 != num_evals {
            return None;
        }
        let codeword = arc.1 .0.downcast_ref::<Vec<F>>()?;
        Some((&arc.0, codeword))
    }
}

impl ExpSerde for WhirScratchPad {
    fn serialize_into<W: std::io::Write>(&self, _writer: W) -> SerdeResult<()> {
        Ok(())
    }
    fn deserialize_from<R: std::io::Read>(_reader: R) -> SerdeResult<Self> {
        Ok(Self::default())
    }
}
