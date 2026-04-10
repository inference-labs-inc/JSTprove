use arith::{ExtensionField, Field};
use gkr_engine::Proof;
use serdes::{ExpSerde, SerdeResult};

use super::prove::HolographicProof;

pub struct PerLayerWiringClaims<E: Field> {
    pub layer_index: usize,
    pub eval_cst: E,
    pub eval_add: E,
    pub eval_mul: E,
}

pub struct CombinedHolographicProof<E: ExtensionField> {
    pub gkr_proof: Proof,
    pub claimed_v: E,
    pub holographic_proof: HolographicProof<E>,
    pub wiring_claims: Vec<PerLayerWiringClaims<E>>,
}

impl<E: ExtensionField> ExpSerde for CombinedHolographicProof<E> {
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        self.gkr_proof.serialize_into(&mut writer)?;
        self.claimed_v.serialize_into(&mut writer)?;
        self.holographic_proof.serialize_into(&mut writer)?;
        (self.wiring_claims.len() as u64).serialize_into(&mut writer)?;
        for claim in &self.wiring_claims {
            (claim.layer_index as u64).serialize_into(&mut writer)?;
            claim.eval_cst.serialize_into(&mut writer)?;
            claim.eval_add.serialize_into(&mut writer)?;
            claim.eval_mul.serialize_into(&mut writer)?;
        }
        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        let gkr_proof = Proof::deserialize_from(&mut reader)?;
        let claimed_v = E::deserialize_from(&mut reader)?;
        let holographic_proof = HolographicProof::deserialize_from(&mut reader)?;
        let n_claims = u64::deserialize_from(&mut reader)? as usize;
        if n_claims > (1 << 16) {
            return Err(serdes::SerdeError::DeserializeError);
        }
        let mut wiring_claims = Vec::with_capacity(n_claims);
        for _ in 0..n_claims {
            let layer_index = u64::deserialize_from(&mut reader)? as usize;
            let eval_cst = E::deserialize_from(&mut reader)?;
            let eval_add = E::deserialize_from(&mut reader)?;
            let eval_mul = E::deserialize_from(&mut reader)?;
            wiring_claims.push(PerLayerWiringClaims {
                layer_index,
                eval_cst,
                eval_add,
                eval_mul,
            });
        }
        Ok(Self {
            gkr_proof,
            claimed_v,
            holographic_proof,
            wiring_claims,
        })
    }
}
