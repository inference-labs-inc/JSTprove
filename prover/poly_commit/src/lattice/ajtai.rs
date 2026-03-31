use arith::Field;
use ethnum::U256;
use gkr_engine::{PCSParams, StructuredReferenceString};
use serdes::{ExpSerde, SerdeResult};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};

pub const LATTICE_COMMITMENT_ROWS: usize = 256;

#[derive(Clone, Debug)]
pub struct LatticePCSParams {
    pub num_vars: usize,
    pub commitment_rows: usize,
}

impl Default for LatticePCSParams {
    fn default() -> Self {
        Self {
            num_vars: 0,
            commitment_rows: LATTICE_COMMITMENT_ROWS,
        }
    }
}

impl PCSParams for LatticePCSParams {
    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

#[derive(Clone, Debug, Default)]
pub struct AjtaiSRS {
    pub seed: [u8; 32],
}

impl ExpSerde for AjtaiSRS {
    fn serialize_into<W: Write>(&self, mut writer: W) -> SerdeResult<()> {
        writer.write_all(&self.seed)?;
        Ok(())
    }

    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let mut seed = [0u8; 32];
        reader.read_exact(&mut seed)?;
        Ok(AjtaiSRS { seed })
    }
}

impl StructuredReferenceString for AjtaiSRS {
    type PKey = AjtaiSRS;
    type VKey = AjtaiSRS;

    fn into_keys(self) -> (Self::PKey, Self::VKey) {
        (self.clone(), self)
    }
}

#[derive(Clone, Debug, Default)]
pub struct AjtaiCommitment<F: Field> {
    pub digest: Vec<F>,
}

impl<F: Field> ExpSerde for AjtaiCommitment<F> {
    fn serialize_into<W: Write>(&self, mut writer: W) -> SerdeResult<()> {
        let len = U256::from(self.digest.len() as u64);
        len.serialize_into(&mut writer)?;
        for v in &self.digest {
            v.serialize_into(&mut writer)?;
        }
        Ok(())
    }

    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let len = U256::deserialize_from(&mut reader)?;
        let mut digest = Vec::with_capacity(len.as_usize());
        for _ in 0..len.as_usize() {
            digest.push(F::deserialize_from(&mut reader)?);
        }
        Ok(AjtaiCommitment { digest })
    }
}

#[derive(Clone, Debug, Default)]
pub struct AjtaiOpening<F: Field> {
    pub coefficients: Vec<F>,
}

impl<F: Field> ExpSerde for AjtaiOpening<F> {
    fn serialize_into<W: Write>(&self, mut writer: W) -> SerdeResult<()> {
        let len = U256::from(self.coefficients.len() as u64);
        len.serialize_into(&mut writer)?;
        for v in &self.coefficients {
            v.serialize_into(&mut writer)?;
        }
        Ok(())
    }

    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let len = U256::deserialize_from(&mut reader)?;
        let mut coefficients = Vec::with_capacity(len.as_usize());
        for _ in 0..len.as_usize() {
            coefficients.push(F::deserialize_from(&mut reader)?);
        }
        Ok(AjtaiOpening { coefficients })
    }
}

#[derive(Clone, Debug, Default)]
pub struct AjtaiScratchPad;

impl ExpSerde for AjtaiScratchPad {
    fn serialize_into<W: Write>(&self, _writer: W) -> SerdeResult<()> {
        Ok(())
    }
    fn deserialize_from<R: Read>(_reader: R) -> SerdeResult<Self> {
        Ok(AjtaiScratchPad)
    }
}

pub fn generate_matrix_row<F: Field>(seed: &[u8; 32], row_index: usize, num_cols: usize) -> Vec<F> {
    let mut row = Vec::with_capacity(num_cols);
    let mut col = 0;

    while col < num_cols {
        let mut hasher = Sha256::new();
        hasher.update(seed);
        hasher.update(row_index.to_le_bytes());
        hasher.update(col.to_le_bytes());
        let hash_output = hasher.finalize();

        let elem = F::from_uniform_bytes(&hash_output);
        row.push(elem);
        col += 1;
    }

    row
}

pub fn ajtai_commit<F: Field>(
    seed: &[u8; 32],
    commitment_rows: usize,
    coefficients: &[F],
) -> Vec<F> {
    let num_cols = coefficients.len();
    let mut digest = vec![F::ZERO; commitment_rows];

    for row_idx in 0..commitment_rows {
        let matrix_row = generate_matrix_row::<F>(seed, row_idx, num_cols);
        let mut acc = F::ZERO;
        for (a, m) in matrix_row.iter().zip(coefficients.iter()) {
            acc += *a * m;
        }
        digest[row_idx] = acc;
    }

    digest
}

pub fn ajtai_verify<F: Field>(
    seed: &[u8; 32],
    commitment_rows: usize,
    commitment: &[F],
    coefficients: &[F],
) -> bool {
    let recomputed = ajtai_commit::<F>(seed, commitment_rows, coefficients);
    if recomputed.len() != commitment.len() {
        return false;
    }
    recomputed
        .iter()
        .zip(commitment.iter())
        .all(|(a, b)| a == b)
}
