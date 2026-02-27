use std::io::{Read, Write};
use std::path::Path;

use anyhow::Result;
use serde::{de::DeserializeOwned, Serialize};

use crate::compress::{auto_reader, compressed_writer};
use crate::version::ArtifactVersion;

const CIRCUIT_FILENAME: &str = "circuit.bin";
const WITNESS_SOLVER_FILENAME: &str = "witness_solver.bin";
const MANIFEST_FILENAME: &str = "manifest.msgpack";

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(bound(
    serialize = "M: serde::Serialize",
    deserialize = "M: serde::de::DeserializeOwned"
))]
pub struct BundleManifest<M> {
    #[serde(default)]
    pub metadata: Option<M>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<ArtifactVersion>,
}

pub fn write_bundle<M: Serialize>(
    dir: &Path,
    circuit: &[u8],
    witness_solver: &[u8],
    metadata: Option<M>,
    version: Option<ArtifactVersion>,
    compress: bool,
) -> Result<()> {
    std::fs::create_dir(dir)?;

    write_blob(dir.join(CIRCUIT_FILENAME), circuit, compress)?;
    write_blob(dir.join(WITNESS_SOLVER_FILENAME), witness_solver, compress)?;

    let manifest = BundleManifest { metadata, version };
    crate::serialize_to_file(&manifest, &dir.join(MANIFEST_FILENAME), false)?;
    Ok(())
}

pub struct BundleBlobs<M> {
    pub circuit: Vec<u8>,
    pub witness_solver: Vec<u8>,
    pub metadata: Option<M>,
    pub version: Option<ArtifactVersion>,
}

pub fn read_bundle<M: DeserializeOwned>(dir: &Path) -> Result<BundleBlobs<M>> {
    let manifest: BundleManifest<M> = crate::deserialize_from_file(&dir.join(MANIFEST_FILENAME))?;

    let circuit = read_blob(dir.join(CIRCUIT_FILENAME))?;
    let witness_solver = read_blob(dir.join(WITNESS_SOLVER_FILENAME))?;

    Ok(BundleBlobs {
        circuit,
        witness_solver,
        metadata: manifest.metadata,
        version: manifest.version,
    })
}

pub fn read_circuit_blob(dir: &Path) -> Result<Vec<u8>> {
    read_blob(dir.join(CIRCUIT_FILENAME))
}

pub fn read_bundle_metadata<M: DeserializeOwned>(
    dir: &Path,
) -> Result<(Option<M>, Option<ArtifactVersion>)> {
    let manifest: BundleManifest<M> = crate::deserialize_from_file(&dir.join(MANIFEST_FILENAME))?;
    Ok((manifest.metadata, manifest.version))
}

fn write_blob(path: impl AsRef<Path>, data: &[u8], compress: bool) -> Result<()> {
    let file = std::fs::File::create(path.as_ref())?;
    let mut writer = compressed_writer(file, compress)?;
    writer.write_all(data)?;
    writer.finish()?;
    Ok(())
}

fn read_blob(path: impl AsRef<Path>) -> Result<Vec<u8>> {
    let file = std::fs::File::open(path.as_ref())?;
    let mut reader = auto_reader(file)?;
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf)?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use tempfile::TempDir;

    use super::*;

    #[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
    struct TestMeta {
        name: String,
        values: HashMap<String, usize>,
    }

    #[test]
    fn roundtrip_uncompressed() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("bundle");

        let circuit = vec![1u8, 2, 3, 4, 5];
        let ws = vec![10u8, 20, 30];
        let meta = TestMeta {
            name: "test".into(),
            values: HashMap::new(),
        };

        write_bundle(&dir, &circuit, &ws, Some(meta.clone()), None, false).unwrap();

        assert!(dir.join(CIRCUIT_FILENAME).exists());
        assert!(dir.join(WITNESS_SOLVER_FILENAME).exists());
        assert!(dir.join(MANIFEST_FILENAME).exists());

        let b: BundleBlobs<TestMeta> = read_bundle(&dir).unwrap();
        assert_eq!(b.circuit, circuit);
        assert_eq!(b.witness_solver, ws);
        assert_eq!(b.metadata.unwrap(), meta);
        assert!(b.version.is_none());
    }

    #[test]
    fn roundtrip_compressed() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("bundle");

        let circuit = vec![0xAA; 4096];
        let ws = vec![0xBB; 2048];

        write_bundle::<TestMeta>(&dir, &circuit, &ws, None, None, true).unwrap();

        let raw_circuit = std::fs::read(dir.join(CIRCUIT_FILENAME)).unwrap();
        assert_eq!(
            &raw_circuit[..4],
            &crate::envelope::ZSTD_MAGIC,
            "blob should be zstd-compressed"
        );
        assert!(
            raw_circuit.len() < circuit.len(),
            "compressed should be smaller"
        );

        let b: BundleBlobs<TestMeta> = read_bundle(&dir).unwrap();
        assert_eq!(b.circuit, circuit);
        assert_eq!(b.witness_solver, ws);
        assert!(b.metadata.is_none());
    }

    #[test]
    fn metadata_only_read() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("bundle");

        let mut vals = HashMap::new();
        vals.insert("layer_count".into(), 42);
        let meta = TestMeta {
            name: "model_v2".into(),
            values: vals,
        };

        write_bundle(&dir, &[0; 64], &[0; 32], Some(meta.clone()), None, true).unwrap();

        let (m, _) = read_bundle_metadata::<TestMeta>(&dir).unwrap();
        assert_eq!(m, Some(meta));
    }
}
