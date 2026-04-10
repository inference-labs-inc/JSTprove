use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{de::DeserializeOwned, Serialize};

use crate::compress::{auto_reader, compressed_writer};
use crate::version::ArtifactVersion;

const CIRCUIT_FILENAME: &str = "circuit.bin";
const WITNESS_SOLVER_FILENAME: &str = "witness_solver.bin";
const MANIFEST_FILENAME: &str = "manifest.msgpack";
const VK_FILENAME: &str = "vk.bin";

struct WriteGuard {
    paths: Vec<PathBuf>,
    committed: bool,
}

impl WriteGuard {
    fn new() -> Self {
        Self {
            paths: vec![],
            committed: false,
        }
    }

    fn create_dir(&mut self, p: &Path) -> Result<()> {
        std::fs::create_dir(p)?;
        self.paths.push(p.to_owned());
        Ok(())
    }

    fn write_blob(&mut self, path: PathBuf, data: &[u8], compress: bool) -> Result<()> {
        write_blob(&path, data, compress)?;
        self.paths.push(path);
        Ok(())
    }

    fn serialize_to_file<T: Serialize>(&mut self, val: &T, path: PathBuf) -> Result<()> {
        crate::serialize_to_file(val, &path, false)?;
        self.paths.push(path);
        Ok(())
    }

    fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for WriteGuard {
    fn drop(&mut self) {
        if !self.committed {
            for p in self.paths.iter().rev() {
                if p.is_dir() {
                    let _ = std::fs::remove_dir(p);
                } else {
                    let _ = std::fs::remove_file(p);
                }
            }
        }
    }
}

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
    write_bundle_with_vk(
        dir,
        circuit,
        witness_solver,
        None,
        metadata,
        version,
        compress,
    )
}

/// Write a bundle that additionally contains an optional `vk.bin`
/// blob — the holographic-GKR verifying key from
/// `gkr::holographic::HolographicVerifyingKey`. Validators that
/// only need the VK can fetch `vk.bin` independently of the much
/// larger `circuit.bin` via [`read_vk_only`].
///
/// Passing `vk = None` is equivalent to [`write_bundle`] and does
/// not create an empty `vk.bin` file, so non-holographic flows
/// produce byte-identical bundles to before.
pub fn write_bundle_with_vk<M: Serialize>(
    dir: &Path,
    circuit: &[u8],
    witness_solver: &[u8],
    vk: Option<&[u8]>,
    metadata: Option<M>,
    version: Option<ArtifactVersion>,
    compress: bool,
) -> Result<()> {
    let mut guard = WriteGuard::new();
    guard.create_dir(dir)?;
    guard.write_blob(dir.join(CIRCUIT_FILENAME), circuit, compress)?;
    guard.write_blob(dir.join(WITNESS_SOLVER_FILENAME), witness_solver, compress)?;
    if let Some(vk_bytes) = vk {
        guard.write_blob(dir.join(VK_FILENAME), vk_bytes, compress)?;
    }
    let manifest = BundleManifest { metadata, version };
    guard.serialize_to_file(&manifest, dir.join(MANIFEST_FILENAME))?;
    guard.commit();
    Ok(())
}

pub struct BundleBlobs<M> {
    pub circuit: Vec<u8>,
    pub witness_solver: Vec<u8>,
    /// Holographic GKR verifying key, if the bundle was written
    /// via [`write_bundle_with_vk`] with a non-`None` VK. Read
    /// independently via [`read_vk_only`] when only the VK is
    /// needed (the validator path).
    pub vk: Option<Vec<u8>>,
    pub metadata: Option<M>,
    pub version: Option<ArtifactVersion>,
}

pub fn read_bundle<M: DeserializeOwned>(dir: &Path) -> Result<BundleBlobs<M>> {
    let manifest: BundleManifest<M> = crate::deserialize_from_file(&dir.join(MANIFEST_FILENAME))?;

    let circuit = read_blob(dir.join(CIRCUIT_FILENAME))?;
    let witness_solver = read_blob(dir.join(WITNESS_SOLVER_FILENAME))?;
    let vk_path = dir.join(VK_FILENAME);
    let vk = if vk_path.exists() {
        Some(read_blob(vk_path)?)
    } else {
        None
    };

    Ok(BundleBlobs {
        circuit,
        witness_solver,
        vk,
        metadata: manifest.metadata,
        version: manifest.version,
    })
}

pub fn read_circuit_blob(dir: &Path) -> Result<Vec<u8>> {
    read_blob(dir.join(CIRCUIT_FILENAME))
}

/// Read the holographic GKR verifying key from a bundle without
/// touching the circuit or witness-solver blobs. This is the
/// validator-side fetch path: validators that only need to verify
/// proofs against a fixed VK can avoid downloading the circuit
/// entirely.
///
/// # Errors
/// Returns an error if `vk.bin` does not exist in the bundle
/// directory or cannot be decompressed.
pub fn read_vk_only(dir: &Path) -> Result<Vec<u8>> {
    let path = dir.join(VK_FILENAME);
    if !path.exists() {
        anyhow::bail!("bundle at {} has no vk.bin", dir.display());
    }
    read_blob(path)
}

/// Returns `true` if the bundle directory contains a `vk.bin`
/// blob.
#[must_use]
pub fn bundle_has_vk(dir: &Path) -> bool {
    dir.join(VK_FILENAME).exists()
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
    let capacity = file
        .metadata()
        .ok()
        .and_then(|m| usize::try_from(m.len()).ok())
        .unwrap_or(0);
    let mut reader = auto_reader(file)?;
    let mut buf = Vec::with_capacity(capacity);
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
    fn vk_round_trip_uncompressed() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("bundle");
        let circuit = vec![1u8, 2, 3];
        let ws = vec![4u8, 5];
        let vk = vec![0xDEu8, 0xAD, 0xBE, 0xEF];

        write_bundle_with_vk::<TestMeta>(&dir, &circuit, &ws, Some(&vk), None, None, false)
            .unwrap();

        assert!(bundle_has_vk(&dir));

        let b: BundleBlobs<TestMeta> = read_bundle(&dir).unwrap();
        assert_eq!(b.circuit, circuit);
        assert_eq!(b.witness_solver, ws);
        assert_eq!(b.vk, Some(vk));
    }

    #[test]
    fn vk_round_trip_compressed() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("bundle");
        let circuit = vec![0xCC; 4096];
        let ws = vec![0xDD; 1024];
        let vk: Vec<u8> = (0u16..=255).map(|i| (i as u8).wrapping_mul(7)).collect();

        write_bundle_with_vk::<TestMeta>(&dir, &circuit, &ws, Some(&vk), None, None, true).unwrap();

        let b: BundleBlobs<TestMeta> = read_bundle(&dir).unwrap();
        assert_eq!(b.vk, Some(vk));
    }

    #[test]
    fn vk_only_read_skips_circuit() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("bundle");
        let circuit = vec![0xCCu8; 1024];
        let ws = vec![0xDDu8; 512];
        let vk = vec![0xAAu8, 0xBB, 0xCC];

        write_bundle_with_vk::<TestMeta>(&dir, &circuit, &ws, Some(&vk), None, None, false)
            .unwrap();

        // Validator path: read just the vk.
        let read_vk = read_vk_only(&dir).unwrap();
        assert_eq!(read_vk, vk);
    }

    #[test]
    fn read_vk_only_errors_on_missing_vk() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("bundle");
        // Old-style bundle without vk
        write_bundle::<TestMeta>(&dir, &[1, 2, 3], &[4, 5], None, None, false).unwrap();
        assert!(!bundle_has_vk(&dir));
        let err = read_vk_only(&dir).unwrap_err();
        assert!(err.to_string().contains("vk.bin"));
    }

    #[test]
    fn write_bundle_without_vk_omits_file() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("bundle");
        write_bundle::<TestMeta>(&dir, &[1, 2, 3], &[4, 5], None, None, false).unwrap();
        // Original write_bundle path must NOT create a vk.bin file
        // — backwards compat with non-holographic flows.
        assert!(!dir.join(VK_FILENAME).exists());
        let b: BundleBlobs<TestMeta> = read_bundle(&dir).unwrap();
        assert!(b.vk.is_none());
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

        let (m, version) = read_bundle_metadata::<TestMeta>(&dir).unwrap();
        assert_eq!(m, Some(meta));
        assert!(version.is_none());
    }
}
