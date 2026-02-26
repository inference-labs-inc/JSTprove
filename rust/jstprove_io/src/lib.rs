#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate
)]

pub mod compress;
pub mod envelope;
pub mod version;

pub use compress::{
    auto_decompress_bytes, auto_reader, compress_bytes, compressed_writer, decompress_bytes,
    maybe_compress_bytes, MaybeCompressed, ZSTD_COMPRESSION_LEVEL,
};
pub use envelope::{DetectedFormat, ENVELOPE_HEADER_LEN, ENVELOPE_MAGIC, ZSTD_MAGIC};
pub use version::ArtifactVersion;

pub type Error = anyhow::Error;

use std::io::{Read, Write};
use std::path::Path;

use anyhow::Result;
use serde::{de::DeserializeOwned, Serialize};

pub fn serialize_to_bytes<T: Serialize>(value: &T, compress: bool) -> Result<Vec<u8>> {
    let msgpack = rmp_serde::to_vec_named(value)?;
    let (payload, compressed) = if compress {
        (compress_bytes(&msgpack)?, true)
    } else {
        (msgpack, false)
    };
    Ok(envelope::write_envelope(&payload, compressed))
}

pub fn serialize_to_file<T: Serialize>(value: &T, path: &Path, compress: bool) -> Result<usize> {
    let bytes = serialize_to_bytes(value, compress)?;
    std::fs::write(path, &bytes)?;
    Ok(bytes.len())
}

pub fn deserialize_from_bytes<T: DeserializeOwned>(data: &[u8]) -> Result<T> {
    match envelope::detect_format(data) {
        DetectedFormat::Envelope => {
            let header = envelope::parse_header(data)?;
            let len: usize = header.payload_len.try_into().map_err(|_| {
                anyhow::anyhow!(
                    "payload length {} exceeds platform address space",
                    header.payload_len
                )
            })?;
            let remaining = data.len().saturating_sub(ENVELOPE_HEADER_LEN);
            if remaining < len {
                anyhow::bail!(
                    "truncated envelope: declared payload {len} bytes but only {remaining} available"
                );
            }
            let payload = &data[ENVELOPE_HEADER_LEN..][..len];
            envelope::verify_crc(payload, header.crc32)?;
            let msgpack = if header.compressed {
                decompress_bytes(payload)?
            } else {
                payload.to_vec()
            };
            Ok(rmp_serde::from_slice(&msgpack)?)
        }
        DetectedFormat::LegacyZstd => {
            let decompressed = decompress_bytes(data)?;
            Ok(rmp_serde::from_slice(&decompressed)?)
        }
        DetectedFormat::RawMsgpack => Ok(rmp_serde::from_slice(data)?),
    }
}

pub fn deserialize_from_file<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let data = std::fs::read(path)?;
    deserialize_from_bytes(&data)
}

pub fn write_msgpack_stdout<T: Serialize>(value: &T) -> Result<()> {
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    value
        .serialize(&mut rmp_serde::Serializer::new(&mut lock).with_struct_map())
        .map_err(|e| anyhow::anyhow!("msgpack stdout serialization failed: {e}"))?;
    lock.flush()?;
    Ok(())
}

pub fn read_msgpack_stdin<T: DeserializeOwned>() -> Result<T> {
    read_msgpack_reader(std::io::BufReader::new(std::io::stdin()))
}

pub fn read_msgpack_reader<T: DeserializeOwned>(reader: impl Read) -> Result<T> {
    Ok(rmp_serde::from_read(reader)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct TestPayload {
        name: String,
        values: Vec<i64>,
        metadata: HashMap<String, usize>,
    }

    fn sample_payload() -> TestPayload {
        let mut metadata = HashMap::new();
        metadata.insert("layer_0".into(), 48);
        metadata.insert("layer_1".into(), 32);
        TestPayload {
            name: "test_model".into(),
            values: vec![1, 2, 3, -42, i64::MAX, i64::MIN],
            metadata,
        }
    }

    #[test]
    fn roundtrip_uncompressed() {
        let original = sample_payload();
        let bytes = serialize_to_bytes(&original, false).unwrap();
        assert_eq!(&bytes[..4], &ENVELOPE_MAGIC);
        let recovered: TestPayload = deserialize_from_bytes(&bytes).unwrap();
        assert_eq!(original, recovered);
    }

    #[test]
    fn roundtrip_compressed() {
        let original = sample_payload();
        let bytes = serialize_to_bytes(&original, true).unwrap();
        assert_eq!(&bytes[..4], &ENVELOPE_MAGIC);
        let recovered: TestPayload = deserialize_from_bytes(&bytes).unwrap();
        assert_eq!(original, recovered);
    }

    #[test]
    fn legacy_zstd_fallback() {
        let original = sample_payload();
        let msgpack = rmp_serde::to_vec_named(&original).unwrap();
        let legacy = zstd::encode_all(msgpack.as_slice(), 3).unwrap();
        assert_eq!(&legacy[..4], &ZSTD_MAGIC);
        let recovered: TestPayload = deserialize_from_bytes(&legacy).unwrap();
        assert_eq!(original, recovered);
    }

    #[test]
    fn legacy_raw_msgpack_fallback() {
        let original = sample_payload();
        let raw = rmp_serde::to_vec_named(&original).unwrap();
        assert_ne!(&raw[..4], &ENVELOPE_MAGIC);
        assert_ne!(&raw[..4], &ZSTD_MAGIC);
        let recovered: TestPayload = deserialize_from_bytes(&raw).unwrap();
        assert_eq!(original, recovered);
    }

    #[test]
    fn crc_corruption_detected() {
        let original = sample_payload();
        let mut bytes = serialize_to_bytes(&original, false).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        let result = deserialize_from_bytes::<TestPayload>(&bytes);
        assert!(result.is_err());
        assert!(
            format!("{:#}", result.unwrap_err()).contains("CRC"),
            "error should mention CRC"
        );
    }

    #[test]
    fn file_roundtrip() {
        let original = sample_payload();
        let dir = std::env::temp_dir().join("jstprove_io_test");
        std::fs::create_dir_all(&dir).unwrap();

        let path = dir.join("test_artifact.bin");
        let size = serialize_to_file(&original, &path, true).unwrap();
        assert!(size > 0);

        let recovered: TestPayload = deserialize_from_file(&path).unwrap();
        assert_eq!(original, recovered);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compressed_smaller_for_repetitive_data() {
        let mut payload = sample_payload();
        payload.values = vec![42; 10_000];

        let uncompressed = serialize_to_bytes(&payload, false).unwrap();
        let compressed = serialize_to_bytes(&payload, true).unwrap();
        assert!(
            compressed.len() < uncompressed.len(),
            "compressed ({}) should be smaller than uncompressed ({})",
            compressed.len(),
            uncompressed.len()
        );
    }
}
