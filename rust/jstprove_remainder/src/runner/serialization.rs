use anyhow::Result;
use serde::{Serialize, de::DeserializeOwned};

const ZSTD_COMPRESSION_LEVEL: i32 = 3;
const ZSTD_MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];

pub fn serialize_to_file<T: Serialize>(value: &T, path: &std::path::Path, compress: bool) -> Result<usize> {
    let serialized = bincode::serialize(value)?;
    let bytes = if compress {
        zstd::encode_all(serialized.as_slice(), ZSTD_COMPRESSION_LEVEL)?
    } else {
        serialized
    };
    std::fs::write(path, &bytes)?;
    Ok(bytes.len())
}

pub fn deserialize_from_file<T: DeserializeOwned>(path: &std::path::Path) -> Result<T> {
    let raw = std::fs::read(path)?;
    let decompressed = if raw.len() >= 4 && raw[..4] == ZSTD_MAGIC {
        zstd::decode_all(raw.as_slice())?
    } else {
        raw
    };
    let value: T = bincode::deserialize(&decompressed)?;
    Ok(value)
}
