use anyhow::Result;
use serde::{de::DeserializeOwned, Serialize};

const ZSTD_COMPRESSION_LEVEL: i32 = 3;
const ZSTD_MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];

pub fn serialize_to_file<T: Serialize>(
    value: &T,
    path: &std::path::Path,
    compress: bool,
) -> Result<usize> {
    let serialized = rmp_serde::to_vec_named(value)?;
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
    let value: T = rmp_serde::from_slice(&decompressed)?;
    Ok(value)
}

pub fn write_msgpack_stdout<T: Serialize>(value: &T) -> Result<()> {
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    value
        .serialize(&mut rmp_serde::Serializer::new(&mut lock).with_struct_map())
        .map_err(|e| anyhow::anyhow!("msgpack stdout serialization failed: {e}"))?;
    Ok(())
}
