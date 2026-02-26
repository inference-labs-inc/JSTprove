use anyhow::{bail, Result};

pub const ENVELOPE_MAGIC: [u8; 4] = *b"JST\x01";
pub const ENVELOPE_HEADER_LEN: usize = 20;

const FLAG_ZSTD: u32 = 1;

pub struct EnvelopeHeader {
    pub compressed: bool,
    pub payload_len: u64,
    pub crc32: u32,
}

pub fn write_envelope(payload: &[u8], compressed: bool) -> Vec<u8> {
    let crc = crc32fast::hash(payload);
    let flags: u32 = if compressed { FLAG_ZSTD } else { 0 };

    let mut out = Vec::with_capacity(ENVELOPE_HEADER_LEN + payload.len());
    out.extend_from_slice(&ENVELOPE_MAGIC);
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(payload);
    out
}

pub fn parse_header(buf: &[u8]) -> Result<EnvelopeHeader> {
    if buf.len() < ENVELOPE_HEADER_LEN {
        bail!(
            "envelope header too short: {} bytes (need {})",
            buf.len(),
            ENVELOPE_HEADER_LEN
        );
    }

    let flags = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    let payload_len = u64::from_le_bytes(buf[8..16].try_into().unwrap());
    let crc32 = u32::from_le_bytes(buf[16..20].try_into().unwrap());

    Ok(EnvelopeHeader {
        compressed: flags & FLAG_ZSTD != 0,
        payload_len,
        crc32,
    })
}

pub fn verify_crc(payload: &[u8], expected: u32) -> Result<()> {
    let actual = crc32fast::hash(payload);
    if actual != expected {
        bail!("CRC mismatch: expected {expected:#010x}, got {actual:#010x}");
    }
    Ok(())
}

pub const ZSTD_MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];

pub enum DetectedFormat {
    Envelope,
    LegacyZstd,
    RawMsgpack,
}

pub fn detect_format(data: &[u8]) -> DetectedFormat {
    if data.len() >= 4 && data[..4] == ENVELOPE_MAGIC {
        DetectedFormat::Envelope
    } else if data.len() >= 4 && data[..4] == ZSTD_MAGIC {
        DetectedFormat::LegacyZstd
    } else {
        DetectedFormat::RawMsgpack
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_roundtrip() {
        let payload = b"hello msgpack world";
        let envelope = write_envelope(payload, false);

        assert_eq!(&envelope[..4], &ENVELOPE_MAGIC);
        assert_eq!(envelope.len(), ENVELOPE_HEADER_LEN + payload.len());

        let header = parse_header(&envelope).unwrap();
        assert!(!header.compressed);
        assert_eq!(header.payload_len, payload.len() as u64);

        let body = &envelope[ENVELOPE_HEADER_LEN..];
        verify_crc(body, header.crc32).unwrap();
        assert_eq!(body, payload);
    }

    #[test]
    fn envelope_roundtrip_compressed_flag() {
        let payload = b"compressed data";
        let envelope = write_envelope(payload, true);

        let header = parse_header(&envelope).unwrap();
        assert!(header.compressed);
    }

    #[test]
    fn crc_mismatch_detected() {
        let payload = b"original";
        let envelope = write_envelope(payload, false);
        let header = parse_header(&envelope).unwrap();

        let tampered = b"tampered";
        assert!(verify_crc(tampered, header.crc32).is_err());
    }

    #[test]
    fn format_detection() {
        assert!(matches!(
            detect_format(&ENVELOPE_MAGIC),
            DetectedFormat::Envelope
        ));
        assert!(matches!(
            detect_format(&ZSTD_MAGIC),
            DetectedFormat::LegacyZstd
        ));
        assert!(matches!(
            detect_format(b"\x92\xa5hello"),
            DetectedFormat::RawMsgpack
        ));
        assert!(matches!(detect_format(b""), DetectedFormat::RawMsgpack));
    }
}
