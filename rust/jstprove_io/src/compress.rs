use std::borrow::Cow;
use std::io::{self, BufWriter, Read, Write};

use anyhow::Result;

use crate::envelope::ZSTD_MAGIC;

pub const ZSTD_COMPRESSION_LEVEL: i32 = 3;

pub fn compress_bytes(data: &[u8]) -> Result<Vec<u8>> {
    Ok(zstd::encode_all(data, ZSTD_COMPRESSION_LEVEL)?)
}

pub fn decompress_bytes(data: &[u8]) -> Result<Vec<u8>> {
    Ok(zstd::decode_all(data)?)
}

pub fn maybe_compress_bytes(data: Vec<u8>, compress: bool) -> Result<Vec<u8>> {
    if compress {
        compress_bytes(&data)
    } else {
        Ok(data)
    }
}

pub fn auto_decompress_bytes(data: &[u8]) -> Result<Cow<'_, [u8]>> {
    if data.len() >= 4 && data[..4] == ZSTD_MAGIC {
        decompress_bytes(data).map(Cow::Owned)
    } else {
        Ok(Cow::Borrowed(data))
    }
}

pub enum MaybeCompressed {
    Compressed(zstd::stream::write::Encoder<'static, BufWriter<std::fs::File>>),
    Plain(BufWriter<std::fs::File>),
}

impl Write for MaybeCompressed {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Self::Compressed(enc) => enc.write(buf),
            Self::Plain(w) => w.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Self::Compressed(enc) => enc.flush(),
            Self::Plain(w) => w.flush(),
        }
    }
}

impl MaybeCompressed {
    pub fn finish(self) -> io::Result<()> {
        match self {
            Self::Compressed(enc) => {
                enc.finish()?;
                Ok(())
            }
            Self::Plain(mut w) => w.flush(),
        }
    }
}

pub fn compressed_writer(file: std::fs::File, compress: bool) -> Result<MaybeCompressed> {
    if compress {
        let encoder =
            zstd::stream::write::Encoder::new(BufWriter::new(file), ZSTD_COMPRESSION_LEVEL)?;
        Ok(MaybeCompressed::Compressed(encoder))
    } else {
        Ok(MaybeCompressed::Plain(BufWriter::new(file)))
    }
}

pub fn auto_reader(file: std::fs::File) -> Result<Box<dyn Read>> {
    use std::io::BufReader;

    let mut reader = BufReader::new(file);
    let mut magic = [0u8; 4];

    match reader.read_exact(&mut magic) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
            let chain: Box<dyn Read> = Box::new(io::Cursor::new(magic).chain(reader));
            return Ok(chain);
        }
        Err(e) => return Err(e.into()),
    }

    if magic == ZSTD_MAGIC {
        let chain = io::Cursor::new(magic).chain(reader);
        let decoder = zstd::stream::read::Decoder::new(chain)?;
        Ok(Box::new(decoder))
    } else {
        Ok(Box::new(io::Cursor::new(magic).chain(reader)))
    }
}
