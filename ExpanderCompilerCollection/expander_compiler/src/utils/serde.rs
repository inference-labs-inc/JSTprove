use std::{
    collections::HashMap,
    hash::Hash,
    io::{Error as IoError, Read, Write},
};

use ethnum::U256;
pub trait Serde: Sized {
    fn serialize_into<W: Write>(&self, writer: W) -> Result<(), IoError>;
    fn deserialize_from<R: Read>(reader: R) -> Result<Self, IoError>;
}

impl Serde for usize {
    fn serialize_into<W: Write>(&self, mut writer: W) -> Result<(), IoError> {
        writer.write_all(&(*self as u64).to_le_bytes())
    }

    fn deserialize_from<R: Read>(mut reader: R) -> Result<Self, IoError> {
        let mut u = [0u8; 8];
        reader.read_exact(&mut u)?;
        Ok(u64::from_le_bytes(u) as usize)
    }
}

impl Serde for u8 {
    fn serialize_into<W: Write>(&self, mut writer: W) -> Result<(), IoError> {
        writer.write_all(&[*self])
    }

    fn deserialize_from<R: Read>(mut reader: R) -> Result<Self, IoError> {
        let mut u = [0u8; 1];
        reader.read_exact(&mut u)?;
        Ok(u[0])
    }
}

impl Serde for bool {
    fn serialize_into<W: Write>(&self, mut writer: W) -> Result<(), IoError> {
        (*self as u8).serialize_into(&mut writer)
    }

    fn deserialize_from<R: Read>(mut reader: R) -> Result<Self, IoError> {
        u8::deserialize_from(&mut reader).map(|u| u != 0)
    }
}

impl<T: Serde> Serde for Vec<T> {
    fn serialize_into<W: Write>(&self, mut writer: W) -> Result<(), IoError> {
        self.len().serialize_into(&mut writer)?;
        for item in self {
            item.serialize_into(&mut writer)?;
        }
        Ok(())
    }

    fn deserialize_from<R: Read>(mut reader: R) -> Result<Self, IoError> {
        let len = usize::deserialize_from(&mut reader)?;
        let mut vec = Vec::with_capacity(len);
        for _ in 0..len {
            vec.push(T::deserialize_from(&mut reader)?);
        }
        Ok(vec)
    }
}

impl<K: Serde + Eq + Hash, V: Serde> Serde for HashMap<K, V> {
    fn serialize_into<W: Write>(&self, mut writer: W) -> Result<(), IoError> {
        self.len().serialize_into(&mut writer)?;
        for (k, v) in self.iter() {
            k.serialize_into(&mut writer)?;
            v.serialize_into(&mut writer)?;
        }
        Ok(())
    }

    fn deserialize_from<R: Read>(mut reader: R) -> Result<Self, IoError> {
        let len = usize::deserialize_from(&mut reader)?;
        let mut map = HashMap::with_capacity(len);
        for _ in 0..len {
            let k = K::deserialize_from(&mut reader)?;
            let v = V::deserialize_from(&mut reader)?;
            map.insert(k, v);
        }
        Ok(map)
    }
}

impl Serde for U256 {
    fn serialize_into<W: Write>(&self, mut writer: W) -> Result<(), IoError> {
        writer.write_all(&self.to_le_bytes())
    }

    fn deserialize_from<R: Read>(mut reader: R) -> Result<Self, IoError> {
        let mut bytes = [0u8; 32];
        reader.read_exact(&mut bytes)?;
        Ok(Self::from_le_bytes(bytes))
    }
}
