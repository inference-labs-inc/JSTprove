use std::{
    collections::HashMap,
    hash::Hash,
    io::{Read, Write},
};

use ethnum::U256;
use halo2curves::{
    bn256::{Fr, G1Affine, G2Affine},
    group::GroupEncoding,
};

use crate::{exp_serde_for_generic_slices, exp_serde_for_number, SerdeError, SerdeResult};

/// Serde for Arithmetic types such as field and group operations
pub trait ExpSerde: Sized {
    /// serialize self into bytes
    fn serialize_into<W: Write>(&self, writer: W) -> SerdeResult<()>;

    /// deserialize bytes into field
    fn deserialize_from<R: Read>(reader: R) -> SerdeResult<Self>;
}

impl ExpSerde for () {
    fn serialize_into<W: std::io::Write>(&self, _writer: W) -> SerdeResult<()> {
        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(_reader: R) -> SerdeResult<Self> {
        Ok(())
    }
}

exp_serde_for_number!(u64, 8);
exp_serde_for_number!(usize, 8);
exp_serde_for_number!(u8, 1);
exp_serde_for_number!(f64, 8);
exp_serde_for_number!(u128, 16);
exp_serde_for_number!(u32, 4);
exp_serde_for_number!(U256, 32);

// macro serdes for [S: N] where S: ExpSerde
exp_serde_for_generic_slices!(2);
exp_serde_for_generic_slices!(3);
exp_serde_for_generic_slices!(4);
exp_serde_for_generic_slices!(8);
exp_serde_for_generic_slices!(16);
exp_serde_for_generic_slices!(32);
exp_serde_for_generic_slices!(64);

impl ExpSerde for bool {
    fn serialize_into<W: Write>(&self, writer: W) -> SerdeResult<()> {
        (*self as u8).serialize_into(writer)
    }

    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        u8::deserialize_from(&mut reader).map(|u| u != 0)
    }
}

impl<V: ExpSerde> ExpSerde for Vec<V> {
    fn serialize_into<W: Write>(&self, mut writer: W) -> SerdeResult<()> {
        self.len().serialize_into(&mut writer)?;
        for v in self.iter() {
            v.serialize_into(&mut writer)?;
        }
        Ok(())
    }

    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let mut v = Self::default();
        let len = usize::deserialize_from(&mut reader)?;
        for _ in 0..len {
            v.push(V::deserialize_from(&mut reader)?);
        }
        Ok(v)
    }
}

impl ExpSerde for Fr {
    #[inline(always)]
    fn serialize_into<W: Write>(&self, mut writer: W) -> SerdeResult<()> {
        writer.write_all(self.to_bytes().as_ref())?;
        Ok(())
    }

    #[inline(always)]
    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let mut buffer = [0u8; 32];
        reader.read_exact(&mut buffer)?;
        match Fr::from_bytes(&buffer).into_option() {
            Some(v) => Ok(v),
            None => Err(SerdeError::DeserializeError),
        }
    }
}

impl ExpSerde for G1Affine {
    fn serialize_into<W: Write>(&self, writer: W) -> SerdeResult<()> {
        let mut buf = [0u8; 32];
        assert!(self.to_bytes().as_ref().len() == 32);
        buf.copy_from_slice(self.to_bytes().as_ref());
        buf.serialize_into(writer)
    }

    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let mut buf = [0u8; 32];
        reader.read_exact(&mut buf)?;

        let mut encoding = <Self as GroupEncoding>::Repr::default();
        encoding.as_mut().copy_from_slice(buf.as_ref());
        match G1Affine::from_bytes(&encoding).into_option() {
            Some(a) => Ok(a),
            None => Err(SerdeError::DeserializeError),
        }
    }
}

impl ExpSerde for G2Affine {
    fn serialize_into<W: Write>(&self, writer: W) -> SerdeResult<()> {
        let mut buf = [0u8; 64];
        assert!(self.to_bytes().as_ref().len() == 64);
        buf.copy_from_slice(self.to_bytes().as_ref());
        buf.serialize_into(writer)
    }

    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let mut buf = [0u8; 64];
        reader.read_exact(&mut buf)?;

        let mut encoding = <Self as GroupEncoding>::Repr::default();
        encoding.as_mut().copy_from_slice(buf.as_ref());
        match G2Affine::from_bytes(&encoding).into_option() {
            Some(a) => Ok(a),
            None => Err(SerdeError::DeserializeError),
        }
    }
}

impl<T1: ExpSerde, T2: ExpSerde> ExpSerde for (T1, T2) {
    fn serialize_into<W: Write>(&self, mut writer: W) -> SerdeResult<()> {
        self.0.serialize_into(&mut writer)?;
        self.1.serialize_into(&mut writer)?;
        Ok(())
    }

    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let t1 = T1::deserialize_from(&mut reader)?;
        let t2 = T2::deserialize_from(&mut reader)?;
        Ok((t1, t2))
    }
}

impl<K: ExpSerde + Eq + Hash, V: ExpSerde> ExpSerde for HashMap<K, V> {
    fn serialize_into<W: Write>(&self, mut writer: W) -> SerdeResult<()> {
        self.len().serialize_into(&mut writer)?;
        for (k, v) in self.iter() {
            k.serialize_into(&mut writer)?;
            v.serialize_into(&mut writer)?;
        }
        Ok(())
    }

    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
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

impl<T: ExpSerde> ExpSerde for Option<T> {
    fn serialize_into<W: Write>(&self, mut writer: W) -> SerdeResult<()> {
        match self {
            Some(v) => {
                true.serialize_into(&mut writer)?;
                v.serialize_into(&mut writer)
            }
            None => false.serialize_into(&mut writer),
        }
    }

    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let has_value = bool::deserialize_from(&mut reader)?;
        if has_value {
            Ok(Some(T::deserialize_from(&mut reader)?))
        } else {
            Ok(None)
        }
    }
}

impl ExpSerde for String {
    fn serialize_into<W: Write>(&self, mut writer: W) -> SerdeResult<()> {
        let bytes = self.as_bytes();
        bytes.len().serialize_into(&mut writer)?;
        writer.write_all(bytes)?;
        Ok(())
    }
    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let len = usize::deserialize_from(&mut reader)?;
        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf)?;
        String::from_utf8(buf).map_err(|_| SerdeError::DeserializeError)
    }
}
