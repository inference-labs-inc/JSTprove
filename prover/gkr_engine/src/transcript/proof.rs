use serdes::ExpSerde;

/// Proof. In the serialized mode.
#[derive(Debug, Clone, Default, PartialEq, ExpSerde)]
pub struct Proof {
    pub bytes: Vec<u8>,
}

impl Proof {
    #[inline(always)]
    pub fn append_u8_slice(&mut self, buffer: &[u8], size: usize) {
        self.bytes.extend_from_slice(&buffer[..size]);
    }
}
