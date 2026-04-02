// Copyright 2024-2025 Irreducible Inc.

use std::slice;

use bytes::{Buf, BufMut, buf::UninitSlice};

use super::Challenger;

#[derive(Debug, Default, Clone)]
pub struct FiatShamirBuf<Inner, Challenger> {
    pub buffer: Inner,
    pub challenger: Challenger,
}

impl<Inner: Buf, Challenger_: Challenger> Buf for FiatShamirBuf<Inner, Challenger_> {
    fn remaining(&self) -> usize {
        self.buffer.remaining()
    }

    fn chunk(&self) -> &[u8] {
        self.buffer.chunk()
    }

    fn advance(&mut self, cnt: usize) {
        assert!(cnt <= self.buffer.remaining());
        // Get the slice that was written to the inner buf, observe that and advance
        let readable = self.buffer.chunk();
        // Because our internal buffer is created from vec, this should never happen.
        assert!(cnt <= readable.len());
        self.challenger.observer().put_slice(&readable[..cnt]);
        self.buffer.advance(cnt);
    }
}

unsafe impl<Inner: BufMut, Challenger_: Challenger> BufMut for FiatShamirBuf<Inner, Challenger_> {
    fn remaining_mut(&self) -> usize {
        self.buffer.remaining_mut()
    }

    unsafe fn advance_mut(&mut self, cnt: usize) {
        assert!(cnt <= self.buffer.remaining_mut());
        let written = self.buffer.chunk_mut();
        // Because out internal buffer is BytesMut cnt <= written.len(), but adding as per
        // implementation notes
        assert!(cnt <= written.len());

        // NOTE: This is the unsafe part, you are reading the next cnt bytes on the assumption that
        // caller has ensured us the next cnt bytes are initialized.
        let written: &[u8] = unsafe { slice::from_raw_parts(written.as_mut_ptr(), cnt) };

        self.challenger.observer().put_slice(written);
        unsafe {
            self.buffer.advance_mut(cnt);
        }
    }

    fn chunk_mut(&mut self) -> &mut UninitSlice {
        self.buffer.chunk_mut()
    }
}
