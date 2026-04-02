// Copyright 2024-2025 Irreducible Inc.

use std::{fs::File, io::Write, iter::repeat_with, slice};

use binius_field::Field;
use binius_utils::{DeserializeBytes, SerializeBytes};
use bytes::{Buf, BufMut, Bytes, BytesMut};

use super::{
    error::Error,
    fiat_shamir::{Challenger, FiatShamirBuf},
};
use crate::fiat_shamir::{CanSample, CanSampleBits, sample_bits_reader};

/// Configuration options for transcript behavior
#[derive(Debug, Clone, Copy)]
pub struct Options {
    /// Whether to enable debug assertions
    pub debug_assertions: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            debug_assertions: cfg!(debug_assertions),
        }
    }
}

/// Verifier transcript over some Challenger that reads from the internal tape and `CanSample<F:
/// Field>`
///
/// You must manually call the destructor with `finalize()` to check anything that's written is
/// fully read out
#[derive(Debug, Clone)]
pub struct VerifierTranscript<Challenger> {
    combined: FiatShamirBuf<Bytes, Challenger>,
    options: Options,
}

impl<Challenger_: Challenger> VerifierTranscript<Challenger_> {
    pub fn new(challenger: Challenger_, vec: Vec<u8>) -> Self {
        Self::with_opts(challenger, vec, Options::default())
    }

    pub fn with_opts(challenger: Challenger_, vec: Vec<u8>, options: Options) -> Self {
        Self {
            combined: FiatShamirBuf {
                buffer: Bytes::from(vec),
                challenger,
            },
            options,
        }
    }
}

impl<Challenger_: Challenger> VerifierTranscript<Challenger_> {
    pub fn finalize(self) -> Result<(), Error> {
        if self.combined.buffer.has_remaining() {
            return Err(Error::TranscriptNotEmpty {
                remaining: self.combined.buffer.remaining(),
            });
        }
        Ok(())
    }

    /// Returns a writable buffer that only observes the data written, without reading it from the
    /// proof tape.
    ///
    /// This method should be used to observe the input statement.
    pub fn observe<'a, 'b>(&'a mut self) -> TranscriptWriter<'b, impl BufMut + 'b>
    where
        'a: 'b,
    {
        TranscriptWriter {
            buffer: self.combined.challenger.observer(),
            options: self.options,
        }
    }

    /// Returns a readable buffer that only reads the data from the proof tape, without observing
    /// it.
    ///
    /// This method should only be used to read advice that was previously written to the transcript
    /// as an observed message.
    pub fn decommitment(&mut self) -> TranscriptReader<'_, impl Buf + '_> {
        TranscriptReader {
            buffer: &mut self.combined.buffer,
            options: self.options,
        }
    }

    /// Returns a readable buffer that observes the data read.
    ///
    /// This method should be used by default to read verifier messages in an interactive protocol.
    pub fn message<'a, 'b>(&'a mut self) -> TranscriptReader<'b, impl Buf>
    where
        'a: 'b,
    {
        TranscriptReader {
            buffer: &mut self.combined,
            options: self.options,
        }
    }
}

// Useful warnings to see if we are neglecting to read any advice or transcript entirely
impl<Challenger> Drop for VerifierTranscript<Challenger> {
    fn drop(&mut self) {
        if self.combined.buffer.has_remaining() {
            tracing::warn!(
                "Transcript reader is not fully read out: {:?} bytes left",
                self.combined.buffer.remaining()
            )
        }
    }
}

impl<F, Challenger_> CanSample<F> for VerifierTranscript<Challenger_>
where
    F: Field,
    Challenger_: Challenger,
{
    fn sample(&mut self) -> F {
        DeserializeBytes::deserialize(self.combined.challenger.sampler())
            .expect("challenger has infinite buffer")
    }
}

impl<Challenger_> CanSampleBits<u32> for VerifierTranscript<Challenger_>
where
    Challenger_: Challenger,
{
    fn sample_bits(&mut self, bits: usize) -> u32 {
        sample_bits_reader(self.combined.challenger.sampler(), bits)
    }
}

pub struct TranscriptReader<'a, B: Buf> {
    buffer: &'a mut B,
    options: Options,
}

impl<B: Buf> TranscriptReader<'_, B> {
    pub const fn buffer(&mut self) -> &mut B {
        self.buffer
    }

    pub fn read<T: DeserializeBytes>(&mut self) -> Result<T, Error> {
        T::deserialize(self.buffer()).map_err(Into::into)
    }

    pub fn read_vec<T: DeserializeBytes>(&mut self, n: usize) -> Result<Vec<T>, Error> {
        let mut buffer = self.buffer();
        repeat_with(move || T::deserialize(&mut buffer).map_err(Into::into))
            .take(n)
            .collect()
    }

    pub fn read_bytes(&mut self, buf: &mut [u8]) -> Result<(), Error> {
        let buffer = self.buffer();
        if buffer.remaining() < buf.len() {
            return Err(Error::NotEnoughBytes);
        }
        buffer.copy_to_slice(buf);
        Ok(())
    }

    pub fn read_scalar<F: Field>(&mut self) -> Result<F, Error> {
        let mut out = F::default();
        self.read_scalar_slice_into(slice::from_mut(&mut out))?;
        Ok(out)
    }

    pub fn read_scalar_slice_into<F: Field>(&mut self, buf: &mut [F]) -> Result<(), Error> {
        let mut buffer = self.buffer();
        for elem in buf {
            *elem = DeserializeBytes::deserialize(&mut buffer)?;
        }
        Ok(())
    }

    pub fn read_scalar_slice<F: Field>(&mut self, len: usize) -> Result<Vec<F>, Error> {
        let mut elems = vec![F::default(); len];
        self.read_scalar_slice_into(&mut elems)?;
        Ok(elems)
    }

    pub fn read_debug(&mut self, msg: &str) {
        if self.options.debug_assertions {
            let msg_bytes = msg.as_bytes();
            let mut buffer = vec![0; msg_bytes.len()];
            assert!(self.read_bytes(&mut buffer).is_ok());
            assert_eq!(msg_bytes, buffer);
        }
    }
}

/// Prover transcript over some Challenger that writes to the internal tape and `CanSample<F:
/// Field>`
///
/// A Transcript is an abstraction over Fiat-Shamir so the prover and verifier can send and receive
/// data.
#[derive(Debug, Clone)]
pub struct ProverTranscript<Challenger> {
    combined: FiatShamirBuf<BytesMut, Challenger>,
    options: Options,
}

impl<Challenger_: Challenger> ProverTranscript<Challenger_> {
    /// Creates a new prover transcript.
    ///
    /// By default debug assertions are set to the feature flag `debug_assertions`.
    pub fn new(challenger: Challenger_) -> Self {
        Self::with_opts(challenger, Options::default())
    }

    pub fn with_opts(challenger: Challenger_, options: Options) -> Self {
        Self {
            combined: FiatShamirBuf {
                buffer: BytesMut::default(),
                challenger,
            },
            options,
        }
    }
}

impl<Challenger_: Default + Challenger> ProverTranscript<Challenger_> {
    pub fn into_verifier(self) -> VerifierTranscript<Challenger_> {
        let options = self.options;
        let transcript = self.finalize();

        VerifierTranscript::with_opts(Challenger_::default(), transcript, options)
    }
}

impl<Challenger_: Default + Challenger> Default for ProverTranscript<Challenger_> {
    fn default() -> Self {
        Self::new(Challenger_::default())
    }
}

impl<Challenger_: Challenger> ProverTranscript<Challenger_> {
    pub fn finalize(self) -> Vec<u8> {
        let transcript = self.combined.buffer.to_vec();

        // Emit proof size as a tracing event
        let proof_size_bytes = transcript.len();
        tracing::event!(
            name: "proof_size",
            tracing::Level::INFO,
            category = "metrics",
            proof_size_bytes = proof_size_bytes,
        );

        // Dumps the transcript to the path set in the BINIUS_DUMP_PROOF env variable.
        if let Ok(path) = std::env::var("BINIUS_DUMP_PROOF") {
            let path = if cfg!(test) {
                // Because tests may run simultaneously, each test includes its name in the file
                // name to avoid collisions.
                let current_thread = std::thread::current();
                let test_name = current_thread.name().unwrap_or("unknown");
                // Adjust "./" to "../../" to ensure files are saved in the project root rather than
                // the package root.
                let path = if let Some(stripped) = path.strip_prefix("./") {
                    format!("../../{stripped}",)
                } else {
                    path
                };
                std::fs::create_dir_all(&path)
                    .unwrap_or_else(|_| panic!("Failed to create directories for path: {path}",));
                format!("{path}/{test_name}.bin")
            } else {
                path
            };

            let mut file = File::create(&path)
                .unwrap_or_else(|_| panic!("Failed to create proof dump file: {path}"));
            file.write_all(&transcript)
                .expect("Failed to write proof to dump file");
        }
        transcript
    }

    /// Returns a writeable buffer that only observes the data written, without writing it to the
    /// proof tape.
    ///
    /// This method should be used to observe the input statement.
    pub fn observe<'a, 'b>(&'a mut self) -> TranscriptWriter<'b, impl BufMut + 'b>
    where
        'a: 'b,
    {
        TranscriptWriter {
            buffer: self.combined.challenger.observer(),
            options: self.options,
        }
    }

    /// Returns a writeable buffer that only writes the data to the proof tape, without observing
    /// it.
    ///
    /// This method should only be used to write openings of commitments that were already written
    /// to the transcript as an observed message. For example, in the FRI protocol, the prover sends
    /// a Merkle tree root as a commitment, and later sends leaf openings. The leaf openings should
    /// be written using [`Self::decommitment`] because they are verified with respect to the
    /// previously sent Merkle root.
    pub fn decommitment(&mut self) -> TranscriptWriter<'_, impl BufMut> {
        TranscriptWriter {
            buffer: &mut self.combined.buffer,
            options: self.options,
        }
    }

    /// Returns a writeable buffer that observes the data written and writes it to the proof tape.
    ///
    /// This method should be used by default to write prover messages in an interactive protocol.
    pub fn message<'a, 'b>(&'a mut self) -> TranscriptWriter<'b, impl BufMut>
    where
        'a: 'b,
    {
        TranscriptWriter {
            buffer: &mut self.combined,
            options: self.options,
        }
    }
}

pub struct TranscriptWriter<'a, B: BufMut> {
    buffer: &'a mut B,
    options: Options,
}

impl<B: BufMut> TranscriptWriter<'_, B> {
    pub const fn buffer(&mut self) -> &mut B {
        self.buffer
    }

    pub fn write<T: SerializeBytes>(&mut self, value: &T) {
        self.proof_size_event_wrapper(move |buffer| {
            value.serialize(buffer).expect("TODO: propagate error");
        });
    }

    pub fn write_slice<T: SerializeBytes>(&mut self, values: &[T]) {
        self.proof_size_event_wrapper(move |buffer| {
            for value in values {
                value
                    .serialize(&mut *buffer)
                    .expect("TODO: propagate error");
            }
        });
    }

    pub fn write_bytes(&mut self, data: &[u8]) {
        self.proof_size_event_wrapper(|buffer| {
            buffer.put_slice(data);
        });
    }

    pub fn write_scalar<F: Field>(&mut self, f: F) {
        self.write_scalar_slice(slice::from_ref(&f));
    }

    pub fn write_scalar_iter<F: Field>(&mut self, it: impl IntoIterator<Item = F>) {
        self.proof_size_event_wrapper(move |buffer| {
            for elem in it {
                SerializeBytes::serialize(&elem, &mut *buffer).expect("TODO: propagate error");
            }
        });
    }

    pub fn write_scalar_slice<F: Field>(&mut self, elems: &[F]) {
        self.write_scalar_iter(elems.iter().copied());
    }

    pub fn write_debug(&mut self, msg: &str) {
        if self.options.debug_assertions {
            self.write_bytes(msg.as_bytes())
        }
    }

    fn proof_size_event_wrapper<F: FnOnce(&mut B)>(&mut self, f: F) {
        let buffer = self.buffer();
        let start_bytes = buffer.remaining_mut();
        f(buffer);
        let end_bytes = buffer.remaining_mut();
        tracing::event!(
            name: "incremental_proof_size",
            tracing::Level::TRACE,
            counter=true,
            incremental=true,
            value=start_bytes - end_bytes,
        );
    }
}

impl<F, Challenger_> CanSample<F> for ProverTranscript<Challenger_>
where
    F: Field,
    Challenger_: Challenger,
{
    fn sample(&mut self) -> F {
        DeserializeBytes::deserialize(self.combined.challenger.sampler())
            .expect("challenger has infinite buffer")
    }
}

impl<Challenger_> CanSampleBits<u32> for ProverTranscript<Challenger_>
where
    Challenger_: Challenger,
{
    fn sample_bits(&mut self, bits: usize) -> u32 {
        sample_bits_reader(self.combined.challenger.sampler(), bits)
    }
}

#[cfg(test)]
mod tests {
    use binius_field::BinaryField128bGhash as B128;
    use sha2::Sha256;

    use super::*;
    use crate::fiat_shamir::{CanSample, HasherChallenger};

    #[test]
    fn test_transcript_interactions() {
        let mut prover_transcript = ProverTranscript::new(HasherChallenger::<Sha256>::default());

        // Write messages using message()
        prover_transcript
            .message()
            .write_scalar(B128::new(0x11111111222222223333333344444444));
        prover_transcript
            .message()
            .write_scalar(B128::new(0xAAAAAAAABBBBBBBBCCCCCCCCDDDDDDDD));

        // Write decommitment (not observed)
        prover_transcript
            .decommitment()
            .write_scalar(B128::new(0x5555555566666666777777778888888));

        // Write observed data
        prover_transcript
            .observe()
            .write_scalar(B128::new(0xFFFFFFFFEEEEEEEEDDDDDDDDCCCCCCCC));

        // Sample a challenge
        let sampled_challenge: B128 = prover_transcript.sample();

        // Convert to verifier transcript
        let mut verifier_transcript = prover_transcript.into_verifier();

        // Read messages
        let msg1: B128 = verifier_transcript.message().read_scalar().unwrap();
        let msg2: B128 = verifier_transcript.message().read_scalar().unwrap();
        assert_eq!(msg1, B128::new(0x11111111222222223333333344444444));
        assert_eq!(msg2, B128::new(0xAAAAAAAABBBBBBBBCCCCCCCCDDDDDDDD));

        // Read decommitment
        let decommit: B128 = verifier_transcript.decommitment().read_scalar().unwrap();
        assert_eq!(decommit, B128::new(0x5555555566666666777777778888888));

        // Observe the same data (doesn't read from tape)
        verifier_transcript
            .observe()
            .write_scalar(B128::new(0xFFFFFFFFEEEEEEEEDDDDDDDDCCCCCCCC));

        // Sample should produce the same challenge
        let verifier_challenge: B128 = verifier_transcript.sample();
        assert_eq!(verifier_challenge, sampled_challenge);

        // Check that transcript is empty
        verifier_transcript.finalize().unwrap();
    }

    #[test]
    fn test_transcript_debug() {
        let options = Options {
            debug_assertions: true,
        };
        let mut transcript =
            ProverTranscript::with_opts(HasherChallenger::<Sha256>::default(), options);

        transcript.message().write_debug("test_transcript_debug");
        transcript
            .into_verifier()
            .message()
            .read_debug("test_transcript_debug");
    }

    #[test]
    #[should_panic]
    fn test_transcript_debug_fail() {
        let options = Options {
            debug_assertions: true,
        };
        let mut transcript =
            ProverTranscript::with_opts(HasherChallenger::<Sha256>::default(), options);

        transcript.message().write_debug("test_transcript_debug");
        transcript
            .into_verifier()
            .message()
            .read_debug("test_transcript_debug_should_fail");
    }
}
