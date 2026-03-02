use std::cmp;

use arith::Field;
use itertools::{chain, izip};
use rand::seq::index;
use serdes::ExpSerde;

use super::{OrionPCSError, OrionResult};

/*
 * IMPLEMENTATIONS FOR ORION EXPANDER GRAPH
 */

pub type DirectedEdge = usize;

pub type DirectedNeighboring = Vec<DirectedEdge>;

#[derive(Clone, Debug, Default, ExpSerde)]
pub struct OrionExpanderGraph {
    // L R vertices size book keeping:
    // keep track of message length (l), and "compressed" code length (r)
    pub l_vertices_size: usize,
    pub r_vertices_size: usize,

    // neighboring stands for all (weighted) connected vertices of a vertex.
    // In this context, the neighborings stands for the neighborings
    // of vertices in R set of the bipariate graph, which explains why it has
    // size of l_vertices_size, while each neighboring reserved r_vertices_size
    // capacity.
    pub neighborings: Vec<DirectedNeighboring>,
}

impl OrionExpanderGraph {
    pub fn new(
        l_vertices_size: usize,
        r_vertices_size: usize,
        expanding_degree: usize,
        mut rng: impl rand::RngCore,
    ) -> Self {
        let mut neighborings: Vec<DirectedNeighboring> =
            vec![Vec::with_capacity(l_vertices_size); r_vertices_size];

        (0..l_vertices_size).for_each(|l_index| {
            let random_r_vertices = index::sample(&mut rng, r_vertices_size, expanding_degree);

            random_r_vertices
                .iter()
                .for_each(|r_index| neighborings[r_index].push(l_index))
        });

        Self {
            neighborings,
            l_vertices_size,
            r_vertices_size,
        }
    }

    #[inline(always)]
    pub fn expander_mul<F: Field>(
        &self,
        l_vertices: &[F],
        r_vertices: &mut [F],
    ) -> OrionResult<()> {
        if l_vertices.len() != self.l_vertices_size || r_vertices.len() != self.r_vertices_size {
            return Err(OrionPCSError::ParameterUnmatchError);
        }

        izip!(r_vertices, &self.neighborings).for_each(|(ri, ni)| {
            *ri = ni.iter().map(|&edge_i| l_vertices[edge_i]).sum();
        });

        Ok(())
    }
}

/*
 * IMPLEMENTATIONS FOR ORION CODE FROM EXPANDER GRAPH
 */

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrionCodeParameter {
    // parameter for graph g0, that maps n -> (\alpha_g0 n)
    // \alpha_g0 should be ranging in (0, 1)
    pub alpha_g0: f64,
    pub degree_g0: usize,

    // parameter regarding graph generation for the code:
    // stopping condition when message is too short for the recursive code
    // in the next round.
    pub length_threshold_g0s: usize,

    // parameter for graph g1, let the message in the middle has length L,
    // then the graph g1 maps L -> (\alpha_g1 L)
    pub alpha_g1: f64,
    pub degree_g1: usize,

    // code's relateive distance
    pub hamming_weight: f64,
}

// NOTE: This instance of code derives from Orion paper Section 5.
pub const ORION_CODE_PARAMETER_INSTANCE: OrionCodeParameter = OrionCodeParameter {
    alpha_g0: 0.33,
    degree_g0: 6,

    length_threshold_g0s: 12,

    alpha_g1: 0.337,
    degree_g1: 6,

    hamming_weight: 0.055,
};

#[allow(clippy::doc_lazy_continuation)]
/// ACKNOWLEDGEMENT: on alphabet being F2 binary case, we appreciate the help from
/// - Section 18 in essential coding theory
/// https://cse.buffalo.edu/faculty/atri/courses/coding-theory/book/web-coding-book.pdf
///
/// - Notes from coding theory
/// https://www.cs.cmu.edu/~venkatg/teaching/codingtheory/notes/notes8.pdf
///
/// - Druk-Ishai 2014
/// https://dl.acm.org/doi/10.1145/2554797.2554815

#[derive(Clone, Debug, Default, ExpSerde)]
pub struct OrionExpanderGraphPositioned {
    pub graph: OrionExpanderGraph,

    pub input_starts: usize,
    pub output_starts: usize,
    pub output_ends: usize,
}

impl OrionExpanderGraphPositioned {
    #[inline(always)]
    pub fn new(
        input_starts: usize,
        output_starts: usize,
        output_ends: usize,
        expanding_degree: usize,
        mut rng: impl rand::RngCore,
    ) -> Self {
        Self {
            graph: OrionExpanderGraph::new(
                output_starts - input_starts,
                output_ends - output_starts + 1,
                expanding_degree,
                &mut rng,
            ),
            input_starts,
            output_starts,
            output_ends,
        }
    }

    #[inline(always)]
    pub fn expander_mul<F: Field>(&self, buffer: &mut [F], scratch: &mut [F]) -> OrionResult<()> {
        let input_ref = &buffer[self.input_starts..self.output_starts];
        let output_ref = &mut scratch[self.output_starts..self.output_ends + 1];

        self.graph.expander_mul(input_ref, output_ref)?;
        buffer[self.output_starts..self.output_ends + 1].copy_from_slice(output_ref);

        Ok(())
    }
}

// NOTE: The OrionCode here is representing an instance of Spielman code
// (Spielman96), that relies on 2 lists of expander graphs serving as
// error reduction code, and thus the linear error correction code derive
// from the parity matrices corresponding to these expander graphs.
#[derive(Clone, Debug, Default, ExpSerde)]
pub struct OrionCode {
    pub hamming_weight: f64,

    // empirical parameters for this instance of expander code on input/codeword
    pub msg_len: usize,
    pub codeword_len: usize,

    // g0s (affecting left side alphabets of the codeword)
    // generated from the largest to the smallest
    pub g0s: Vec<OrionExpanderGraphPositioned>,

    // g1s (affecting right side alphabets of the codeword)
    // generated from the smallest to the largest
    pub g1s: Vec<OrionExpanderGraphPositioned>,
}

pub type OrionCodeword<F> = Vec<F>;

impl OrionCode {
    pub fn new(params: OrionCodeParameter, msg_len: usize, mut rng: impl rand::RngCore) -> Self {
        // NOTE: sanity check - 1 / threshold_len > hamming_weight
        // as was part of Druk-Ishai-14 distance proof by induction
        assert!(1f64 / (params.length_threshold_g0s as f64) > params.hamming_weight);

        // NOTE: sanity check for both alpha_g0 and alpha_g1
        assert!(0f64 < params.alpha_g0 && params.alpha_g0 < 1f64);
        assert!(0f64 < params.alpha_g1 && params.alpha_g1 < 1f64);

        // NOTE: the real deal of code instance generation starts here
        let mut recursive_g0_output_starts: Vec<usize> = Vec::new();

        let mut g0s: Vec<OrionExpanderGraphPositioned> = Vec::new();
        let mut g1s: Vec<OrionExpanderGraphPositioned> = Vec::new();

        let mut g0_input_starts = 0;
        let mut g0_output_starts = msg_len;

        while g0_output_starts - g0_input_starts > params.length_threshold_g0s {
            let n = g0_output_starts - g0_input_starts;
            let g0_output_len = (n as f64 * params.alpha_g0).round() as usize;
            let degree_g0 = cmp::min(params.degree_g0, g0_output_len);

            g0s.push(OrionExpanderGraphPositioned::new(
                g0_input_starts,
                g0_output_starts,
                g0_output_starts + g0_output_len - 1,
                degree_g0,
                &mut rng,
            ));

            recursive_g0_output_starts.push(g0_output_starts);

            (g0_input_starts, g0_output_starts) =
                (g0_output_starts, g0_output_starts + g0_output_len);
        }

        // After g0s are generated, we generate g1s
        let mut g1_output_starts = g0_output_starts;

        while let Some(g1_input_starts) = recursive_g0_output_starts.pop() {
            let n = g1_output_starts - g1_input_starts;
            let g1_output_len = (n as f64 * params.alpha_g1).round() as usize;
            let degree_g1 = cmp::min(params.degree_g1, g1_output_len);

            g1s.push(OrionExpanderGraphPositioned::new(
                g1_input_starts,
                g1_output_starts,
                g1_output_starts + g1_output_len - 1,
                degree_g1,
                &mut rng,
            ));

            g1_output_starts += g1_output_len;
        }

        let codeword_len = g1_output_starts;
        Self {
            hamming_weight: params.hamming_weight,
            msg_len,
            codeword_len,
            g0s,
            g1s,
        }
    }

    #[inline(always)]
    pub fn code_len(&self) -> usize {
        self.codeword_len
    }

    #[inline(always)]
    pub fn msg_len(&self) -> usize {
        self.msg_len
    }

    #[inline(always)]
    pub fn hamming_weight(&self) -> f64 {
        self.hamming_weight
    }

    #[inline(always)]
    pub fn encode<F: Field>(&self, msg: &[F]) -> OrionResult<OrionCodeword<F>> {
        let mut codeword = vec![F::ZERO; self.code_len()];
        self.encode_in_place(msg, &mut codeword)?;
        Ok(codeword)
    }

    #[inline(always)]
    pub fn encode_in_place<F: Field>(&self, msg: &[F], buffer: &mut [F]) -> OrionResult<()> {
        if msg.len() != self.msg_len() || buffer.len() != self.code_len() {
            return Err(OrionPCSError::ParameterUnmatchError);
        }

        buffer[..self.msg_len()].copy_from_slice(msg);
        let mut scratch = vec![F::ZERO; self.code_len()];

        chain!(&self.g0s, &self.g1s).try_for_each(|g| g.expander_mul(buffer, &mut scratch))
    }
}
