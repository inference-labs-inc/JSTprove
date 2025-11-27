use std::collections::{HashMap, HashSet};

/// External crate imports
use crate::circuit_functions::utils::{PatternError, onnx_types::ONNXLayer};
use strum::IntoEnumIterator;
use strum_macros::{AsRefStr, EnumIter};

/*

Pattern matching of layers

*/

type PatternMatchResult = Result<Option<(PatternRegistry, Vec<String>, Vec<String>)>, PatternError>;
// type PatternMatchResult = Result<Option<(GraphPattern, Vec<String>, Vec<String>)>, PatternError>;
/// Attempts to optimize a sequence of layers by skipping redundant ones.
///
/// Examines the provided `optimization_match` for consistent patterns and validates
/// that the outputs of the first layer in each match align with the expected `outputs`.
/// If all validations succeed, returns the common pattern, the deduplicated outputs
/// from the final layers, and the names of the skipped layers.
///
/// # Arguments
/// - `optimization_match`: Optional reference to a vector of `OptimizationMatch`
///   structs describing candidate matches.
/// - `outputs`: Slice of expected output names for validation.
///
/// # Returns
/// - `Ok(Some((pattern, new_outputs, skipped_layers)))` on success, where:
///   - `pattern` is the common `GraphPattern` shared by all matches.
///   - `new_outputs` is a deduplicated set of outputs from the final layers.
///   - `skipped_layers` is the list of layer names that were skipped.
/// - `Ok(None)` if no optimization matches were provided.
///
/// # Errors
/// - [`PatternError::EmptyMatch`] if:
///   - `optimization_match` is non-empty but contains no elements, or
///   - a match has no layers.
/// - [`PatternError::InconsistentPattern`] if matches contain different patterns.
/// - [`PatternError::OutputMismatch`] if the outputs of the first layer in a match
///   do not align with the expected `outputs`.
pub fn optimization_skip_layers(
    optimization_match: Option<&Vec<OptimizationMatch>>,
    outputs: &[String],
) -> PatternMatchResult {
    match optimization_match {
        Some(opt) => {
            let pattern = opt.first().ok_or(PatternError::EmptyMatch)?.pattern.clone();

            let mut new_outputs = Vec::new();
            let mut skipped_layers: Vec<String> = Vec::new();

            // Loop through all potential branches
            for opt_match in opt {
                // Assert all the patterns are the same
                if pattern.as_graph_pattern().name != opt_match.pattern.as_graph_pattern().name {
                    return Err(PatternError::InconsistentPattern {
                        expected: pattern.as_graph_pattern().name.to_string(),
                        got: opt_match.pattern.as_graph_pattern().name.to_string(),
                    });
                }
                // Get final layer of pattern
                let layers = opt_match.layers.clone();
                let final_layer = layers.last().cloned().ok_or(PatternError::EmptyMatch)?; // or a more specific variant

                let first_layer = layers.first().cloned().ok_or(PatternError::EmptyMatch)?;

                // Assert outputs match
                eprintln!("{:?}", first_layer.outputs);
                eprintln!("{outputs:?}");
                if !first_layer.outputs.iter().all(|o| outputs.contains(o)) {
                    return Err(PatternError::OutputMismatch {
                        expected: outputs.to_owned(),
                        actual: first_layer.outputs.clone(),
                    });
                }
                new_outputs.extend(final_layer.outputs);
                skipped_layers.extend(opt_match.layers.iter().map(|layer| layer.name.clone()));
            }

            let set: HashSet<_> = new_outputs.into_iter().collect();
            let unique_new_outputs: Vec<String> = set.into_iter().collect();

            Ok(Some((pattern, unique_new_outputs, skipped_layers)))
        }
        None => Ok(None),
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BranchMatchMode {
    Any,
    All,
}

// TODO untested with actual branching
fn find_pattern_matches<'a>(
    layers: &'a [ONNXLayer],
    pattern: &GraphPattern,
    mode: BranchMatchMode,
) -> Result<Vec<Vec<&'a ONNXLayer>>, PatternError> {
    if pattern.ops.is_empty() {
        return Err(PatternError::EmptyMatch);
    }
    let mut matches = Vec::new();

    let first_op = pattern.ops.first().ok_or(PatternError::EmptyMatch)?;

    for layer in layers {
        if &layer.op_type == first_op {
            dfs(layer, pattern.ops, 1, &[layer], layers, &mut matches, mode);
        }
    }
    Ok(matches)
}
/*
Inputs:
    - current_layer: current position in the graph
    - ops: list of op names we're trying to match (e.g. ["Conv", "Relu"])
    - depth:  index in the pattern we're trying to match
    - path: vector of matched layers so far
    - all_matches: where completed match paths get collected
    - mode: "Any" (at least one path matches) or "All" (every branch must match)
*/
// Recursive DFS search across branches
fn dfs<'a>(
    current_layer: &'a ONNXLayer,
    ops: &[&'static str],
    depth: usize,
    path: &[&'a ONNXLayer],
    layers: &'a [ONNXLayer],
    all_matches: &mut Vec<Vec<&'a ONNXLayer>>,
    mode: BranchMatchMode,
) {
    if depth > ops.len() {
        return;
    }

    // Base case
    // Save full match if we reach the end of the pattern
    if depth == ops.len() {
        all_matches.push(path.to_owned());
        return;
    }

    // Only consider layers that:
    // - Those whose op matches the next step in the pattern (ops[depth])
    // - and that directly consume one of the outputs from the current layer
    let matching_next_layers: Vec<&ONNXLayer> = layers
        .iter()
        .filter(|l| {
            l.op_type == ops[depth]
                && l.inputs
                    .iter()
                    .any(|inp| current_layer.outputs.contains(inp))
        })
        .collect();

    match mode {
        BranchMatchMode::Any => {
            // Try matching each of the next layers
            // Recurse with new layer and keep going
            // If any completes the pattern, add to all matches
            for next_layer in matching_next_layers {
                let mut new_path = path.to_owned();
                new_path.push(next_layer);
                dfs(
                    next_layer,
                    ops,
                    depth + 1,
                    &new_path,
                    layers,
                    all_matches,
                    mode,
                );
            }
        }
        BranchMatchMode::All => {
            // If there are no next layers that match the next op — we abort early.
            if matching_next_layers.is_empty() {
                return;
            }

            let mut all_paths = vec![];
            for next_layer in matching_next_layers {
                let mut new_path = path.to_owned();
                new_path.push(next_layer);
                let mut sub_matches = Vec::new();
                dfs(
                    next_layer,
                    ops,
                    depth + 1,
                    &new_path,
                    layers,
                    &mut sub_matches,
                    mode,
                );
                if !sub_matches.is_empty() {
                    all_paths.push(sub_matches);
                }
                // We explore every matching direct consumer
                // Recurse on each one
                // Keep only those that reach a complete match
            }

            // Only accept if all direct consumer branches found matching paths
            if !all_paths.is_empty() && all_paths.iter().all(|paths| !paths.is_empty()) {
                for branch in all_paths {
                    for b in branch {
                        all_matches.push(b);
                    }
                }
            }
        }
    }
}

// TODO, somewhere must include priority in sequence, for example, conv relu batchnorm takes priority over conv relu

#[derive(Debug, EnumIter, AsRefStr, Clone)]
pub enum PatternRegistry {
    None,
    ConvRelu,
    GemmRelu,
    BatchnormRelu,
}

impl PatternRegistry {
    #[must_use]
    pub fn as_graph_pattern(&self) -> GraphPattern {
        match self {
            PatternRegistry::None => GraphPattern {
                name: "None",
                ops: &[],
            },
            PatternRegistry::ConvRelu => GraphPattern {
                name: "Conv+Relu",
                ops: &["Conv", "Relu"],
            },
            PatternRegistry::GemmRelu => GraphPattern {
                name: "Gemm+Relu",
                ops: &["Gemm", "Relu"],
            },
            PatternRegistry::BatchnormRelu => GraphPattern {
                name: "Batchnorm+Relu",
                ops: &["Batchnorm", "Relu"],
            },
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GraphPattern {
    pub name: &'static str,
    pub ops: &'static [&'static str],
}

#[derive(Debug, Clone)]
pub struct OptimizationMatch {
    pub pattern: PatternRegistry,
    pub layers: Vec<ONNXLayer>,
}

pub struct PatternMatcher;

impl Default for PatternMatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternMatcher {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Runs pattern matching over a sequence of ONNX layers.
    ///
    /// Iterates through all registered patterns in `self.patterns` and finds matches
    /// in the provided `layers`. Collects all matches into a `HashMap` keyed by the
    /// first layer name of each match.
    ///
    /// # Arguments
    /// - `layers`: Slice of `ONNXLayer` to search for pattern matches.
    ///
    /// # Returns
    /// A `HashMap` mapping layer names (first layer in the match) to a `Vec` of
    /// `OptimizationMatch` structs representing each pattern match.
    ///
    /// # Errors
    /// - [`PatternError::EmptyPattern`] if any pattern has no operations.
    /// - [`PatternError::EmptyMatch`] if a found match contains no layers.
    /// - Any error returned by `find_pattern_matches`.
    pub fn run(
        &self,
        layers: &[ONNXLayer],
    ) -> Result<HashMap<std::string::String, Vec<OptimizationMatch>>, PatternError> {
        use std::time::SystemTime;
        let now = SystemTime::now();

        let mut all_matches: HashMap<String, Vec<OptimizationMatch>> = HashMap::new();

        for pat_enum in PatternRegistry::iter() {
            let graph_pat = pat_enum.as_graph_pattern();
            if graph_pat.ops.is_empty() {
                continue;
            }
            let matches = find_pattern_matches(layers, &graph_pat, BranchMatchMode::All)?;

            for m in matches {
                let first_match = m.first().ok_or(PatternError::EmptyMatch)?;
                all_matches
                    .entry(first_match.name.clone())
                    .or_default()
                    .push(OptimizationMatch {
                        pattern: pat_enum.clone(), // store the enum, not the GraphPattern
                        layers: m.into_iter().cloned().collect(),
                    });
            }
        }
        eprintln!("{all_matches:?}");

        match now.elapsed() {
            Ok(elapsed) => eprintln!(
                "Model pattern match took: {} nano seconds",
                elapsed.as_nanos()
            ),
            Err(e) => eprintln!("Error calculating time: {e:?}"),
        }
        Ok(all_matches)
    }
}

/*

Pattern matching of layers

*/
