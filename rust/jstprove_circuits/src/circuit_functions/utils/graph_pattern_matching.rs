use std::collections::{HashMap, HashSet};

/// External crate imports
use crate::circuit_functions::utils::onnx_types::ONNXLayer;

/*

Pattern matching of layers

*/

pub fn optimization_skip_layers(
    optimization_match: Option<&Vec<OptimizationMatch>>,
    outputs: Vec<String>,
) -> Option<(GraphPattern, Vec<String>, Vec<String>)> {
    match optimization_match {
        Some(opt) => {
            let pattern = opt[0].pattern;
            let mut new_outputs = Vec::new();
            let mut skipped_layers: Vec<String> = Vec::new();
            // Loop through all potential branches
            for opt_match in opt {
                // Assert all the patterns are the same
                assert!(pattern.name == opt_match.pattern.name);
                // Get final layer of pattern
                let layers = opt_match.layers.clone();
                let final_layer = layers[layers.len() - 1].clone();
                let first_layer = layers[0].clone();

                // Assert outputs match
                eprintln!("{:?}", first_layer.outputs);
                eprintln!("{:?}", outputs);
                assert!(first_layer
                    .outputs
                    .iter()
                    .all(|item| outputs.contains(item)));
                new_outputs.extend(final_layer.outputs);
                skipped_layers.extend(opt_match.layers.iter().map(|layer| layer.name.clone()))
            }
            // Search the other way. Makes sure both sides of inequality holds
            // assert!(outputs.iter().all(|item| new_outputs.contains(item)));

            let set: HashSet<_> = new_outputs.into_iter().collect();
            let unique_new_outputs: Vec<String> = set.into_iter().collect();
            // let set: HashSet<_> = outputs.into_iter().collect();
            // let unique_old_outputs: Vec<String> = set.into_iter().collect();

            // assert!(unique_new_outputs.len() == unique_old_outputs.len());

            Some((pattern, unique_new_outputs, skipped_layers))
        }
        None => return None,
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
) -> Vec<Vec<&'a ONNXLayer>> {
    let mut matches = Vec::new();
    for layer in layers {
        if layer.op_type == pattern.ops[0] {
            dfs(
                layer,
                pattern.ops,
                1,
                vec![layer],
                layers,
                &mut matches,
                mode,
            );
        }
    }
    matches
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
    path: Vec<&'a ONNXLayer>,
    layers: &'a [ONNXLayer],
    all_matches: &mut Vec<Vec<&'a ONNXLayer>>,
    mode: BranchMatchMode,
) {
    // Base case
    // Save full match if we reach the end of the pattern
    if depth == ops.len() {
        all_matches.push(path.clone());
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
                let mut new_path = path.clone();
                new_path.push(next_layer);
                dfs(
                    next_layer,
                    ops,
                    depth + 1,
                    new_path,
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
                let mut new_path = path.clone();
                new_path.push(next_layer);
                let mut sub_matches = Vec::new();
                dfs(
                    next_layer,
                    ops,
                    depth + 1,
                    new_path.clone(),
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
            if all_paths.len() >= 1 && all_paths.iter().all(|paths| !paths.is_empty()) {
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
fn build_pattern_registry() -> Vec<GraphPattern> {
    vec![
        GraphPattern {
            name: "Conv+Relu".into(),
            ops: &["Conv", "Relu"],
        },
        GraphPattern {
            name: "Gemm+Relu".into(),
            ops: &["Gemm", "Relu"],
        },
    ]
}
#[derive(Debug, Clone, Copy, Default)]
pub struct GraphPattern {
    pub name: &'static str,
    pub ops: &'static [&'static str],
}

#[derive(Debug, Clone)]
pub struct OptimizationMatch {
    pub pattern: GraphPattern,
    pub layers: Vec<ONNXLayer>,
}

pub struct PatternMatcher {
    patterns: Vec<GraphPattern>,
}

impl PatternMatcher {
    pub fn new() -> Self {
        Self {
            patterns: build_pattern_registry(),
        }
    }

    pub fn run(
        &self,
        layers: &[ONNXLayer],
    ) -> HashMap<std::string::String, Vec<OptimizationMatch>> {
        use std::time::SystemTime;
        let now = SystemTime::now();

        let mut all_matches: HashMap<String, Vec<OptimizationMatch>> = HashMap::new();

        for pat in &self.patterns {
            let matches = find_pattern_matches(layers, pat, BranchMatchMode::All);
            eprintln!("Pattern `{}` matched {} times", pat.name, matches.len());

            for m in matches {
                all_matches
                    .entry(m[0].name.clone())
                    .or_default()
                    .push(OptimizationMatch {
                        pattern: *pat,
                        layers: m.into_iter().cloned().collect(),
                    });
            }
            // eprintln!("{:?}", matches[0]);
        }
        eprintln!("{:?}", all_matches);

        match now.elapsed() {
            Ok(elapsed) => {
                // it prints '2'
                eprintln!(
                    "Model pattern match took: {} nano seconds",
                    elapsed.as_nanos()
                );
            }
            Err(e) => {
                // an error occurred!
                eprintln!("Error calculating time: {e:?}");
            }
        }
        all_matches
        // panic!("");
    }
}

/*

Pattern matching of layers

*/
