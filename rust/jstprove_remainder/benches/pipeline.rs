use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use jstprove_remainder::runner::circuit_builder::{self, Visibility};
use jstprove_remainder::runner::compile;
use jstprove_remainder::runner::prove;

const LENET_INPUT_ELEMENTS: usize = 3072; // [1, 3, 32, 32]

fn generate_input_file(path: &Path, num_elements: usize) {
    let input: Vec<f64> = (0..num_elements)
        .map(|i| (i as f64 / num_elements as f64))
        .collect();
    let mut map = HashMap::new();
    map.insert("input", input);
    let bytes = rmp_serde::to_vec_named(&map).unwrap();
    std::fs::write(path, bytes).unwrap();
}

fn fmt_bytes(n: u64) -> String {
    if n >= 1_073_741_824 {
        format!("{:.1} GiB", n as f64 / 1_073_741_824.0)
    } else if n >= 1_048_576 {
        format!("{:.1} MiB", n as f64 / 1_048_576.0)
    } else if n >= 1024 {
        format!("{:.1} KiB", n as f64 / 1024.0)
    } else {
        format!("{n} B")
    }
}

fn fmt_duration(ms: f64) -> String {
    if ms >= 1000.0 {
        format!("{:.2}s", ms / 1000.0)
    } else {
        format!("{ms:.1}ms")
    }
}

fn dump_circuit_manifest(model_path: &Path) {
    let model = compile::load_model(model_path).unwrap();

    println!("\n=== CIRCUIT MANIFEST ===");
    println!(
        "scale: {}^{} (alpha={})",
        model.scale_config.base, model.scale_config.exponent, model.scale_config.alpha
    );
    println!("n_bits_config: {:?}", model.n_bits_config);

    let range_plan = circuit_builder::compute_range_check_plan(&model).unwrap();
    println!("\nrange check tables:");
    for (table_nv, shred_names) in &range_plan {
        println!(
            "  table_nv={} (2^{} = {} elements): {} checks: {:?}",
            table_nv,
            table_nv,
            1u64 << table_nv,
            shred_names.len(),
            shred_names
        );
    }

    let build_result = circuit_builder::build_circuit(&model, LENET_INPUT_ELEMENTS).unwrap();

    let mut public_shreds = Vec::new();
    let mut committed_shreds = Vec::new();
    let mut total_public_elements: u64 = 0;
    let mut total_committed_elements: u64 = 0;

    for (name, entry) in &build_result.manifest {
        let elements = 1u64 << entry.num_vars;
        match entry.visibility {
            Visibility::Public => {
                total_public_elements += elements;
                public_shreds.push((name.clone(), entry.num_vars, elements));
            }
            Visibility::Committed => {
                total_committed_elements += elements;
                committed_shreds.push((name.clone(), entry.num_vars, elements));
            }
        }
    }

    public_shreds.sort_by(|a, b| b.1.cmp(&a.1));
    committed_shreds.sort_by(|a, b| b.1.cmp(&a.1));

    println!(
        "\npublic shreds: {} ({} total elements, {} at 32B/Fr)",
        public_shreds.len(),
        total_public_elements,
        fmt_bytes(total_public_elements * 32)
    );
    for (name, nv, elems) in &public_shreds {
        println!("  {}: nv={} (2^{}={} elements)", name, nv, nv, elems);
    }

    println!(
        "\ncommitted shreds: {} ({} total elements, {} at 32B/Fr)",
        committed_shreds.len(),
        total_committed_elements,
        fmt_bytes(total_committed_elements * 32)
    );
    for (name, nv, elems) in &committed_shreds {
        println!("  {}: nv={} (2^{}={} elements)", name, nv, nv, elems);
    }

    println!(
        "\ntotal shreds: {}, total elements: {} ({} at 32B/Fr)",
        public_shreds.len() + committed_shreds.len(),
        total_public_elements + total_committed_elements,
        fmt_bytes((total_public_elements + total_committed_elements) * 32)
    );
    println!("=== END MANIFEST ===\n");
}

fn extract_label(line: &str) -> String {
    if let Some(start) = line.find('"') {
        if let Some(end) = line[start + 1..].find('"') {
            return line[start + 1..start + 1 + end].to_string();
        }
    }
    "unknown".to_string()
}

fn dump_proof_breakdown(proof_path: &Path) {
    let raw = std::fs::read(proof_path).unwrap();
    let proof: prove::SerializableProof = jstprove_io::deserialize_from_bytes(&raw).unwrap();

    let config_bytes = rmp_serde::to_vec_named(&proof.proof_config).unwrap();
    let transcript_bytes = rmp_serde::to_vec_named(&proof.transcript).unwrap();
    let output_bytes = rmp_serde::to_vec_named(&proof.expected_output).unwrap();

    let transcript_display = format!("{}", proof.transcript);
    let mut append_ops = 0u64;
    let mut append_input_ops = 0u64;
    let mut squeeze_ops = 0u64;
    let mut append_elements = 0u64;
    let mut append_input_elements = 0u64;
    let mut label_counts: HashMap<String, (u64, u64)> = HashMap::new();
    for line in transcript_display.lines() {
        if line.starts_with("Append input:") {
            append_input_ops += 1;
            let label = extract_label(line);
            if let Some(pos) = line.find("with ") {
                let rest = &line[pos + 5..];
                if let Some(end) = rest.find(' ') {
                    if let Ok(n) = rest[..end].parse::<u64>() {
                        append_input_elements += n;
                        let entry = label_counts.entry(format!("INPUT:{}", label)).or_default();
                        entry.0 += 1;
                        entry.1 += n;
                    }
                }
            }
        } else if line.starts_with("Append:") {
            append_ops += 1;
            let label = extract_label(line);
            if let Some(pos) = line.find("with ") {
                let rest = &line[pos + 5..];
                if let Some(end) = rest.find(' ') {
                    if let Ok(n) = rest[..end].parse::<u64>() {
                        append_elements += n;
                        let entry = label_counts.entry(label).or_default();
                        entry.0 += 1;
                        entry.1 += n;
                    }
                }
            }
        } else if line.starts_with("Squeeze:") {
            squeeze_ops += 1;
        }
    }

    println!("\n=== PROOF BREAKDOWN ===");
    println!("total proof file:   {}", fmt_bytes(raw.len() as u64));
    println!(
        "  proof_config:     {}",
        fmt_bytes(config_bytes.len() as u64)
    );
    println!(
        "  transcript:       {}",
        fmt_bytes(transcript_bytes.len() as u64)
    );
    println!(
        "  expected_output:  {}",
        fmt_bytes(output_bytes.len() as u64)
    );
    println!("\ntranscript operations:");
    println!(
        "  Append ops:       {} ({} elements)",
        append_ops, append_elements
    );
    println!(
        "  AppendInput ops:  {} ({} elements)",
        append_input_ops, append_input_elements
    );
    println!("  Squeeze ops:      {}", squeeze_ops);
    println!(
        "  total elements:   {} ({} at 32B/Fr)",
        append_elements + append_input_elements,
        fmt_bytes((append_elements + append_input_elements) * 32)
    );

    let mut label_vec: Vec<_> = label_counts.into_iter().collect();
    label_vec.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
    println!("\ntop transcript labels by element count:");
    for (label, (ops, elems)) in label_vec.iter().take(20) {
        println!(
            "  {:>12} elems ({:>8}) in {:>6} ops  {}",
            elems,
            fmt_bytes(*elems * 32),
            ops,
            label
        );
    }
    if label_vec.len() > 20 {
        let rest_elems: u64 = label_vec[20..].iter().map(|(_, (_, e))| e).sum();
        let rest_ops: u64 = label_vec[20..].iter().map(|(_, (o, _))| o).sum();
        println!(
            "  {:>12} elems ({:>8}) in {:>6} ops  ... {} more labels",
            rest_elems,
            fmt_bytes(rest_elems * 32),
            rest_ops,
            label_vec.len() - 20
        );
    }
    println!("=== END BREAKDOWN ===\n");
}

fn main() {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/lenet.onnx");
    assert!(
        model_path.exists(),
        "lenet.onnx not found at {}",
        model_path.display()
    );

    let tmp = tempfile::TempDir::new().unwrap();
    let compiled_path = tmp.path().join("compiled.bin");
    let input_path = tmp.path().join("input.msgpack");
    let witness_path = tmp.path().join("witness.bin");
    let proof_path = tmp.path().join("proof.bin");

    generate_input_file(&input_path, LENET_INPUT_ELEMENTS);

    println!("model:   lenet.onnx (12 layers, input [1,3,32,32])");
    println!("backend: remainder");
    println!("{}", "-".repeat(60));

    let t = Instant::now();
    jstprove_remainder::runner::compile::run(&model_path, &compiled_path, false).unwrap();
    let compile_ms = t.elapsed().as_secs_f64() * 1000.0;
    let compiled_size = std::fs::metadata(&compiled_path).unwrap().len();
    println!(
        "compile: {:>10}  artifact: {}",
        fmt_duration(compile_ms),
        fmt_bytes(compiled_size)
    );

    dump_circuit_manifest(&compiled_path);

    let t = Instant::now();
    jstprove_remainder::runner::witness::run(&compiled_path, &input_path, &witness_path, false)
        .unwrap();
    let witness_ms = t.elapsed().as_secs_f64() * 1000.0;
    let witness_size = std::fs::metadata(&witness_path).unwrap().len();
    println!(
        "witness: {:>10}  artifact: {}",
        fmt_duration(witness_ms),
        fmt_bytes(witness_size)
    );

    let t = Instant::now();
    jstprove_remainder::runner::prove::run(&compiled_path, &witness_path, &proof_path, false)
        .unwrap();
    let prove_ms = t.elapsed().as_secs_f64() * 1000.0;
    let proof_size = std::fs::metadata(&proof_path).unwrap().len();
    println!(
        "prove:   {:>10}  artifact: {}",
        fmt_duration(prove_ms),
        fmt_bytes(proof_size)
    );

    dump_proof_breakdown(&proof_path);

    let t = Instant::now();
    jstprove_remainder::runner::verify::run(&compiled_path, &proof_path, &input_path).unwrap();
    let verify_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("verify:  {:>10}", fmt_duration(verify_ms));

    let total_ms = compile_ms + witness_ms + prove_ms + verify_ms;
    println!("{}", "-".repeat(60));
    println!("total:   {:>10}", fmt_duration(total_ms));
}
