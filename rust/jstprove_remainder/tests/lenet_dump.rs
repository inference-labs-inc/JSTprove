use jstprove_remainder::onnx::parser;
use jstprove_remainder::onnx::graph::LayerGraph;
use std::path::Path;

#[test]
fn test_dump_lenet_structure() {
    let parsed = parser::parse_onnx(Path::new("models/lenet.onnx")).unwrap();
    println!("\n=== ONNX inputs ===");
    for inp in &parsed.inputs {
        println!("  {} shape={:?}", inp.name, inp.shape);
    }
    println!("=== ONNX outputs ===");
    for out in &parsed.outputs {
        println!("  {} shape={:?}", out.name, out.shape);
    }
    println!("=== Initializers ===");
    for (name, td) in &parsed.initializers {
        println!("  {} dims={:?} floats={} ints={}", name, td.dims, td.float_data.len(), td.int_data.len());
    }
    println!("=== Nodes ===");
    for node in &parsed.nodes {
        println!("  [{}] op={} inputs={:?} outputs={:?}", node.name, node.op_type, node.inputs, node.outputs);
        for (k, v) in &node.attributes {
            println!("    attr {}: {:?}", k, v);
        }
    }

    let graph = LayerGraph::from_parsed(&parsed).unwrap();
    println!("=== Topo order ===");
    for layer in graph.iter_topo() {
        println!("  [{}] {} ({})", layer.id, layer.name, format!("{:?}", layer.op_type));
        println!("    inputs: {:?}", layer.inputs);
        println!("    outputs: {:?}", layer.outputs);
        println!("    weights: {:?}", layer.weights.keys().collect::<Vec<_>>());
    }
}
