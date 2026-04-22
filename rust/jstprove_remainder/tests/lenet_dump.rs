use jstprove_remainder::onnx::graph::LayerGraph;
use jstprove_remainder::onnx::parser;
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
        let floats = td.float_data.len();
        let ints = td.int_data.len();
        println!("  {name} dims={:?} floats={floats} ints={ints}", td.dims);
    }
    println!("=== Nodes ===");
    for node in &parsed.nodes {
        println!(
            "  [{}] op={} inputs={:?} outputs={:?}",
            node.name, node.op_type, node.inputs, node.outputs
        );
        for (k, v) in &node.attributes {
            println!("    attr {k}: {v:?}");
        }
    }

    let graph = LayerGraph::from_parsed(&parsed).unwrap();
    println!("=== Topo order ===");
    for layer in graph.iter_topo() {
        println!("  [{}] {} ({:?})", layer.id, layer.name, layer.op_type);
        println!("    inputs: {:?}", layer.inputs);
        println!("    outputs: {:?}", layer.outputs);
        let weights: Vec<_> = layer.weights.keys().collect();
        println!("    weights: {weights:?}");
    }
}
