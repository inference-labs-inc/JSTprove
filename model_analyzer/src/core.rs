use tract_onnx::prelude::*;

pub fn analyze_model_internal(path: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let model = tract_onnx::onnx().model_for_path(path)?.into_typed()?;
    Ok(model.nodes()
        .iter()
        .map(|n| format!("{}: {}", n.name, n.op.name()))
        .collect())
}