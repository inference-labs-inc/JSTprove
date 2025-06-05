use std::{collections::HashMap, error::Error};

use tract_onnx::{prelude::*, tract_hir::infer::GenericFactoid};


// pub fn analyze_model_internal(path: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
pub fn analyze_model_internal(path: &str, batch_size: Option<usize>) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut model = tract_onnx::onnx().model_for_path(path)?.into_typed()?;
    let inputs = model.inputs.clone();
    let outputs = model.outputs.clone();
    let mut shapes: Vec<usize> = Vec::new();

    for (i, id) in inputs.iter().enumerate() {
            let input = model.node_mut(id.node);

            if input.outputs.is_empty() {
                // return Err(GraphError::MissingOutput(id.node));
                return Err("Missing output".into());
            }
            let mut fact: InferenceFact = input.outputs[0].fact.clone().into();
            // println!("TEST");

            for (i, x) in fact.clone().shape.dims().enumerate() {
                if format!("{}", x).eq("batch_size"){
                        let x = match batch_size {
                            Some(batch_size) => batch_size,
                            None => 1,
                    };
                    println!("{}",x);
                }
                else{
                    println!("{}",x);
                    // shapes.push(x.into());
                }
                // if matches!(x, GenericFactoid::Any) {
                //     let batch_size = match batch_size {
                //         Some(x) => x,
                //         None => return Err,
                //     };
                //     fact.shape
                //         .set_dim(i, tract_onnx::prelude::TDim::Val(batch_size as i64));
                println!("{}", x);
                // }
                // model.set_input_fact(i, fact)?;
            }
            // model.set_input_fact(i, fact)?;
            // shapes.push(fact.shape);
            shapes.push(id.node)

        }
    println!("TEST3");
    // Ok(vec![format!("{:?}", shapes)])
    Ok(model.nodes()
        .iter()
        .map(|n| format!("{}: {}, id: {}", n.name, n.op.name(), n.id))
        .collect())
}


fn get_w_and_b(){

}

fn get_architecture(){

}
fn quantize_layer(){

}
fn quantize_model(){

}