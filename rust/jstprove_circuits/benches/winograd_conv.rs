use expander_compiler::frontend::*;
use ndarray::ArrayD;

use jstprove_circuits::circuit_functions::utils::tensor_ops::load_circuit_constant;

declare_circuit!(ConvOnlyCircuit {
    input: [Variable; 784],
});

const IN_C: usize = 1;
const IN_H: usize = 28;
const IN_W: usize = 28;
const OUT_C: usize = 8;
const KH: usize = 3;
const KW: usize = 3;
const OUT_H: usize = IN_H - KH + 1;
const OUT_W: usize = IN_W - KW + 1;

impl Define<BN254Config> for ConvOnlyCircuit<Variable> {
    fn define<Builder: RootAPI<BN254Config>>(&self, api: &mut Builder) {
        let input_shape = vec![1, IN_C, IN_H, IN_W];
        let input_arr =
            ArrayD::from_shape_vec(ndarray::IxDyn(&input_shape), self.input.to_vec()).unwrap();

        let weight_shape = vec![OUT_C, IN_C, KH, KW];
        let num_weights = OUT_C * IN_C * KH * KW;
        let weight_vals: Vec<Variable> = (0..num_weights)
            .map(|i| load_circuit_constant(api, (i as i64 % 7) - 3))
            .collect();
        let weights = ArrayD::from_shape_vec(ndarray::IxDyn(&weight_shape), weight_vals).unwrap();

        let bias = ArrayD::from_shape_vec(ndarray::IxDyn(&[0]), vec![]).unwrap();

        let conv_params = jstprove_circuits::circuit_functions::layers::conv::Conv2DParams {
            dilations: vec![1, 1],
            kernel_shape: vec![KH as u32, KW as u32],
            pads: vec![0, 0, 0, 0],
            strides: vec![1, 1],
            input_shape: vec![1, IN_C as u32, IN_H as u32, IN_W as u32],
            groups: vec![1],
        };

        let out = jstprove_circuits::circuit_functions::layers::conv::conv_shape_4(
            api,
            &input_arr,
            &conv_params,
            &weights,
            &bias,
        )
        .unwrap();

        let zero = load_circuit_constant(api, 0);
        for val in out.iter() {
            api.assert_is_equal(*val, zero);
        }
    }
}

fn main() {
    println!("Compiling conv-only circuit (no rescale, no ReLU)...");
    println!(
        "Config: {}x{}x{} input, {}x{} kernel, {} output channels",
        IN_C, IN_H, IN_W, KH, KW, OUT_C
    );
    println!(
        "Output: {}x{}x{} = {} elements",
        OUT_C,
        OUT_H,
        OUT_W,
        OUT_C * OUT_H * OUT_W
    );
    println!(
        "Standard dot product muls per output: {} (kernel_h * kernel_w * in_c)",
        KH * KW * IN_C
    );
    println!(
        "Expected total api.mul calls: {} (output_elements * muls_per_output)",
        OUT_C * OUT_H * OUT_W * KH * KW * IN_C
    );

    let compile_result = compile(&ConvOnlyCircuit::default(), CompileOptions::default()).unwrap();
    let stats = compile_result.layered_circuit.get_stats();

    println!("\n--- Circuit Stats ---");
    println!("numLayers:    {}", stats.num_layers);
    println!("numVariables: {}", stats.num_total_gates);
    println!("numUsedVars:  {}", stats.num_used_gates);
    println!("numMul:       {}", stats.num_expanded_mul);
    println!("numAdd:       {}", stats.num_expanded_add);
    println!("numCst:       {}", stats.num_expanded_cst);
    println!("totalCost:    {}", stats.total_cost);

    let total_api_mul_calls = OUT_C * OUT_H * OUT_W * KH * KW * IN_C;
    println!(
        "\napi.mul() calls in conv: {}  |  actual mul gates: {}",
        total_api_mul_calls, stats.num_expanded_mul
    );
    if stats.num_expanded_mul == 0 {
        println!("CONFIRMED: constant*variable multiplications produce ZERO mul gates.");
        println!("Winograd optimization cannot reduce mul gates because they are already zero.");
    } else {
        println!("Mul gates detected. Further investigation needed to determine source.");
    }
}
