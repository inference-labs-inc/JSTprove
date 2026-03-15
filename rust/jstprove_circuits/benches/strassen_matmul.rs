use std::time::Instant;

use expander_compiler::frontend::{
    BN254Config, CompileOptions, Config, Define, RootAPI, Variable, compile, declare_circuit,
};
use jstprove_circuits::circuit_functions::gadgets::linear_algebra::{
    matrix_multiplication, strassen_matrix_multiplication,
};
use jstprove_circuits::circuit_functions::layers::LayerKind;
use ndarray::{Array2, ArrayD};

const SIZES: &[usize] = &[64, 128];

declare_circuit!(StandardMatmulCircuit {
    mat_a: [Variable],
    mat_b: [Variable],
    mat_c: [Variable]
});

declare_circuit!(StrassenMatmulCircuit {
    mat_a: [Variable],
    mat_b: [Variable],
    mat_c: [Variable]
});

fn flat_to_2d(flat: &[Variable], rows: usize, cols: usize) -> ArrayD<Variable> {
    let mut arr = Array2::default((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            arr[(i, j)] = flat[i * cols + j];
        }
    }
    arr.into_dyn()
}

impl<C: Config> Define<C> for StandardMatmulCircuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        let n = (self.mat_a.len() as f64).sqrt() as usize;
        let a = flat_to_2d(&self.mat_a, n, n);
        let b = flat_to_2d(&self.mat_b, n, n);

        let c = matrix_multiplication(api, a, b, LayerKind::Gemm).unwrap();

        let c_flat: Vec<Variable> = c.into_iter().collect();
        for (i, &expected) in self.mat_c.iter().enumerate() {
            api.assert_is_equal(c_flat[i], expected);
        }
    }
}

impl<C: Config> Define<C> for StrassenMatmulCircuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        let n = (self.mat_a.len() as f64).sqrt() as usize;
        let a = flat_to_2d(&self.mat_a, n, n);
        let b = flat_to_2d(&self.mat_b, n, n);

        let c = strassen_matrix_multiplication(api, a, b, LayerKind::Gemm).unwrap();

        let c_flat: Vec<Variable> = c.into_iter().collect();
        for (i, &expected) in self.mat_c.iter().enumerate() {
            api.assert_is_equal(c_flat[i], expected);
        }
    }
}

fn make_standard(n: usize) -> StandardMatmulCircuit<Variable> {
    let nn = n * n;
    StandardMatmulCircuit {
        mat_a: vec![Variable::default(); nn],
        mat_b: vec![Variable::default(); nn],
        mat_c: vec![Variable::default(); nn],
    }
}

fn make_strassen(n: usize) -> StrassenMatmulCircuit<Variable> {
    let nn = n * n;
    StrassenMatmulCircuit {
        mat_a: vec![Variable::default(); nn],
        mat_b: vec![Variable::default(); nn],
        mat_c: vec![Variable::default(); nn],
    }
}

fn fmt(ms: f64) -> String {
    if ms >= 1000.0 {
        format!("{:.2}s", ms / 1000.0)
    } else {
        format!("{ms:.1}ms")
    }
}

fn bench_compile<
    C: Config,
    Cir: expander_compiler::frontend::internal::DumpLoadTwoVariables<Variable> + Define<C> + Clone,
>(
    label: &str,
    circuit: &Cir,
) {
    let t = Instant::now();
    let result = compile(circuit, CompileOptions::default()).unwrap();
    let elapsed = t.elapsed().as_secs_f64() * 1000.0;

    let stats = result.layered_circuit.get_stats();
    println!(
        "  {label:<12} compile={:<10} numMul={:<10} numAdd={:<10} numVars={:<10} layers={} totalCost={}",
        fmt(elapsed),
        stats.num_expanded_mul,
        stats.num_expanded_add,
        stats.num_total_gates,
        stats.num_layers,
        stats.total_cost,
    );
}

fn main() {
    println!("Strassen vs Standard matmul circuit comparison");
    println!("{}", "=".repeat(80));

    for &n in SIZES {
        println!("\n--- {n}x{n} x {n}x{n} matmul ---");

        let std_circuit = make_standard(n);
        bench_compile::<BN254Config, _>("standard", &std_circuit);

        let str_circuit = make_strassen(n);
        bench_compile::<BN254Config, _>("strassen", &str_circuit);
    }
}
