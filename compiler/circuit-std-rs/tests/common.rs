use circuit_std_rs::StdCircuit;
use expander_compiler::frontend::*;
use rand::thread_rng;
use serdes::ExpSerde;

pub fn circuit_test_helper<Cfg, Cir>(params: &Cir::Params)
where
    Cfg: Config,
    Cir: StdCircuit<Cfg>,
{
    circuit_test_helper_with_hint::<Cfg, Cir>(params, &mut EmptyHintCaller);
}

pub fn circuit_test_helper_with_hint<Cfg, Cir>(
    params: &Cir::Params,
    hint: &mut impl HintCaller<CircuitField<Cfg>>,
) where
    Cfg: Config,
    Cir: StdCircuit<Cfg>,
{
    let mut rng = thread_rng();
    let compile_result: CompileResult<Cfg> =
        compile(&Cir::new_circuit(params), CompileOptions::default()).unwrap();
    let assignment = Cir::new_assignment(params, &mut rng);
    let witness = compile_result
        .witness_solver
        .solve_witness_with_hints(&assignment, hint)
        .unwrap();
    let output = compile_result.layered_circuit.run(&witness);
    assert_eq!(output, vec![true]);

    let file = std::fs::File::create("circuit.txt").unwrap();
    let writer = std::io::BufWriter::new(file);
    compile_result
        .layered_circuit
        .serialize_into(writer)
        .unwrap();

    let file = std::fs::File::create("witness.txt").unwrap();
    let writer = std::io::BufWriter::new(file);
    witness.serialize_into(writer).unwrap();

    let file = std::fs::File::create("witness_solver.txt").unwrap();
    let writer = std::io::BufWriter::new(file);
    compile_result
        .witness_solver
        .serialize_into(writer)
        .unwrap();
}
