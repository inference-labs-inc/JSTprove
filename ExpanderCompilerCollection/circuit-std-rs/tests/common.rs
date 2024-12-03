use circuit_std_rs::StdCircuit;
use expander_compiler::frontend::*;
use extra::Serde;
use rand::thread_rng;

pub fn circuit_test_helper<Cfg, Cir>(params: &Cir::Params)
where
    Cfg: Config,
    Cir: StdCircuit<Cfg>,
{
    let mut rng = thread_rng();
    let compile_result: CompileResult<Cfg> = compile(&Cir::new_circuit(&params)).unwrap();
    let assignment = Cir::new_assignment(&params, &mut rng);
    let witness = compile_result
        .witness_solver
        .solve_witness(&assignment)
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
