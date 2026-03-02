use std::{sync::Arc, thread};

use expander_compiler::frontend::*;

declare_circuit!(Circuit {
    x: Variable,
    y: Variable,
});

impl Define<M31Config> for Circuit<Variable> {
    fn define<Builder: RootAPI<M31Config>>(&self, builder: &mut Builder) {
        builder.assert_is_equal(self.x, self.y);
    }
}

#[test]
fn multithreading_witness_solving() {
    let compile_result = compile(&Circuit::default(), CompileOptions::default()).unwrap();
    let mut assignments = Vec::new();
    for _ in 0..1024 {
        assignments.push(Circuit::<M31> {
            x: M31::from(123),
            y: M31::from(123),
        });
    }
    // Since our SimdField is M31x16, we can solve 16 assignments at once
    let assignment_chunks: Vec<Vec<Circuit<M31>>> =
        assignments.chunks(16).map(|x| x.to_vec()).collect();
    // We use Arc to share the WitnessSolver between threads
    let witness_solver = Arc::new(compile_result.witness_solver);
    // In this example, we start a thread for each chunk of assignments
    // You may use a thread pool for better performance
    let handles = assignment_chunks
        .into_iter()
        .map(|assignments| {
            let witness_solver = Arc::clone(&witness_solver);
            thread::spawn(move || witness_solver.solve_witnesses(&assignments).unwrap())
        })
        .collect::<Vec<_>>();
    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.join().unwrap());
    }
    for result in results {
        let output = compile_result.layered_circuit.run(&result);
        assert_eq!(output, vec![true; 16]);
    }
}
