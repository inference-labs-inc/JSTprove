use expander_compiler::{circuit::layered::InputUsize, frontend::*};

declare_circuit!(Circuit {
    x: [Variable; 16],
    y: [Variable; 512],
    sum: Variable,
});

impl Define<M31Config> for Circuit<Variable> {
    fn define<Builder: RootAPI<M31Config>>(&self, builder: &mut Builder) {
        let mut sum = builder.constant(0);
        for i in 0..16 {
            for j in 0..512 {
                let t = builder.mul(self.x[i], self.y[j]);
                sum = builder.add(sum, t);
            }
        }
        builder.assert_is_equal(self.sum, sum);
    }
}

fn mul_fanout_limit(limit: usize) {
    let compile_result = compile(
        &Circuit::default(),
        CompileOptions::default().with_mul_fanout_limit(limit),
    )
    .unwrap();
    let circuit = compile_result.layered_circuit;
    for segment in circuit.segments.iter() {
        let mut ref_num = vec![0; segment.num_inputs.get(0)];
        for m in segment.gate_muls.iter() {
            ref_num[m.inputs[0].offset] += 1;
            ref_num[m.inputs[1].offset] += 1;
        }
        for x in ref_num.iter() {
            assert!(*x <= limit);
        }
    }
}

#[test]
fn mul_fanout_limit_2() {
    mul_fanout_limit(2);
}

#[test]
fn mul_fanout_limit_3() {
    mul_fanout_limit(3);
}

#[test]
fn mul_fanout_limit_4() {
    mul_fanout_limit(4);
}

#[test]
fn mul_fanout_limit_16() {
    mul_fanout_limit(16);
}

#[test]
fn mul_fanout_limit_64() {
    mul_fanout_limit(64);
}

#[test]
fn mul_fanout_limit_256() {
    mul_fanout_limit(256);
}

#[test]
fn mul_fanout_limit_1024() {
    mul_fanout_limit(1024);
}
