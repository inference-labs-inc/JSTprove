//! A tiny demo circuit that proves each public input `a_i` satisfies
//!     0 ≤ a_i ≤ UPPER_BOUND
//! using our generic `range_check` helper.
//!
//!  • radix  = 10
//!  • κ      = 3            (so we’re really checking a_i < 10³)
//!  • bound  = 999          (fits that radix window)
//!
//! The bound and radix are *compile-time constants* baked into the circuit;
//! only the a_i’s come from the witness / public JSON.

use expander_compiler::frontend::*;
use circuit_std_rs::logup::LogUpRangeProofTable;

use gravy_circuits::runner::main_runner::handle_args;
use gravy_circuits::io::io_reader::{FileReader, IOReader};

// <-- bring in the helper ---------------------------------------------
use gravy_circuits::circuit_functions::range_check::range_check;
// ----------------------------------------------------------------------

/* ---------- demo parameters ---------- */
const RADIX:       u32   = 10;
const KAPPA:       usize = 3;
const UPPER_BOUND: u32   = 999;

const BATCH_SIZE:  usize = 32;   // number of a_i values

/* ---------- circuit I/O declaration  ---------- */
declare_circuit!(RangeCheckCircuit {
    a_vec: [PublicVariable; BATCH_SIZE],   // the integers we’ll check
});

/* ---------- circuit logic  ---------- */
impl<C: Config> Define<C> for RangeCheckCircuit<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {

        /* hard-coded bound as field constant */
        let b1_bar = api.constant(UPPER_BOUND);

        /* we’ll *not* use lookup for this tiny demo => table == None */
        let use_lookup = false;
        let mut table_opt: Option<&mut LogUpRangeProofTable> = None;

        for &a in self.a_vec.iter() {
            range_check(
                api,
                a,                                     /* the value            */
                Some((RADIX, KAPPA, b1_bar)),          /* upper-bound tuple    */
                None,                                  /* no lower bound       */
                use_lookup,
                &mut table_opt,
            );
        }
    }
}

/* ---------- JSON I/O plumbing  ---------- */
#[derive(serde::Deserialize, Clone)]
struct InputData {
    a_vec: Vec<u32>,
}
#[derive(serde::Deserialize, Clone)]
struct OutputData {}   // empty – circuit only asserts constraints

impl<C: Config> IOReader<RangeCheckCircuit<C::CircuitField>, C> for FileReader {
    fn read_inputs(
        &mut self,
        file_path: &str,
        mut assignment: RangeCheckCircuit<C::CircuitField>,
    ) -> RangeCheckCircuit<C::CircuitField> {
        let data: InputData = Self::read_data_from_json::<InputData>(file_path);
        assert_eq!(data.a_vec.len(), BATCH_SIZE);
        for (i, &v) in data.a_vec.iter().enumerate() {
            assignment.a_vec[i] = C::CircuitField::from(v);
        }
        assignment
    }
    fn read_outputs(
        &mut self,
        _file_path: &str,
        assignment: RangeCheckCircuit<C::CircuitField>,
    ) -> RangeCheckCircuit<C::CircuitField> {
        assignment   // nothing to check
    }
    fn get_path(&self) -> &str { &self.path }
}

/* ---------- main runner  ---------- */
fn main() {
    let mut reader = FileReader { path: "range_check_demo".into() };
    handle_args::<RangeCheckCircuit<Variable>, RangeCheckCircuit<_>, _>(&mut reader);
}
