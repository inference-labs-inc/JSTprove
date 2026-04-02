// Copyright 2025 Irreducible Inc.
mod ast;
mod parser;
mod randblast;
#[cfg(feature = "z3")]
mod smt_check;
mod translate;

use std::fs;

use anyhow::{Result, anyhow};
use clap::Parser as ClapParser;
#[cfg(feature = "z3")]
use z3::{Config, Context};

#[cfg(feature = "z3")]
use crate::smt_check::SmtChecker;
use crate::{
    ast::{ConstraintSet, TestItem},
    parser::parse_test_file,
    randblast::RandBlast,
};

#[derive(Debug, Clone, Copy, PartialEq)]
enum EquivalenceResult {
    /// Systems are definitely equivalent
    #[allow(dead_code)] // Only used when z3 feature is enabled
    Equivalent,
    /// Systems are definitely not equivalent
    NotEquivalent,
    /// Cannot determine (e.g., no SMT solver and random testing didn't find difference)
    Inconclusive,
}

#[derive(ClapParser, Debug)]
#[command(author, version, about = "Constraint Equivalence Checker", long_about = None)]
struct Args {
    /// Test file path (use '-' to read from stdin)
    file: String,

    /// Number of random tests to run
    #[arg(short = 'r', long, default_value = "10000")]
    random_tests: usize,

    /// Skip random testing
    #[arg(short = 's', long)]
    skip_random: bool,

    /// Skip SMT checking
    #[arg(long = "skip-smt")]
    skip_smt: bool,

    /// Random seed for testing
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Be optimistic: don't fail on inconclusive results
    #[arg(short = 'o', long)]
    optimistic: bool,
}

fn check_constraint_systems(
    lhs_cs: &ConstraintSet,
    rhs_cs: &ConstraintSet,
    args: &Args,
) -> Result<EquivalenceResult> {
    let mut cx = translate::Context::new();
    cx.preprocess(lhs_cs);
    cx.preprocess(rhs_cs);
    cx.perform_witness_assignment();
    let lhs = cx.build(lhs_cs);
    let rhs = cx.build(rhs_cs);

    let mut randblast_found_difference = false;

    // Randblast testing
    if !args.skip_random {
        let mut blaster = RandBlast::new(args.seed);
        match blaster.test_equivalence(&lhs, &rhs, args.random_tests) {
            Ok(()) => {
                // No difference found by random testing
            }
            Err(e) => {
                eprintln!("Randblast testing failed: {e}");
                randblast_found_difference = true;
            }
        }
    }

    // If randblast found a difference, we're definitely not equivalent
    if randblast_found_difference {
        return Ok(EquivalenceResult::NotEquivalent);
    }

    // SMT check for definitive answer
    if !args.skip_smt {
        #[cfg(feature = "z3")]
        {
            let config = Config::new();
            let ctx = Context::new(&config);
            let mut checker = SmtChecker::new(&ctx);

            match checker.check_equivalence(&lhs, &rhs) {
                Ok(()) => return Ok(EquivalenceResult::Equivalent),
                Err(e) => {
                    eprintln!("{e}");
                    return Ok(EquivalenceResult::NotEquivalent);
                }
            }
        }
        #[cfg(not(feature = "z3"))]
        {
            eprintln!("Warning: SMT check skipped (ceck was compiled without z3 support)");
            if args.skip_random {
                eprintln!("         Cannot determine equivalence without both randblast and SMT");
            } else {
                eprintln!(
                    "         Randblast provides ~99% confidence but not mathematical certainty"
                );
            }
        }
    }

    // No definitive answer: randblast didn't find difference and no SMT
    Ok(EquivalenceResult::Inconclusive)
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Read content from file or stdin
    let content = if args.file == "-" {
        // Read from stdin
        use std::io::Read;
        let mut buffer = String::new();
        std::io::stdin()
            .read_to_string(&mut buffer)
            .map_err(|e| anyhow!("Failed to read from stdin: {}", e))?;
        buffer
    } else {
        // Read from file
        fs::read_to_string(&args.file)
            .map_err(|e| anyhow!("Failed to read file {}: {}", args.file, e))?
    };

    // Parse the test file
    let test_file =
        parse_test_file(&content).map_err(|e| anyhow!("Failed to parse test file: {}", e))?;

    if test_file.assertions.is_empty() {
        return Err(anyhow!("Test file contains no assertions"));
    }

    let mut all_passed = true;
    for (i, assertion) in test_file.assertions.iter().enumerate() {
        match assertion {
            TestItem::AssertEqv(assert_eqv) => {
                let result = check_constraint_systems(&assert_eqv.lhs, &assert_eqv.rhs, &args)?;
                match result {
                    EquivalenceResult::Equivalent => {
                        println!("Test {}: EQUIVALENT", i + 1);
                    }
                    EquivalenceResult::NotEquivalent => {
                        println!("Test {}: NOT EQUIVALENT", i + 1);
                        all_passed = false;
                    }
                    EquivalenceResult::Inconclusive => {
                        if args.optimistic {
                            println!("Test {}: INCONCLUSIVE (assumed equivalent)", i + 1);
                        } else {
                            println!("Test {}: INCONCLUSIVE (failing)", i + 1);
                            all_passed = false;
                        }
                    }
                }
            }
            TestItem::AssertNotEqv(assert_not_eqv) => {
                let result =
                    check_constraint_systems(&assert_not_eqv.lhs, &assert_not_eqv.rhs, &args)?;
                match result {
                    EquivalenceResult::NotEquivalent => {
                        println!("Test {}: NOT EQUIVALENT (as expected)", i + 1);
                    }
                    EquivalenceResult::Equivalent => {
                        println!("Test {}: EQUIVALENT (unexpected!)", i + 1);
                        all_passed = false;
                    }
                    EquivalenceResult::Inconclusive => {
                        if args.optimistic {
                            println!("Test {}: INCONCLUSIVE (assumed not equivalent)", i + 1);
                        } else {
                            println!("Test {}: INCONCLUSIVE (failing)", i + 1);
                            all_passed = false;
                        }
                    }
                }
            }
        }
    }

    if all_passed {
        Ok(())
    } else {
        Err(anyhow!("Some tests failed"))
    }
}
