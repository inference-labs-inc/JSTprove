use crate::{
    builder,
    circuit::{
        config::Config,
        input_mapping::InputMapping,
        ir,
        layered::{self, InputType},
    },
    layering,
    utils::error::Error,
};

#[cfg(test)]
mod random_circuit_tests;
#[cfg(test)]
mod tests;

#[derive(Debug, Clone)]
pub struct CompileOptions {
    pub mul_fanout_limit: Option<usize>,
    pub allow_input_reorder: bool,
    pub opt_level: usize,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            mul_fanout_limit: None,
            allow_input_reorder: true,
            opt_level: 3,
        }
    }
}

impl CompileOptions {
    pub fn with_mul_fanout_limit(mut self, mul_fanout_limit: usize) -> Self {
        self.mul_fanout_limit = Some(mul_fanout_limit);
        self
    }
    pub fn without_input_reorder(mut self) -> Self {
        self.allow_input_reorder = false;
        self
    }
    pub fn with_opt_level(mut self, opt_level: usize) -> Self {
        self.opt_level = opt_level;
        self
    }
    pub fn validate(&self) -> Result<(), Error> {
        if self.mul_fanout_limit.is_some() && self.mul_fanout_limit.unwrap() <= 1 {
            return Err(Error::UserError("mul_fanout_limit must be > 1".to_string()));
        }
        if self.opt_level < 1 || self.opt_level > 3 {
            return Err(Error::UserError("opt_level must be 1, 2 or 3".to_string()));
        }
        Ok(())
    }
}

fn optimize_until_fixed_point<T, F>(x: &T, im: &mut InputMapping, f: F) -> T
where
    T: Clone + Eq,
    F: Fn(&T) -> (T, InputMapping),
{
    let (mut y, imy) = f(x);
    if *x == y {
        return y;
    }
    im.compose_in_place(&imy);
    loop {
        let (z, imz) = f(&y);
        if y == z {
            return y;
        }
        y = z;
        im.compose_in_place(&imz);
    }
}

fn print_info(info: &str) {
    print!(
        "\x1b[90m{}\x1b[0m \x1b[32mINF\x1b[0m {} ",
        chrono::Local::now().format("%H:%M:%S"),
        info
    );
}

fn print_stat(stat_name: &str, stat: usize, is_last: bool) {
    print!("\x1b[36m{stat_name}=\x1b[0m{stat}");
    if !is_last {
        print!(" ");
    } else {
        println!();
    }
}

pub fn compile_step_1<C: Config>(
    r_source: &ir::source::RootCircuit<C>,
    options: CompileOptions,
) -> Result<(ir::hint_normalized::RootCircuit<C>, InputMapping), Error> {
    r_source.validate()?;

    let mut src_im = InputMapping::new_identity(r_source.input_size());

    let mut r_source = r_source.clone();

    let r_source_opt = if options.opt_level >= 3 {
        r_source.detect_chains();
        optimize_until_fixed_point(&r_source, &mut src_im, |r| {
            let (mut r, im) = r.remove_unreachable();
            r.reassign_duplicate_sub_circuit_outputs(false);
            r.detect_chains();
            (r, im)
        })
    } else if options.opt_level >= 1 {
        r_source.detect_chains();
        let (r, im) = r_source.remove_unreachable();
        src_im.compose_in_place(&im);
        r
    } else {
        r_source
    };
    r_source_opt
        .validate()
        .map_err(|e| e.prepend("source ir circuit invalid"))?;

    let r_hint_normalized = builder::hint_normalize::process(&r_source_opt)
        .map_err(|e| e.prepend("hint normalization failed"))?;

    let r_hint_normalized_opt = if options.opt_level >= 2 {
        optimize_until_fixed_point(&r_hint_normalized, &mut src_im, |r| {
            let (mut r, im) = r.remove_unreachable();
            r.reassign_duplicate_sub_circuit_outputs(false);
            (r, im)
        })
    } else if options.opt_level >= 1 {
        let (r, im) = r_hint_normalized.remove_unreachable();
        src_im.compose_in_place(&im);
        r
    } else {
        r_hint_normalized
    };
    r_hint_normalized_opt
        .validate()
        .map_err(|e| e.prepend("hint normalized ir circuit invalid"))?;
    Ok((r_hint_normalized_opt, src_im))
}

pub fn compile_step_2<C: Config, I: InputType>(
    r_hint_less: ir::hint_less::RootCircuit<C>,
    options: CompileOptions,
) -> Result<(ir::dest::RootCircuit<C>, InputMapping), Error> {
    let mut hl_im = InputMapping::new_identity(r_hint_less.input_size());

    let r_hint_less_opt = if options.opt_level >= 2 {
        optimize_until_fixed_point(&r_hint_less, &mut hl_im, |r| {
            let (mut r, im) = r.remove_unreachable();
            r.reassign_duplicate_sub_circuit_outputs(false);
            (r, im)
        })
    } else {
        r_hint_less
    };
    r_hint_less_opt
        .validate()
        .map_err(|e| e.prepend("hint less ir circuit invalid"))?;

    let r_dest_relaxed = builder::final_build_opt::process(&r_hint_less_opt)
        .map_err(|e| e.prepend("final build failed"))?;

    let r_dest_relaxed_opt = if options.opt_level >= 2 {
        optimize_until_fixed_point(&r_dest_relaxed, &mut hl_im, |r| {
            let (mut r, im) = r.remove_unreachable();
            r.reassign_duplicate_sub_circuit_outputs(false);
            (r, im)
        })
    } else if options.opt_level >= 1 {
        let (r, im) = r_dest_relaxed.remove_unreachable();
        hl_im.compose_in_place(&im);
        r
    } else {
        r_dest_relaxed
    };
    r_dest_relaxed_opt
        .validate()
        .map_err(|e| e.prepend("dest relaxed ir circuit invalid"))?;

    let r_dest_relaxed_opt = if let Some(limit) = options.mul_fanout_limit {
        let r = r_dest_relaxed_opt.solve_mul_fanout_limit(limit);
        r.validate()
            .map_err(|e| e.prepend("dest relaxed ir circuit invalid"))?;
        r
    } else {
        r_dest_relaxed_opt
    };

    let r_dest_relaxed_p2 = if C::ENABLE_RANDOM_COMBINATION {
        r_dest_relaxed_opt
    } else {
        let mut r1 = r_dest_relaxed_opt.export_constraints();
        r1.reassign_duplicate_sub_circuit_outputs(false);
        let (r2, im) = r1.remove_unreachable();
        hl_im.compose_in_place(&im);
        r2.validate()
            .map_err(|e| e.prepend("dest relaxed ir circuit invalid"))?;
        r2
    };

    let mut r_dest_relaxed_p3 = if I::CROSS_LAYER_RELAY {
        r_dest_relaxed_p2
    } else {
        let r = layering::ir_split::split_to_single_layer(&r_dest_relaxed_p2);
        r.validate()
            .map_err(|e| e.prepend("dest relaxed ir circuit invalid"))?;

        if options.opt_level >= 2 {
            optimize_until_fixed_point(&r, &mut hl_im, |r| {
                let (mut r, im) = r.remove_unreachable();
                r.reassign_duplicate_sub_circuit_outputs(false);
                (r, im)
            })
        } else if options.opt_level >= 1 {
            let (r, im) = r.remove_unreachable();
            hl_im.compose_in_place(&im);
            r
        } else {
            r
        }
    };

    if options.opt_level == 1 {
        r_dest_relaxed_p3.reassign_duplicate_sub_circuit_outputs(true);
    }

    let r_dest = r_dest_relaxed_p3.solve_duplicates();

    let r_dest_opt = if options.opt_level >= 2 {
        optimize_until_fixed_point(&r_dest, &mut hl_im, |r| {
            let (mut r, im) = r.remove_unreachable();
            r.reassign_duplicate_sub_circuit_outputs(false);
            (r, im)
        })
    } else if options.opt_level >= 1 {
        let (r, im) = r_dest.remove_unreachable();
        hl_im.compose_in_place(&im);
        r
    } else {
        r_dest
    };
    r_dest_opt
        .validate()
        .map_err(|e| e.prepend("dest ir circuit invalid"))?;
    r_dest_opt
        .validate_circuit_has_inputs()
        .map_err(|e| e.prepend("dest ir circuit invalid"))?;
    Ok((r_dest_opt, hl_im))
}

pub fn compile_step_3<C: Config, I: InputType>(
    mut lc: layered::Circuit<C, I>,
    options: CompileOptions,
) -> Result<layered::Circuit<C, I>, Error> {
    lc.validate()
        .map_err(|e| e.prepend("layered circuit invalid"))?;

    if options.opt_level >= 1 {
        lc.dedup_gates();
    }
    if options.opt_level >= 3 {
        loop {
            let lc1 = lc.expand_small_segments();
            let lc2 = if lc1.segments.len() <= 100 {
                lc1.find_common_parts()
            } else {
                lc1
            };
            if lc2 == lc {
                break;
            }
            lc = lc2;
        }
    }
    lc.validate()
        .map_err(|e| e.prepend("layered circuit invalid1"))?;
    lc.sort_everything(); // for deterministic output
    Ok(lc)
}

pub fn compile_step_4<C: Config>(
    r_hint_exported: ir::hint_normalized::RootCircuit<C>,
    src_im: &mut InputMapping,
    options: CompileOptions,
) -> Result<ir::hint_normalized::RootCircuit<C>, Error> {
    r_hint_exported
        .validate()
        .map_err(|e| e.prepend("final hint exported circuit invalid"))?;
    let r_hint_exported_opt = if options.opt_level >= 2 {
        optimize_until_fixed_point(&r_hint_exported, src_im, |r| {
            let (r, im) = r.remove_unreachable();
            (r, im)
        })
    } else {
        r_hint_exported
    };
    Ok(r_hint_exported_opt)
}

pub fn compile<C: Config, I: InputType>(
    r_source: &ir::source::RootCircuit<C>,
) -> Result<(ir::hint_normalized::RootCircuit<C>, layered::Circuit<C, I>), Error> {
    compile_with_options(r_source, CompileOptions::default())
}

pub fn print_ir_stats<C: Config>(r_hint_normalized: &ir::hint_normalized::RootCircuit<C>) {
    let ho_stats = r_hint_normalized.get_stats();
    print_info("built hint normalized ir");
    print_stat("numInputs", ho_stats.num_inputs, false);
    print_stat("numConstraints", ho_stats.num_constraints, false);
    print_stat("numInsns", ho_stats.num_insns, false);
    print_stat("numVars", ho_stats.num_variables, false);
    print_stat("numTerms", ho_stats.num_terms, true);
}

pub fn print_layered_circuit_stats<C: Config, I: InputType>(lc: &layered::Circuit<C, I>) {
    let lc_stats = lc.get_stats();
    print_info("built layered circuit");
    print_stat("numSegment", lc_stats.num_segments, false);
    print_stat("numLayer", lc_stats.num_layers, false);
    print_stat("numUsedInputs", lc_stats.num_inputs, false);
    print_stat("numUsedVariables", lc_stats.num_used_gates, false);
    print_stat("numVariables", lc_stats.num_total_gates, false);
    print_stat("numAdd", lc_stats.num_expanded_add, false);
    print_stat("numCst", lc_stats.num_expanded_cst, false);
    print_stat("numMul", lc_stats.num_expanded_mul, false);
    print_stat("totalCost", lc_stats.total_cost, true);
}

pub fn compile_with_options<C: Config, I: InputType>(
    r_source: &ir::source::RootCircuit<C>,
    options: CompileOptions,
) -> Result<(ir::hint_normalized::RootCircuit<C>, layered::Circuit<C, I>), Error> {
    options.validate()?;

    let (r_hint_normalized_opt, mut src_im) = compile_step_1(r_source, options.clone())?;

    print_ir_stats(&r_hint_normalized_opt);

    let (r_hint_less, mut r_hint_exported) = r_hint_normalized_opt.remove_and_export_hints();
    r_hint_exported
        .validate()
        .map_err(|e| e.prepend("hint exported circuit invalid"))?;

    let (r_dest_opt, mut hl_im) = compile_step_2::<C, I>(r_hint_less, options.clone())?;

    let (lc, dest_im) = layering::compile(
        &r_dest_opt,
        layering::CompileOptions {
            allow_input_reorder: options.allow_input_reorder,
        },
    );

    let lc = compile_step_3(lc, options.clone())?;

    print_layered_circuit_stats(&lc);

    hl_im.compose_in_place(&dest_im);

    let rhe_c0 = r_hint_exported.circuits.get_mut(&0).unwrap();
    rhe_c0.outputs = hl_im
        .map_inputs(&rhe_c0.outputs)
        .iter()
        .map(|&x| x.max(1))
        .collect();

    let mut r_hint_exported_opt = compile_step_4(r_hint_exported, &mut src_im, options.clone())?;
    r_hint_exported_opt.add_back_removed_inputs(&src_im);
    r_hint_exported_opt
        .validate()
        .map_err(|e| e.prepend("final hint exported circuit invalid"))?;

    Ok((r_hint_exported_opt, lc))
}
