// Copyright 2025 Irreducible Inc.
use std::{fs, path::Path};

use anyhow::Result;
use binius_core::constraint_system::{ConstraintSystem, ValueVec, ValuesData};
use binius_frontend::{CircuitBuilder, CircuitStat};
use binius_utils::serialization::{DeserializeBytes, SerializeBytes};
use clap::{Arg, Args, Command, FromArgMatches, Subcommand};

use crate::{CompressionType, ExampleCircuit, prove_verify, setup_sha256, setup_vision4};

/// Serialize a value implementing `SerializeBytes` and write it to the given path.
fn write_serialized<T: SerializeBytes>(value: &T, path: &str) -> Result<()> {
    if let Some(parent) = Path::new(path).parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|e| {
            anyhow::anyhow!("Failed to create directory '{}': {}", parent.display(), e)
        })?;
    }
    let mut buf: Vec<u8> = Vec::new();
    value.serialize(&mut buf)?;
    fs::write(path, &buf)
        .map_err(|e| anyhow::anyhow!("Failed to write serialized data to '{}': {}", path, e))?;
    Ok(())
}

/// Deserialize a value implementing `DeserializeBytes` from the given path.
fn read_deserialized<T: DeserializeBytes>(path: &str) -> Result<T> {
    let buf =
        fs::read(path).map_err(|e| anyhow::anyhow!("Failed to read file '{}': {}", path, e))?;
    T::deserialize(buf.as_slice())
        .map_err(|e| anyhow::anyhow!("Failed to deserialize data from '{}': {}", path, e))
}

/// A CLI builder for circuit examples that handles all command-line parsing and execution.
///
/// This provides a clean API for circuit examples where developers only need to:
/// 1. Implement the `ExampleCircuit` trait
/// 2. Define their `Params` and `Instance` structs with `#[derive(Args)]`
/// 3. Call `Cli::new("name").run()` in their main function
///
/// The CLI supports multiple subcommands:
/// - `prove` (default): Generate and verify a proof
/// - `stat`: Display circuit statistics
/// - `composition`: Output circuit composition in JSON format
/// - `check-snapshot`: Verify circuit statistics against a snapshot
/// - `bless-snapshot`: Update the snapshot with current statistics
///
/// # Example
///
/// ```rust,ignore
/// fn main() -> Result<()> {
///     Cli::<MyExample>::new("my_circuit")
///         .about("Description of my circuit")
///         .run()
/// }
/// ```
pub struct Cli<E: ExampleCircuit> {
    name: String,
    command: Command,
    _phantom: std::marker::PhantomData<E>,
}

/// Subcommands available for circuit examples
#[derive(Subcommand, Clone)]
enum Commands {
    /// Generate and verify a proof (default)
    Prove {
        /// Log of the inverse rate for the proof system
        #[arg(short = 'l', long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
        log_inv_rate: u32,

        /// Compression function to use
        #[arg(short = 'c', long, value_enum, default_value_t = CompressionType::Sha256)]
        compression: CompressionType,

        #[command(flatten)]
        params: CommandArgs,

        #[command(flatten)]
        instance: CommandArgs,
    },

    /// Display circuit statistics
    Stat {
        #[command(flatten)]
        params: CommandArgs,
    },

    /// Output circuit composition in JSON format
    Composition {
        #[command(flatten)]
        params: CommandArgs,
    },

    /// Verify circuit statistics against a snapshot
    CheckSnapshot {
        #[command(flatten)]
        params: CommandArgs,
    },

    /// Update the snapshot with current statistics
    BlessSnapshot {
        #[command(flatten)]
        params: CommandArgs,
    },

    /// Save constraint system, public witness, non-public data, and key collection to files if
    /// paths are provided
    Save {
        /// Output path for the constraint system binary
        #[arg(long = "cs-path")]
        cs_path: Option<String>,

        /// Output path for the public witness binary
        #[arg(long = "pub-witness-path")]
        pub_witness_path: Option<String>,

        /// Output path for the non-public data (witness + internal) binary
        #[arg(long = "non-pub-data-path")]
        non_pub_data_path: Option<String>,

        /// Output path for the key collection binary (for fast prover setup)
        #[arg(long = "key-collection-path")]
        key_collection_path: Option<String>,

        #[command(flatten)]
        params: CommandArgs,

        #[command(flatten)]
        instance: CommandArgs,
    },

    /// Load constraint system, witness data, and optionally key collection from files and prove
    ///
    /// If key-collection-path is provided, it will be loaded to skip the expensive
    /// key building phase during setup.
    LoadProve {
        /// Input path for the constraint system binary
        #[arg(long = "cs-path", required = true)]
        cs_path: String,

        /// Input path for the public witness binary
        #[arg(long = "pub-witness-path", required = true)]
        pub_witness_path: String,

        /// Input path for the non-public data (witness + internal) binary
        #[arg(long = "non-pub-data-path", required = true)]
        non_pub_data_path: String,

        /// Input path for the key collection binary (optional, for fast prover setup)
        #[arg(long = "key-collection-path")]
        key_collection_path: Option<String>,

        /// Log of the inverse rate for the proof system
        #[arg(
			short = 'l', long, default_value_t = 1,
			value_parser = clap::value_parser!(u32).range(1..)
		)]
        log_inv_rate: u32,
    },
}

/// Wrapper for dynamic command arguments
#[derive(Args, Clone)]
struct CommandArgs {
    #[arg(skip)]
    _phantom: (),
}

impl<E: ExampleCircuit> Cli<E>
where
    E::Params: Args,
    E::Instance: Args,
{
    /// Create a new CLI for the given circuit example.
    ///
    /// The `name` parameter sets the command name (shown in help and usage).
    pub fn new(name: &'static str) -> Self {
        let command = Command::new(name)
            .subcommand_required(false)
            .arg_required_else_help(false);

        // Build subcommands
        let prove_cmd = Self::build_prove_subcommand();
        let stat_cmd = Self::build_stat_subcommand();
        let composition_cmd = Self::build_composition_subcommand();
        let check_snapshot_cmd = Self::build_check_snapshot_subcommand();
        let bless_snapshot_cmd = Self::build_bless_snapshot_subcommand();
        let save_cmd = Self::build_save_subcommand();
        let load_prove_cmd = Self::build_load_prove_subcommand();

        let command = command
            .subcommand(prove_cmd)
            .subcommand(stat_cmd)
            .subcommand(composition_cmd)
            .subcommand(check_snapshot_cmd)
            .subcommand(bless_snapshot_cmd)
            .subcommand(save_cmd)
            .subcommand(load_prove_cmd);

        // Add top-level args for default prove behavior (when no subcommand specified)
        let command = command
            .arg(
                Arg::new("log_inv_rate")
                    .short('l')
                    .long("log-inv-rate")
                    .value_name("RATE")
                    .help("Log of the inverse rate for the proof system")
                    .default_value("1")
                    .value_parser(clap::value_parser!(u32).range(1..)),
            )
            .arg(
                Arg::new("compression")
                    .short('c')
                    .long("compression")
                    .value_name("TYPE")
                    .help("Compression function to use")
                    .value_parser(clap::value_parser!(CompressionType))
                    .default_value("sha256"),
            );

        // Augment with Params arguments at top level for default behavior
        let command = E::Params::augment_args(command);
        let command = E::Instance::augment_args(command);

        Self {
            name: name.to_string(),
            command,
            _phantom: std::marker::PhantomData,
        }
    }

    fn build_prove_subcommand() -> Command {
        let mut cmd = Command::new("prove")
            .about("Generate and verify a proof")
            .arg(
                Arg::new("log_inv_rate")
                    .short('l')
                    .long("log-inv-rate")
                    .value_name("RATE")
                    .help("Log of the inverse rate for the proof system")
                    .default_value("1")
                    .value_parser(clap::value_parser!(u32).range(1..)),
            )
            .arg(
                Arg::new("compression")
                    .short('c')
                    .long("compression")
                    .value_name("TYPE")
                    .help("Compression function to use")
                    .value_parser(clap::value_parser!(CompressionType))
                    .default_value("sha256"),
            );
        cmd = E::Params::augment_args(cmd);
        cmd = E::Instance::augment_args(cmd);
        cmd
    }

    fn build_stat_subcommand() -> Command {
        let cmd = Command::new("stat").about("Display circuit statistics");
        E::Params::augment_args(cmd)
    }

    fn build_composition_subcommand() -> Command {
        let cmd = Command::new("composition").about("Output circuit composition in JSON format");
        E::Params::augment_args(cmd)
    }

    fn build_check_snapshot_subcommand() -> Command {
        let cmd =
            Command::new("check-snapshot").about("Verify circuit statistics against a snapshot");
        E::Params::augment_args(cmd)
    }

    fn build_bless_snapshot_subcommand() -> Command {
        let cmd =
            Command::new("bless-snapshot").about("Update the snapshot with current statistics");
        E::Params::augment_args(cmd)
    }

    fn build_save_subcommand() -> Command {
        let mut cmd = Command::new("save").about(
			"Save constraint system, public witness, non-public data, and key collection to files if paths are provided",
		);
        cmd = cmd
            .arg(
                Arg::new("cs_path")
                    .long("cs-path")
                    .value_name("PATH")
                    .help("Output path for the constraint system binary"),
            )
            .arg(
                Arg::new("pub_witness_path")
                    .long("pub-witness-path")
                    .value_name("PATH")
                    .help("Output path for the public witness binary"),
            )
            .arg(
                Arg::new("non_pub_data_path")
                    .long("non-pub-data-path")
                    .value_name("PATH")
                    .help("Output path for the non-public data (witness + internal) binary"),
            )
            .arg(
                Arg::new("key_collection_path")
                    .long("key-collection-path")
                    .value_name("PATH")
                    .help("Output path for the key collection binary (for fast prover setup)"),
            );
        cmd = E::Params::augment_args(cmd);
        cmd = E::Instance::augment_args(cmd);
        cmd
    }

    fn build_load_prove_subcommand() -> Command {
        Command::new("load-prove")
			.about("Load constraint system, witness data, and optionally key collection from files and generate/verify proof")
			.arg(
				Arg::new("cs_path")
					.long("cs-path")
					.value_name("PATH")
					.help("Input path for the constraint system binary")
					.required(true),
			)
			.arg(
				Arg::new("pub_witness_path")
					.long("pub-witness-path")
					.value_name("PATH")
					.help("Input path for the public witness binary")
					.required(true),
			)
			.arg(
				Arg::new("non_pub_data_path")
					.long("non-pub-data-path")
					.value_name("PATH")
					.help("Input path for the non-public data (witness + internal) binary")
					.required(true),
			)
			.arg(
				Arg::new("key_collection_path")
					.long("key-collection-path")
					.value_name("PATH")
					.help("Input path for the key collection binary (optional, for fast prover setup)"),
			)
			.arg(
				Arg::new("log_inv_rate")
					.short('l')
					.long("log-inv-rate")
					.value_name("RATE")
					.help("Log of the inverse rate for the proof system")
					.default_value("1")
					.value_parser(clap::value_parser!(u32).range(1..)),
			)
			.arg(
				Arg::new("compression")
					.short('c')
					.long("compression")
					.value_name("TYPE")
					.help("Compression function to use")
					.value_parser(clap::value_parser!(CompressionType))
					.default_value("sha256"),
			)
    }

    /// Set the about/description text for the command.
    ///
    /// This appears in the help output.
    pub fn about(mut self, about: &'static str) -> Self {
        self.command = self.command.about(about);
        self
    }

    /// Set the long about text for the command.
    ///
    /// This appears in the detailed help output (--help).
    pub fn long_about(mut self, long_about: &'static str) -> Self {
        self.command = self.command.long_about(long_about);
        self
    }

    /// Set the version information for the command.
    pub fn version(mut self, version: &'static str) -> Self {
        self.command = self.command.version(version);
        self
    }

    /// Set the author information for the command.
    pub fn author(mut self, author: &'static str) -> Self {
        self.command = self.command.author(author);
        self
    }

    /// Run the circuit with parsed ArgMatches (implementation).
    #[allow(unused_variables)]
    fn run_with_matches_impl(matches: clap::ArgMatches, circuit_name: &str) -> Result<()> {
        // Initialize tracing once at the beginning for all commands
        let _tracing_guard = {
            #[cfg(feature = "perfetto")]
            {
                // Detect threading information
                let thread_count = binius_utils::rayon::current_num_threads();
                let thread_mode = if thread_count == 1 { "st" } else { "mt" };

                let mut builder = tracing_profile::TraceFilenameBuilder::for_benchmark(circuit_name)
						.output_dir("perfetto_traces")
						.timestamp() // Add timestamp for uniqueness
						.git_info() // Include git status
						.platform() // Include OS info
						.thread_mode(thread_mode);

                // Try to extract params from the appropriate matches for richer context
                // This will succeed for most commands (prove, stat, save, etc.)
                // and fail gracefully for commands without params (like load-prove)
                let matches_for_params =
                    matches.subcommand().map(|(_, sub)| sub).unwrap_or(&matches);

                if let Ok(params) = E::Params::from_arg_matches(matches_for_params)
                    && let Some(param_summary) = E::param_summary(&params)
                {
                    builder = builder.add("params", param_summary);
                }

                tracing_profile::init_tracing_with_builder(builder)?
            }
            #[cfg(not(feature = "perfetto"))]
            {
                tracing_profile::init_tracing()?
            }
        };

        // Check if a subcommand was used
        match matches.subcommand() {
            Some(("prove", sub_matches)) => Self::run_prove(sub_matches.clone()),
            Some(("stat", sub_matches)) => Self::run_stat(sub_matches.clone()),
            Some(("composition", sub_matches)) => Self::run_composition(sub_matches.clone()),
            Some(("check-snapshot", sub_matches)) => {
                Self::run_check_snapshot_impl(sub_matches.clone(), circuit_name)
            }
            Some(("bless-snapshot", sub_matches)) => {
                Self::run_bless_snapshot_impl(sub_matches.clone(), circuit_name)
            }
            Some(("save", sub_matches)) => Self::run_save(sub_matches.clone()),
            Some(("load-prove", sub_matches)) => Self::run_load_prove(sub_matches.clone()),
            Some((cmd, _)) => anyhow::bail!("Unknown subcommand: {}", cmd),
            None => {
                // No subcommand - default to prove behavior for backward compatibility
                Self::run_prove(matches)
            }
        }
    }

    fn run_prove(matches: clap::ArgMatches) -> Result<()> {
        // Extract common arguments
        let log_inv_rate = *matches
            .get_one::<u32>("log_inv_rate")
            .expect("has default value");
        let compression = matches
            .get_one::<CompressionType>("compression")
            .expect("has default value")
            .clone();
        tracing::info!("Parsed compression type: {compression:?}");

        // Parse Params and Instance from matches
        let params = E::Params::from_arg_matches(&matches)?;
        let instance = E::Instance::from_arg_matches(&matches)?;

        // Build the circuit
        let build_scope = tracing::info_span!("Building circuit").entered();
        let mut builder = CircuitBuilder::new();
        let example = E::build(params, &mut builder)?;
        let circuit = builder.build();
        drop(build_scope);

        // Set up prover and verifier
        let cs = circuit.constraint_system().clone();

        // Population of the input to the witness and then evaluating the circuit.
        let witness_population = tracing::info_span!(
            "Generating witness",
            operation = "witness_generation",
            perfetto_category = "operation"
        )
        .entered();
        let mut filler = circuit.new_witness_filler();
        tracing::info_span!("Input population")
            .in_scope(|| example.populate_witness(instance, &mut filler))?;
        tracing::info_span!("Circuit evaluation")
            .in_scope(|| circuit.populate_wire_witness(&mut filler))?;
        let witness = filler.into_value_vec();
        drop(witness_population);

        match compression {
            CompressionType::Sha256 => {
                tracing::info!("Using SHA256 compression for Merkle tree");
                let (verifier, prover) = setup_sha256(cs, log_inv_rate as usize, None)?;
                prove_verify(&verifier, &prover, witness)?;
            }
            CompressionType::Vision4 => {
                tracing::info!("Using Vision4 compression for Merkle tree");
                let (verifier, prover) = setup_vision4(cs, log_inv_rate as usize, None)?;
                prove_verify(&verifier, &prover, witness)?;
            }
        }

        Ok(())
    }

    fn run_stat(matches: clap::ArgMatches) -> Result<()> {
        // Parse Params from matches
        let params = E::Params::from_arg_matches(&matches)?;

        // Build the circuit
        let mut builder = CircuitBuilder::new();
        let _example = E::build(params, &mut builder)?;
        let circuit = builder.build();

        // Print statistics
        let stat = CircuitStat::collect(&circuit);
        print!("{}", stat);

        Ok(())
    }

    fn run_composition(matches: clap::ArgMatches) -> Result<()> {
        // Parse Params from matches
        let params = E::Params::from_arg_matches(&matches)?;

        // Build the circuit
        let mut builder = CircuitBuilder::new();
        let _example = E::build(params, &mut builder)?;
        let circuit = builder.build();

        // Print composition
        let dump = circuit.simple_json_dump();
        println!("{}", dump);

        Ok(())
    }

    fn run_check_snapshot_impl(matches: clap::ArgMatches, circuit_name: &str) -> Result<()> {
        // Parse Params from matches
        let params = E::Params::from_arg_matches(&matches)?;

        // Build the circuit
        let mut builder = CircuitBuilder::new();
        let _example = E::build(params, &mut builder)?;
        let circuit = builder.build();

        // Check snapshot
        crate::snapshot::check_snapshot(circuit_name, &circuit)?;

        Ok(())
    }

    fn run_bless_snapshot_impl(matches: clap::ArgMatches, circuit_name: &str) -> Result<()> {
        // Parse Params from matches
        let params = E::Params::from_arg_matches(&matches)?;

        // Build the circuit
        let mut builder = CircuitBuilder::new();
        let _example = E::build(params, &mut builder)?;
        let circuit = builder.build();

        // Bless snapshot
        crate::snapshot::bless_snapshot(circuit_name, &circuit)?;

        Ok(())
    }

    fn run_save(matches: clap::ArgMatches) -> Result<()> {
        // Extract optional output paths
        let cs_path = matches.get_one::<String>("cs_path").cloned();
        let pub_witness_path = matches.get_one::<String>("pub_witness_path").cloned();
        let non_pub_data_path = matches.get_one::<String>("non_pub_data_path").cloned();
        let key_collection_path = matches.get_one::<String>("key_collection_path").cloned();

        // If nothing to save, exit early
        if cs_path.is_none()
            && pub_witness_path.is_none()
            && non_pub_data_path.is_none()
            && key_collection_path.is_none()
        {
            tracing::info!("No output paths provided; nothing to save");
            return Ok(());
        }

        // Parse Params and Instance
        let params = E::Params::from_arg_matches(&matches)?;
        let instance = E::Instance::from_arg_matches(&matches)?;

        // Build circuit
        let mut builder = CircuitBuilder::new();
        let example = E::build(params, &mut builder)?;
        let circuit = builder.build();

        // Generate witness
        let mut filler = circuit.new_witness_filler();
        example.populate_witness(instance, &mut filler)?;
        circuit.populate_wire_witness(&mut filler)?;
        let witness: ValueVec = filler.into_value_vec();

        // Conditionally write artifacts
        let cs = circuit.constraint_system();
        if let Some(path) = cs_path.as_deref() {
            write_serialized(cs, path)?;
            tracing::info!("Constraint system saved to '{}'", path);
        }

        if let Some(path) = pub_witness_path.as_deref() {
            let data = ValuesData::from(witness.public());
            write_serialized(&data, path)?;
            tracing::info!("Public witness saved to '{}'", path);
        }

        if let Some(path) = non_pub_data_path.as_deref() {
            let data = ValuesData::from(witness.non_public());
            write_serialized(&data, path)?;
            tracing::info!("Non-public witness saved to '{}'", path);
        }

        // Save KeyCollection if requested
        if let Some(path) = key_collection_path.as_deref() {
            let key_collection_scope = tracing::info_span!("Building key collection").entered();
            let key_collection = binius_prover::protocols::shift::build_key_collection(cs);
            drop(key_collection_scope);
            write_serialized(&key_collection, path)?;
            tracing::info!("Key collection saved to '{}'", path);
        }

        Ok(())
    }

    fn run_load_prove(matches: clap::ArgMatches) -> Result<()> {
        // Extract file paths and parameters
        let cs_path = matches
            .get_one::<String>("cs_path")
            .expect("cs_path is required");
        let pub_witness_path = matches
            .get_one::<String>("pub_witness_path")
            .expect("pub_witness_path is required");
        let non_pub_data_path = matches
            .get_one::<String>("non_pub_data_path")
            .expect("non_pub_data_path is required");
        let key_collection_path = matches.get_one::<String>("key_collection_path").cloned();
        let log_inv_rate = *matches
            .get_one::<u32>("log_inv_rate")
            .expect("has default value");
        let compression = matches
            .get_one::<CompressionType>("compression")
            .expect("has default value")
            .clone();

        // Load constraint system
        let cs_load_scope = tracing::info_span!("Loading constraint system").entered();
        let cs: ConstraintSystem = read_deserialized(cs_path)?;
        tracing::info!("Constraint system loaded from '{}'", cs_path);
        drop(cs_load_scope);

        // Get the layout from the constraint system
        let layout = cs.value_vec_layout.clone();

        // Load pre-built KeyCollection if path provided
        let maybe_key_collection = key_collection_path
            .map(|kc_path| -> Result<_> {
                let kc_load_scope = tracing::info_span!("Loading key collection").entered();
                let key_collection: binius_prover::KeyCollection = read_deserialized(&kc_path)?;
                tracing::info!("Key collection loaded from '{kc_path}'");
                drop(kc_load_scope);
                Ok(key_collection)
            })
            .transpose()?;

        // Load witness data
        let witness_load_scope = tracing::info_span!("Loading witness data").entered();
        let pub_witness_data: ValuesData = read_deserialized(pub_witness_path)?;
        tracing::info!("Public witness loaded from '{}'", pub_witness_path);

        let non_pub_data: ValuesData = read_deserialized(non_pub_data_path)?;
        tracing::info!("Non-public data loaded from '{}'", non_pub_data_path);

        // Reconstruct the full witness using the layout
        let witness = ValueVec::new_from_data(
            layout,
            pub_witness_data.into_owned(),
            non_pub_data.into_owned(),
        )?;
        drop(witness_load_scope);

        match compression {
            CompressionType::Sha256 => {
                tracing::info!("Using SHA256 compression for Merkle tree");
                let (verifier, prover) =
                    setup_sha256(cs, log_inv_rate as usize, maybe_key_collection)?;
                prove_verify(&verifier, &prover, witness)?;
            }
            CompressionType::Vision4 => {
                tracing::info!("Using Vision4 compression for Merkle tree");
                let (verifier, prover) =
                    setup_vision4(cs, log_inv_rate as usize, maybe_key_collection)?;
                prove_verify(&verifier, &prover, witness)?;
            }
        };

        Ok(())
    }

    /// Parse arguments and run the circuit example.
    ///
    /// This orchestrates the entire flow:
    /// 1. Parse command-line arguments
    /// 2. Build the circuit using the params
    /// 3. Set up prover and verifier
    /// 4. Generate witness using the instance
    /// 5. Create and verify proof
    pub fn run(self) -> Result<()> {
        let name = self.name.clone();
        let matches = self.command.get_matches();
        Self::run_with_matches_impl(matches, &name)
    }

    /// Parse arguments and run with custom argument strings (useful for testing).
    ///
    /// This is similar to `run()` but takes explicit argument strings instead of
    /// reading from `std::env::args()`.
    pub fn run_from<I, T>(self, args: I) -> Result<()>
    where
        I: IntoIterator<Item = T>,
        T: Into<std::ffi::OsString> + Clone,
    {
        let name = self.name.clone();
        let matches = self.command.try_get_matches_from(args)?;
        Self::run_with_matches_impl(matches, &name)
    }
}
