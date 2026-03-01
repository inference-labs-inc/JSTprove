#![allow(clippy::type_complexity)]

use crate::zkcuda::context::ComputationGraph;
use crate::zkcuda::proving_system::expander::structs::{
    ExpanderProverSetup, ExpanderVerifierSetup,
};
use crate::zkcuda::proving_system::expander_parallelized::server_fns::ServerFns;
use crate::zkcuda::proving_system::expander_parallelized::shared_memory_utils::SharedMemoryEngine;

use axum::routing::{get, post};
use axum::Router;
use clap::Parser;
use expander_utils::timer::Timer;

use crate::frontend::{Config, SIMDField};

use axum::{extract::State, Json};
use gkr_engine::{GKREngine, MPIConfig, MPIEngine};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::sync::Mutex as SyncMutex;
use tokio::sync::{oneshot, Mutex};

pub static SERVER_IP: &str = "127.0.0.1";
pub static SERVER_PORT: Lazy<SyncMutex<u16>> = Lazy::new(|| SyncMutex::new(3000));

pub fn parse_port_number() -> u16 {
    let mut port = SERVER_PORT.lock().unwrap();
    *port = std::env::var("PORT_NUMBER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(*port);
    *port
}

#[derive(Serialize, Deserialize)]
pub enum RequestType {
    Setup(String),
    Prove,
    Exit,
}

pub struct ServerState<C: GKREngine, ECCConfig: Config<FieldConfig = C::FieldConfig>> {
    pub lock: Arc<Mutex<()>>,
    pub global_mpi_config: MPIConfig,

    pub prover_setup: Arc<Mutex<ExpanderProverSetup<C::FieldConfig, C::PCSConfig>>>,
    pub verifier_setup: Arc<Mutex<ExpanderVerifierSetup<C::FieldConfig, C::PCSConfig>>>,

    pub computation_graph: Arc<Mutex<ComputationGraph<ECCConfig>>>,
    pub witness: Arc<Mutex<Vec<Vec<SIMDField<C>>>>>,

    pub shutdown_tx: Arc<Mutex<Option<oneshot::Sender<()>>>>,
}

unsafe impl<C: GKREngine, ECCConfig: Config<FieldConfig = C::FieldConfig>> Send
    for ServerState<C, ECCConfig>
{
}

unsafe impl<C: GKREngine, ECCConfig: Config<FieldConfig = C::FieldConfig>> Sync
    for ServerState<C, ECCConfig>
{
}

impl<C: GKREngine, ECCConfig: Config<FieldConfig = C::FieldConfig>> Clone
    for ServerState<C, ECCConfig>
{
    fn clone(&self) -> Self {
        ServerState {
            lock: Arc::clone(&self.lock),
            global_mpi_config: self.global_mpi_config.clone(),
            prover_setup: Arc::clone(&self.prover_setup),
            verifier_setup: Arc::clone(&self.verifier_setup),
            computation_graph: Arc::clone(&self.computation_graph),
            witness: Arc::clone(&self.witness),
            shutdown_tx: Arc::clone(&self.shutdown_tx),
        }
    }
}

pub async fn root_main<C, ECCConfig, S>(
    State(state): State<ServerState<C, ECCConfig>>,
    Json(request_type): Json<RequestType>,
) -> Json<bool>
where
    C: GKREngine,
    ECCConfig: Config<FieldConfig = C::FieldConfig>,

    S: ServerFns<C, ECCConfig>,
{
    let _lock = state.lock.lock().await;
    match request_type {
        RequestType::Setup(setup_file) => {
            println!("Received setup request with file: {setup_file}");
            let setup_timer = Timer::new("server setup", true);

            let mut computation_graph = state.computation_graph.lock().await;
            let mut prover_setup_guard = state.prover_setup.lock().await;
            let mut verifier_setup_guard = state.verifier_setup.lock().await;
            S::setup_request_handler(
                &state.global_mpi_config,
                Some(setup_file),
                &mut computation_graph,
                &mut prover_setup_guard,
                &mut verifier_setup_guard,
            );

            SharedMemoryEngine::write_pcs_setup_to_shared_memory(&(
                prover_setup_guard.clone(),
                verifier_setup_guard.clone(),
            ));

            setup_timer.stop();
        }
        RequestType::Prove => {
            println!("Received prove request");
            let prove_timer = Timer::new("server prove", true);

            let mut witness = state.witness.lock().await;
            S::setup_shared_witness(&state.global_mpi_config, &mut witness);

            let prover_setup_guard = state.prover_setup.lock().await;
            let computation_graph = state.computation_graph.lock().await;

            let proof = S::prove_request_handler(
                &state.global_mpi_config,
                &*prover_setup_guard,
                &*computation_graph,
                &witness,
            );

            SharedMemoryEngine::write_proof_to_shared_memory(proof.as_ref().unwrap());
            prove_timer.stop();
        }
        RequestType::Exit => {
            println!("Received exit request, shutting down server");

            state
                .shutdown_tx
                .lock()
                .await
                .take()
                .map(|tx| tx.send(()).ok());
        }
    }

    axum::Json(true)
}

pub fn broadcast_request_type(global_mpi_config: &MPIConfig, request_type: u8) -> u8 {
    let mut bytes = vec![request_type];
    global_mpi_config.root_broadcast_bytes(&mut bytes);
    if bytes.len() != 1 {
        panic!("Failed to broadcast request type");
    }
    bytes[0]
}

pub fn generate_local_mpi_config(
    _global_mpi_config: &MPIConfig,
    _n_parties: usize,
) -> Option<MPIConfig> {
    Some(MPIConfig::prover_new())
}

pub async fn serve<C, ECCConfig, S>(port_number: String)
where
    C: GKREngine + 'static,
    ECCConfig: Config<FieldConfig = C::FieldConfig> + 'static,

    S: ServerFns<C, ECCConfig> + 'static,
{
    let global_mpi_config = MPIConfig::prover_new();

    let state = ServerState {
        lock: Arc::new(Mutex::new(())),
        global_mpi_config: global_mpi_config.clone(),
        prover_setup: Arc::new(Mutex::new(ExpanderProverSetup::default())),
        verifier_setup: Arc::new(Mutex::new(ExpanderVerifierSetup::default())),
        computation_graph: Arc::new(Mutex::new(ComputationGraph::default())),
        witness: Arc::new(Mutex::new(Vec::new())),
        shutdown_tx: Arc::new(Mutex::new(None)),
    };

    let (tx, rx) = oneshot::channel::<()>();
    state.shutdown_tx.lock().await.replace(tx);

    let app = Router::new()
        .route("/", post(root_main::<C, ECCConfig, S>))
        .route("/", get(|| async { "Expander Server is running" }))
        .with_state(state.clone());

    let ip: IpAddr = SERVER_IP.parse().expect("Invalid SERVER_IP");
    let port_val = port_number.parse::<u16>().unwrap_or_else(|e| {
        eprintln!("Error: Invalid port number '{port_number}'. {e}.");
        std::process::exit(1);
    });
    let addr = SocketAddr::new(ip, port_val);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    println!("Server running at http://{addr}");
    axum::serve(listener, app.into_make_service())
        .with_graceful_shutdown(async {
            rx.await.ok();
            println!("Shutting down server...");
        })
        .await
        .unwrap();

    loop {
        match Arc::strong_count(&state.computation_graph) {
            1 => {
                break;
            }
            _ => {
                println!("Waiting for server to shutdown...");
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        }
    }

    if state.global_mpi_config.is_root() {
        println!("Server has been shut down.");
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct ExpanderExecArgs {
    /// M31, GF2, BN254, Goldilocks, BabyBear
    #[arg(short, long, default_value = "M31")]
    pub field_type: String,

    /// Fiat-Shamir Hash: SHA256, or Poseidon, or MiMC5
    #[arg(short, long, default_value = "SHA256")]
    pub fiat_shamir_hash: String,

    /// Polynomial Commitment Scheme: Raw, or Orion
    #[arg(short, long, default_value = "Raw")]
    pub poly_commit: String,

    /// The port number for the server to listen on.
    #[arg(short, long, default_value = "3000")]
    pub port_number: String,

    /// Whether to batch PCS opening in proving.
    #[arg(short, long, default_value_t = false)]
    pub batch_pcs: bool,
}
