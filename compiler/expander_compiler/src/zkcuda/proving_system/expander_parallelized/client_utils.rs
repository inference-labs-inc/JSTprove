use std::fs;

use crate::{
    frontend::{Config, SIMDField},
    zkcuda::{
        context::ComputationGraph,
        proving_system::{
            expander::structs::{ExpanderProverSetup, ExpanderVerifierSetup},
            expander_parallelized::{
                cmd_utils::start_server, server_ctrl::parse_port_number,
                shared_memory_utils::SharedMemoryEngine,
            },
            CombinedProof, Expander,
        },
    },
};

use super::server_ctrl::{RequestType, SERVER_IP, SERVER_PORT};

use expander_utils::timer::Timer;
use gkr_engine::GKREngine;
use reqwest::Client;
use serdes::ExpSerde;

pub struct ClientHttpHelper;

impl ClientHttpHelper {
    pub async fn request_setup(setup_file: &str) {
        Self::post_request(RequestType::Setup(setup_file.to_string())).await;
    }

    pub async fn request_prove() {
        Self::post_request(RequestType::Prove).await;
    }

    pub async fn request_exit() {
        Self::post_request(RequestType::Exit).await;
    }

    pub async fn post_request(request: RequestType) {
        let client = Client::new();
        let port = {
            let port = SERVER_PORT.lock().unwrap();
            *port
        };
        let server_url = format!("{SERVER_IP}:{port}");
        let server_url = format!("http://{server_url}/");

        let res = client
            .post(server_url)
            .json(&request)
            .send()
            .await
            .expect("Failed to send request");

        if res.status().is_success() {
            println!("Request successful");
        } else {
            eprintln!("Request failed: {}", res.status());
        }
    }
}

pub fn client_parse_args() -> Option<String> {
    let args = std::env::args().collect::<Vec<_>>();
    let mut string = None;
    for (i, arg) in args.iter().take(args.len() - 1).enumerate() {
        if arg == "--server-binary" || arg == "-s" {
            string = Some(args[i + 1].clone());
            break;
        }
    }
    string
}

pub fn client_launch_server_and_setup<C, ECCConfig>(
    server_binary: &str,
    computation_graph: &ComputationGraph<ECCConfig>,
    batch_pcs: bool,
) -> (
    ExpanderProverSetup<C::FieldConfig, C::PCSConfig>,
    ExpanderVerifierSetup<C::FieldConfig, C::PCSConfig>,
)
where
    C: GKREngine,
    ECCConfig: Config<FieldConfig = C::FieldConfig>,
{
    let setup_timer = Timer::new("setup", true);
    println!("Starting server with binary: {server_binary}");

    let mut bytes = vec![];
    computation_graph.serialize_into(&mut bytes).unwrap();
    println!("Serialized computation graph, size: {}", bytes.len());

    // append current timestamp to the file name to avoid conflicts
    let setup_filename = format!(
        "/tmp/computation_graph_{}.bin",
        chrono::Utc::now().timestamp_millis()
    );
    fs::write(&setup_filename, bytes).expect("Failed to write computation graph to file");

    let port = parse_port_number();
    let server_url = format!("{SERVER_IP}:{port}");
    start_server::<C>(server_binary, port, batch_pcs);

    // Keep trying until the server is ready
    loop {
        match wait_async(Client::new().get(format!("http://{server_url}/")).send()) {
            Ok(_) => break,
            Err(_) => std::thread::sleep(std::time::Duration::from_secs(1)),
        }
    }

    wait_async(ClientHttpHelper::request_setup(&setup_filename));

    setup_timer.stop();

    SharedMemoryEngine::read_pcs_setup_from_shared_memory()
}

pub fn client_send_witness_and_prove<C, ECCConfig>(
    device_memories: Vec<Vec<SIMDField<ECCConfig>>>,
) -> CombinedProof<ECCConfig, Expander<C>>
where
    C: GKREngine,
    ECCConfig: Config<FieldConfig = C::FieldConfig>,
{
    let timer = Timer::new("prove", true);

    SharedMemoryEngine::write_witness_to_shared_memory::<C::FieldConfig>(device_memories);
    wait_async(ClientHttpHelper::request_prove());

    let proof = SharedMemoryEngine::read_proof_from_shared_memory();

    timer.stop();

    proof
}

/// Run an async function in a blocking context.
#[inline(always)]
pub fn wait_async<F, T>(f: F) -> T
where
    F: std::future::Future<Output = T>,
{
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(f)
}
