#include <cstdio>
#include <tuple>
#include <chrono>
#include <iostream>
#include <ctime>
#include <getopt.h>
#include <iomanip>

#include "LinearGKR/sumcheck.cuh"
#include "circuit/circuit.cuh"

// Function to display usage/help information
void print_usage() {
    std::cout << "Usage: ./sumcheck.bin -m [cpu|gpu] -p [2^(size) of circuit] [-v]" << std::endl;
}

int main(int argc, char* argv[]){
    // Seed the random number generator with the current time
    srand(time(nullptr));

    using namespace gkr;

    // Define optional parameters
    const char* mode = nullptr;
    uint32_t circuit_size = 20;

    // Parse command line options
    int opt;
    while ((opt = getopt(argc, argv, "m:p:v")) != -1) {
        switch (opt) {
            case 'm':
                mode = optarg;  // Get the argument for -m (mode)
                break;
            case 'p':
                circuit_size = std::atoi(optarg);
                break;
            case 'v':
                verbose = true;  // Set verbose to true if -v is passed
                break;
            default:
                print_usage();  // Print help if invalid options are passed
                return 1;
        }
    }

    // Set useGPU variable based on the mode argument
    if (mode != nullptr) {
        if (strcmp(mode, "gpu") == 0) {
            useGPU = true;  // Use GPU if "gpu" is specified
        } else if (strcmp(mode, "cpu") == 0) {
            useGPU = false;  // Use CPU if "cpu" is specified
        } else {
            std::cerr << "Invalid mode. Use 'cpu' or 'gpu'." << std::endl;
            print_usage();
            return 1;  // Exit if an invalid mode is passed
        }
    }

    // Output the current settings if verbose is enabled
    if (verbose) {
        std::cout << "Verbose mode enabled." << std::endl;
        std::cout << "Using " << (useGPU ? "GPU" : "CPU") << " for computation." << std::endl;
    }

    // Choose the Field
#ifdef useBN254
    // use BN254
    using F           = BN254_field::Bn254;
    using F_primitive = BN254_field::Bn254;
    auto field_type   = "BN254";
#else
#ifdef useM31ext3
    // use M31ext3
    using F           = M31ext3_field::M31ext3;
    using F_primitive = M31ext3_field::M31ext3;
    auto field_type   = "M31ext3";
#else
    using F           = M31_field::M31;
    using F_primitive = M31_field::M31;
    auto field_type   = "M31";
#endif
#endif
    std::cout   << "\n-------------------------------------------\n\t\t* Random *" << std::endl
                << "-------------------------------------------" << std::endl ;

    // Determine the size of circuit
    uint32_t nb_output_vars = circuit_size, nb_input_vars = circuit_size;

    // Create timer
    struct TimingBreakdown timer;

    // Create Random Circuit
    std::cout << "Randomizing Input ..." << std::endl;
    CircuitLayer<F, F_primitive> layer = CircuitLayer<F, F_primitive>::random(nb_output_vars, nb_input_vars);
    Circuit<F, F_primitive> circuit;
    circuit.add_layer(layer);
    std::cout << "Randomization Done!" << std::endl;

    // Evaluate the output
    std::cout << "Evaluating Output ..." << std::endl;
    uint32_t output_len = 1 << nb_output_vars;
    F* output = (F*) malloc(output_len * sizeof(F));
    layer.evaluate(output, output_len);
    std::cout << "Evaluation Done!" << std::endl;

    uint32_t rz1_len = nb_output_vars;
    uint32_t rz2_len = nb_output_vars;
    auto* rz1 = (F_primitive*) malloc (rz1_len * sizeof(F_primitive));
    auto* rz2 = (F_primitive*) malloc (rz2_len * sizeof(F_primitive));

    // Generate random numbers to commit
    std::cout << "Commit Output ..." << std::endl;
    for (uint32_t i = 0; i < nb_output_vars; i++){
        rz1[i] = F_primitive::random();
        rz2[i] = F_primitive::random();
    }
    F claim_v1 = eval_multilinear(output, output_len, rz1, nb_output_vars);
    F claim_v2 = eval_multilinear(output, output_len, rz2, nb_output_vars);
    std::cout << "Commit Done!" << std::endl;

    F_primitive alpha = F_primitive::random();
    F_primitive beta = F_primitive::random();

    GKRScratchPad<F, F_primitive> spad{};
    spad.prepare(circuit);

    // Define the result of sumcheck prove
    F_primitive* rz1s = nullptr;
    F_primitive* rz2s = nullptr;

    // Entering GPU proof generation
    auto start = std::chrono::high_resolution_clock::now();
    Transcript<F, F_primitive> prover_transcript;
    std::cout << std::fixed << std::setprecision(4); // Set precision
    std::cout   << "\n-------------------------------------------"  << std::endl;
    if(useGPU){
        std::cout   << "\t\t* GPU Prover *"                                 << std::endl;
    }else{
        std::cout   << "\t\t* CPU Prover *"                                 << std::endl;
    }
    std::cout   << "-------------------------------------------"    << std::endl ;
    sumcheck_prove_gkr_layer(layer,
                             rz1, nb_output_vars,
                             rz2, nb_output_vars,
                             alpha, beta,
                             prover_transcript,
                             spad,
                             rz1s, rz2s,
                             timer);
    Proof<F> &proof = prover_transcript.proof;
    free(rz1s); free(rz2s);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Input size: 2 ^ " << nb_input_vars << ", Output size: 2 ^ " << nb_output_vars << std::endl;
    std::cout << "Field type: " << field_type << std::endl;
    std::cout << "Prove time: " << (float)duration.count() / 1000 << " seconds" << std::endl;
    printf("Proof size: %u bytes\n", proof.bytes_write_ptr);

    // Doing Verification on CPU
    std::cout   << "\n-------------------------------------------\n\t\t* Verifier *" << std::endl
                << "-------------------------------------------" << std::endl ;
    Transcript<F, F_primitive> verifier_transcript;
    bool verified = std::get<0>(sumcheck_verify_gkr_layer(layer,
                                                          rz1, rz2,
                                                          claim_v1, claim_v2,
                                                          alpha, beta,
                                                          proof, verifier_transcript));
    printf("Verify pass = %d\n", verified);
}
