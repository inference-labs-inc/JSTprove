#pragma once

#include <vector>  // Only Verifier needs vector
#include <cassert>
#include <cstdio>

#include "fiat_shamir/transcript.cuh"
#include "circuit/circuit.cuh"
#include "sumcheck_helper.cuh"
#include "sumcheck_verifier_utils.cuh"

namespace gkr{

    template<typename F, typename F_primitive>
    void sumcheck_prove_gkr_layer(
            // Circuit
            const CircuitLayer<F, F_primitive>& poly,

            // rz1[rz1_outer_len][rz1_inner_len]
            const F_primitive* rz1, const uint32_t & rz1_inner_len,

            // rz2[rz2_outer_len][rz2_inner_len]
            const F_primitive* rz2, const uint32_t & rz2_inner_len,

            // Random Combination
            const F_primitive& alpha, const F_primitive& beta,

            // Proof Transcript
            Transcript<F, F_primitive>& transcript,

            // Scratchpad
            GKRScratchPad<F, F_primitive>& scratch_pad,

            // Return results
            F_primitive* rz1s, F_primitive* rz2s,

            // Timer
            TimingBreakdown& timer

    ){

        // Define the helper
        SumcheckGKRHelper<F, F_primitive> helper;

        // Timer
        auto total_prepare = std::chrono::milliseconds ::zero();
        auto total_polyeval = std::chrono::milliseconds ::zero();
        auto total_fiathash = std::chrono::nanoseconds ::zero();
        auto total_sumcheck = std::chrono::milliseconds ::zero();

        // Tic-Toc
        auto start = std::chrono::high_resolution_clock::now();
        auto end = std::chrono::high_resolution_clock::now();

        // Prepare for GKR
        start = std::chrono::high_resolution_clock::now();
        helper.prepare(poly,
                       rz1, rz1_inner_len,
                       rz2,  rz2_inner_len,
                       alpha, beta, scratch_pad);
        end = std::chrono::high_resolution_clock::now();
        timer.prepare_time += (double) std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // Sumcheck Main Loop
        for (uint32_t i_var = 0; i_var < (2 * poly.nb_input_vars); i_var++){
            // Prepare for Phase two (prepare Y)
            start = std::chrono::high_resolution_clock::now();
            if (i_var == poly.nb_input_vars){ helper._prepare_phase_two(); }
            end = std::chrono::high_resolution_clock::now();
            timer.prepare_time += (double) std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            // Polynomial Evluation
            F evals[3];
            helper.poly_evals_at(i_var, 2, evals, timer);

            // Fiat Shamir to get random challenge
            start = std::chrono::high_resolution_clock::now();
            transcript.append_f(evals[0]);
            transcript.append_f(evals[1]);
            transcript.append_f(evals[2]);
            auto r = transcript.challenge_f();
            end = std::chrono::high_resolution_clock::now();
            timer.fiathash_time += (double) std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

            // Evaluate on challenge
            helper.receive_challenge(i_var, r, timer);

            // If this is the final one of each phase, record it in transcript
            if (i_var == poly.nb_input_vars - 1){
                transcript.append_f(helper.vx_claim());
            }
        }


        transcript.append_f(helper.vy_claim());

        uint32_t rx_len = helper.rx_len;
        uint32_t ry_len = helper.ry_len;

        // Memory freed outside the function call
        rz1s = (F_primitive*) malloc(sizeof(F_primitive) * rx_len);
        rz2s = (F_primitive*) malloc(sizeof(F_primitive) * ry_len);

        for(int x_i = 0; x_i < rx_len; x_i++){
            rz1s[x_i] = helper.rx[x_i];
        }
        for(int y_i = 0; y_i < ry_len; y_i++){
            rz2s[y_i] = helper.ry[y_i];
        }

        // Print out timing breakdown at the end of proof
        std::cout << "Total LinearGKR Prepare:\t"     << (float) timer.prepare_time  / 1000.0 << "\ts" << std::endl;
        std::cout << "-------------------------------------------" << std::endl;
        std::cout << "Total CPU <> CUDA (PCIe):\t"     << (float) timer.pcie_time / 1000.0 << "\tms" << std::endl;
        std::cout << "-------------------------------------------" << std::endl;
        std::cout << "    - PolyEval:  \t\t"    << (float) timer.polyeval_time / 1000.0 << "\tms" << std::endl;
        std::cout << "    - Fiat-Shamir:  \t\t" << (float) timer.fiathash_time / 1000000.0 << "\tms" << std::endl;
        std::cout << "    - Challenge:  \t\t"   << (float) timer.challenge_time/ 1000.0 << "\tms" << std::endl;
        std::cout << "Total Sum-check:  \t\t"  <<
        (float) ((timer.challenge_time + timer.polyeval_time) / 1000.0 + (timer.fiathash_time / 1000000.0)) << "\tms" << std::endl;
        std::cout << "-------------------------------------------" << std::endl;
    }

    template<typename F, typename F_primitive>
    std::tuple<
    bool,
    std::vector<F_primitive>, std::vector<F_primitive>,
    F, F > sumcheck_verify_gkr_layer(
            const CircuitLayer<F, F_primitive>& poly,
            const F_primitive* rz1,
            const F_primitive* rz2,
            const F& claimed_v1,
            const F& claimed_v2,
            const F_primitive& alpha,
            const F_primitive& beta,
            Proof<F>& proof,
            Transcript<F, F_primitive>& transcript){

        // Start Verification
        uint32_t nb_vars = poly.nb_input_vars;
        F sum = claimed_v1 * alpha + claimed_v2 * beta;
        std::vector<F_primitive> rx, ry;
        std::vector<F_primitive> *rs = &rx;
        F vx_claim;

        bool verified = true;
        for (uint32_t i_var = 0; i_var < (2 * nb_vars); i_var++){
            const std::vector<F> low_degree_evals = {proof.get_next_and_step(), proof.get_next_and_step(), proof.get_next_and_step()};

            transcript.append_f(low_degree_evals[0]);
            transcript.append_f(low_degree_evals[1]);
            transcript.append_f(low_degree_evals[2]);
            auto r = transcript.challenge_f();

            (*rs).emplace_back(r);
            verified &= (low_degree_evals[0] + low_degree_evals[1]) == sum;
            sum = degree_2_eval(low_degree_evals, r);

            if (i_var == nb_vars - 1){
                auto start = std::chrono::high_resolution_clock::now();
                vx_claim = proof.get_next_and_step();
                sum -= vx_claim * eval_sparse_circuit_connect_poly<F, F_primitive, 1>(poly.add,
                                                                                      rz1, poly.nb_output_vars,
                                                                                      rz2, poly.nb_output_vars,
                                                                                      alpha, beta, {rx}
                );
                transcript.append_f(vx_claim);
                auto end = std::chrono::high_resolution_clock::now();
                float ms = (float) std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                if(verbose) printf("verify: X sparse eval time = %f s\n", ms/1000.0);
            }

            // Reach the end of phase one, switch to verify Y
            if (i_var == nb_vars - 1){
                rs = &ry;
            }
        }

        auto start = std::chrono::high_resolution_clock::now();
        F vy_claim = proof.get_next_and_step();
        verified &= sum == vx_claim * vy_claim * eval_sparse_circuit_connect_poly<F, F_primitive, 2>(poly.mul,
                                                                                                     rz1, poly.nb_output_vars,
                                                                                                     rz2, poly.nb_output_vars,
                                                                                                     alpha, beta, {rx, ry});
        transcript.append_f(vy_claim);
        auto end = std::chrono::high_resolution_clock::now();
        float ms = (float) std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        if(verbose) printf("verify: Y sparse eval time = %f s\n", ms/1000.0);

        // Return verification result
        return {verified, rx, ry, vx_claim, vy_claim};
    }
}
