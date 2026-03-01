#pragma once

#include <cuda_runtime.h>
#include <chrono>

#include "circuit/circuit.cuh"
#include "field/M31.cuh"
#include "field/M31ext3.cuh"
#include "field/bn254.cuh"

namespace gkr{

    // GPU / CUDA knob
    static bool verbose = false;
    static bool useGPU = true;

    // Timing Breakdown
    struct TimingBreakdown{
        // Linear GKR, prepare time                 (ms)
        double prepare_time = 0.0;

        // PCIe transfer time                       (us)
        double pcie_time = 0.0;

        // Sum-check's Polynomial Evaluation time   (us)
        double polyeval_time = 0.0;

        // Sum-check's Fiat-Shamir Hash time        (ns)
        double fiathash_time = 0.0;

        // Sum-check's Receive Challenge time       (us)
        double challenge_time = 0.0;
    };

    template<typename F_primitive>
    __host__ __device__
    inline F_primitive _eq(const F_primitive& x, const F_primitive& y){
        // x * y + (1 - x) * (1 - y)
        return x * y * 2 - x - y + 1;
    }

    template<typename F_primitive>
    __global__
    void _eq_evals_kernel(const F_primitive* __restrict__ eq_evals_src,
                                F_primitive* __restrict__ eq_evals_dst,
                          const F_primitive* eq_z_i_one,
                          const F_primitive* eq_z_i_zero,
                                uint32_t nb_cur_evals){
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < nb_cur_evals){
            eq_evals_dst[idx + nb_cur_evals] = eq_evals_src[idx] * (*eq_z_i_one);
            eq_evals_dst[idx]                = eq_evals_src[idx] * (*eq_z_i_zero);
        }
    }

    template<typename F_primitive>
    void _eq_evals_at_primitive(const F_primitive* r,
                                const uint32_t & r_len,
                                const F_primitive& mul_factor,
                                F_primitive* eq_evals){
        eq_evals[0] = mul_factor;
        for (uint32_t i = 0; i < r_len; i++){
            uint32_t nb_cur_evals = 1 << i; // max(nb_cur_evals) = 1 << (r_len - 1)
            F_primitive eq_z_i_zero = _eq(r[i], F_primitive::zero());
            F_primitive eq_z_i_one  = _eq(r[i], F_primitive::one());
            // Runs on GPU
            if(useGPU && i > 10){
                // Define variables
                F_primitive* d_eq_evals_src;
                F_primitive* d_eq_evals_dst;
                F_primitive* d_eq_z_i_one;
                F_primitive* d_eq_z_i_zero;
                // Malloc space
                cudaMalloc((void **)&d_eq_evals_src, nb_cur_evals * sizeof(F_primitive));
                cudaMalloc((void **)&d_eq_evals_dst, 2 * nb_cur_evals * sizeof(F_primitive));
                cudaMalloc((void **)&d_eq_z_i_one, sizeof(F_primitive));
                cudaMalloc((void **)&d_eq_z_i_zero, sizeof(F_primitive));
                // Move input
                cudaMemcpy(d_eq_evals_src,    eq_evals,    nb_cur_evals * sizeof(F_primitive), cudaMemcpyHostToDevice);
                cudaMemcpy(d_eq_z_i_one,    &eq_z_i_one,    sizeof(F_primitive), cudaMemcpyHostToDevice);
                cudaMemcpy(d_eq_z_i_zero,    &eq_z_i_zero,    sizeof(F_primitive), cudaMemcpyHostToDevice);
                // Launch kernel
                uint32_t num_thread = 256;
                uint32_t num_block = (nb_cur_evals + num_thread - 1) / num_thread;
                _eq_evals_kernel<<<num_block, num_thread>>>(
                        d_eq_evals_src, d_eq_evals_dst,
                        d_eq_z_i_one, d_eq_z_i_zero,
                        nb_cur_evals);
                // Copy the result back
                cudaMemcpy(eq_evals,    d_eq_evals_dst,    2 * nb_cur_evals * sizeof(F_primitive), cudaMemcpyDeviceToHost);
                // Free cuda memory
                cudaFree(d_eq_evals_src);
                cudaFree(d_eq_evals_dst);
                cudaFree(d_eq_z_i_one);
                cudaFree(d_eq_z_i_zero);
            }else{
                // Too small, run on CPU
                for (uint32_t j = 0; j < nb_cur_evals; j++){
                    eq_evals[j + nb_cur_evals] = eq_evals[j] * eq_z_i_one;
                    eq_evals[j]                = eq_evals[j] * eq_z_i_zero;
                }
            }
        }
    }

    template<typename F_primitive>
    __global__
    void cross_prod_eq(const F_primitive* __restrict__ d_sqrtN1st,
                       const F_primitive* __restrict__ d_sqrtN2nd,
                             F_primitive* __restrict__ d_eq_evals,
                             uint32_t r_len
                             ){
        // Mimic what CPU does
        auto first_half_bits = r_len / 2;
        auto first_half_mask = (1 << first_half_bits) - 1;
        // Get the i loop variables
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < (1 << r_len)){
            uint32_t first_half  = i &  first_half_mask;
            uint32_t second_half = i >> first_half_bits;
            d_eq_evals[i] = d_sqrtN1st[first_half] * d_sqrtN2nd[second_half];
        }
    }

    // compute the multilinear extension eq(a, b) at
    // a = r, b = bit at all bits
    // the bits are interpreted as little endian numbers
    // The returned value is multiplied by the 'mul_factor' argument
    template<typename F_primitive>
    void _eq_evals_at(const F_primitive*    r,
                      const uint32_t&       r_len,
                      const F_primitive& mul_factor,
                      F_primitive* eq_evals,
                      F_primitive* sqrtN1st,
                      F_primitive* sqrtN2nd){

        auto first_half_bits = r_len / 2;
        auto first_half_mask = (1 << first_half_bits) - 1;

        _eq_evals_at_primitive(r, first_half_bits, mul_factor, sqrtN1st);
        _eq_evals_at_primitive(&r[first_half_bits], r_len - first_half_bits, F_primitive(1), sqrtN2nd);

        // Use GPU / CPU to do cross product of eq
        if(useGPU){
            // Prepare CUDA parameters
            uint32_t num_thread = 128;
            uint32_t num_block = ((1 << r_len) + num_thread - 1) / num_thread;
            // Malloc CUDA
            F_primitive* d_sqrtN1st;
            F_primitive* d_sqrtN2nd;
            F_primitive* d_eq_evals;
            cudaMalloc((void **)&d_sqrtN1st, sizeof(F_primitive) * (1 << first_half_bits));
            cudaMalloc((void **)&d_sqrtN2nd, sizeof(F_primitive) * (1 << (r_len - first_half_bits)));
            cudaMalloc((void **)&d_eq_evals, sizeof(F_primitive) * (1 << r_len));
            // Transfer input from Host to Device
            cudaMemcpy(d_sqrtN1st, sqrtN1st, sizeof(F_primitive) * (1 << first_half_bits), cudaMemcpyHostToDevice);
            cudaMemcpy(d_sqrtN2nd, sqrtN2nd, sizeof(F_primitive) * (1 << (r_len - first_half_bits)), cudaMemcpyHostToDevice);
            // Launch Kernel
            cross_prod_eq<<<num_block, num_thread>>>(d_sqrtN1st, d_sqrtN2nd, d_eq_evals, r_len);
            // Transfer output from device to host
            cudaMemcpy(eq_evals, d_eq_evals, sizeof(F_primitive) * (1 << r_len), cudaMemcpyDeviceToHost);
            // Free
            cudaFree(d_sqrtN1st);cudaFree(d_sqrtN2nd);cudaFree(d_eq_evals);
        }else{
            for (uint32_t i = 0; i < (uint32_t)(1 << r_len); i++){
                uint32_t first_half  = i &  first_half_mask;
                uint32_t second_half = i >> first_half_bits;
                eq_evals[i] = sqrtN1st[first_half] * sqrtN2nd[second_half];
            }
        }
    }
} // namespace gkr
