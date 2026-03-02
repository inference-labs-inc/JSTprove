#pragma once

#include "circuit/circuit.cuh"
#include "sumcheck_common.cuh"
#include "scratchpad.cuh"

#define MAX_RESULT_LEN  32

namespace gkr{
    // Try to acquire the lock until it succeeds
    __device__ void acquire_lock(uint32_t* lock_threadId, uint32_t x, uint32_t thread_index) {
        uint32_t old;
        do {
            old = atomicCAS(&lock_threadId[x], UINT32_MAX, thread_index);
        } while (old != UINT32_MAX);
        // Exit acquire after lock get
    }

    // Release the lock when update finished
    __device__ void release_lock(uint32_t* lock_threadId, uint32_t x) {
        atomicExch(&lock_threadId[x], UINT32_MAX);
    }

    // Accumulate the HG for add
    template<typename F, typename F_primitive>
    __global__
    void build_gx_add(      // Input
            const Gate<F_primitive, 1>* __restrict__ d_gate_sparse_evals,
            const F_primitive*          __restrict__ d_eq_evals_rz1,
            // Output
            F*                    __restrict__ d_hg_vals,
            bool*                 __restrict__ d_gate_exists,
            // Lock
            uint32_t *            __restrict__ d_lock_threadId,
            // Loop trip count
            uint32_t sparse_evals_len){
        // Calculate the loop variable
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < sparse_evals_len){
            const Gate<F_primitive, 1> &gate = d_gate_sparse_evals[i];
            uint32_t x = gate.i_ids[0];
            uint32_t z = gate.o_id;
            F coef = gate.coef;
            F_primitive eq_z = d_eq_evals_rz1[z];
            // Acquire lock
            acquire_lock(d_lock_threadId, x, i);
            // Do accumulation atomically
            F old_hg_val = d_hg_vals[x];
            F new_hg_val = old_hg_val + (coef * eq_z);
            d_hg_vals[x] = new_hg_val;
            d_gate_exists[x] = true;
            // Release lock
            release_lock(d_lock_threadId, x);
        }
    }

    // Accumulate the HG for mult
    template<typename F, typename F_primitive>
    __global__
    void build_gx_mult(      // Input
                       const Gate<F_primitive, 2>* __restrict__ d_gate_sparse_evals,
                       const F*                    __restrict__ d_val_evals,
                       const F_primitive*          __restrict__ d_eq_evals_rz1,
                             // Output
                             F*                    __restrict__ d_hg_vals,
                             bool*                 __restrict__ d_gate_exists,
                             // Lock
                             uint32_t *            __restrict__ d_lock_threadId,
                             // Loop trip count
                             uint32_t sparse_evals_len){
        // Calculate the loop variable
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < sparse_evals_len){
            // Read gate
            const Gate<F_primitive, 2> &gate = d_gate_sparse_evals[i];
            uint32_t x = gate.i_ids[0];
            uint32_t y = gate.i_ids[1];
            uint32_t z = gate.o_id;
            F coef     = gate.coef;
            F val_y    = d_val_evals[y];
            F_primitive eq_z = d_eq_evals_rz1[z];
            // Acquire lock from x
            acquire_lock(d_lock_threadId, x, i);
            // Update the hg_vals
            F old_hg_val = d_hg_vals[x];
            F new_hg_val = old_hg_val + val_y * (coef * eq_z);
            d_hg_vals[x] = new_hg_val;
            d_gate_exists[x] = true;
            // After update, release the lock to X
            release_lock(d_lock_threadId, x);
        }
    }

    // Accumulate the Hg for phase 2 of mult
    template<typename F, typename F_primitive>
    __global__
    void build_hy_mult(      // Input
            const Gate<F_primitive, 2>* __restrict__ d_gate_sparse_evals,
            const F_primitive*          __restrict__ d_eq_rx,
            const F_primitive*          __restrict__ d_eq_evals_rz1,
            const F*                    __restrict__ d_v_rx,
            // Output
            F*                    __restrict__ d_hg_vals,
            bool*                 __restrict__ d_gate_exists,
            // Lock
            uint32_t *            __restrict__ d_lock_threadId,
            // Loop trip count
            uint32_t sparse_evals_len){
        // Calculate the loop variable
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < sparse_evals_len){
            const Gate<F_primitive, 2> &gate = d_gate_sparse_evals[i];
            uint32_t x = gate.i_ids[0];
            uint32_t y = gate.i_ids[1];
            uint32_t z = gate.o_id;
            F v_rx = *d_v_rx;
            F_primitive rz1_z = d_eq_evals_rz1[z];
            F_primitive rx_x  = d_eq_rx[x];
            F coef            = gate.coef;
            // Acquire lock
            acquire_lock(d_lock_threadId, y, i);
            // Do actual compute
            F old_hg_val = d_hg_vals[y];
            F new_hg_val = old_hg_val + (v_rx * rz1_z * rx_x * coef); // g(y) += eq(rz, z) * eq(rx, x) * v(y) * coef
            d_hg_vals[y] = new_hg_val;
            d_gate_exists[y] = true;
            // Release lock
            release_lock(d_lock_threadId, y);
        }
    }

    // CUDA Kernel for Sum-check
    template<typename F, typename F_primitive>
    __global__
    void sumcheck_kernel(F_primitive*       d_r,                   // Challenge kernel received
                         F* __restrict__    d_src_v,               // Read  only
                         F* __restrict__    d_bookkeeping_f,       // Write only
                         F* __restrict__    d_bookkeeping_hg_src,  // Read  only
                         F* __restrict__    d_bookkeeping_hg_dst,  // Write only
                         uint32_t size
    ){
        // Get the loop variable
        uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;

        // Read the new random challenge
        F_primitive r = *d_r;

        // Do the same thing as main loop
        if(i < size){
            d_bookkeeping_f     [i] = d_src_v[2 * i]              + (d_src_v[2 * i + 1]              - d_src_v[2 * i]             ) * r;
            d_bookkeeping_hg_dst[i] = d_bookkeeping_hg_src[2 * i] + (d_bookkeeping_hg_src[2 * i + 1] - d_bookkeeping_hg_src[2 * i]) * r;
        }
    }

    // CUDA Kernel for Polynomial Evaluation
    template<typename F>
    __global__
    void poly_eval_kernel(F* __restrict__ d_src_v,
                          F* __restrict__ d_bookkeeping_hg,
                          F* __restrict__ d_block_results,
                          int evalSize){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int tid = threadIdx.x;

        // Arrange the shared memory
        extern __shared__ F s_data[];
        F* s_p0 = s_data;
        F* s_p1 = &s_data[blockDim.x];
        F* s_p2 = &s_data[2 * blockDim.x];

        s_p0[tid] = F::zero();
        s_p1[tid] = F::zero();
        s_p2[tid] = F::zero();

        if (idx < evalSize) {
            auto f_v_0 = d_src_v[idx * 2];
            auto f_v_1 = d_src_v[idx * 2 + 1];
            auto hg_v_0 = d_bookkeeping_hg[idx * 2];
            auto hg_v_1 = d_bookkeeping_hg[idx * 2 + 1];

            s_p0[tid] = f_v_0 * hg_v_0;
            s_p1[tid] = f_v_1 * hg_v_1;
            s_p2[tid] = (f_v_0 + f_v_1) * (hg_v_0 + hg_v_1);
        }

        __syncthreads();

        // Perform parallel reduction in shared memory
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_p0[tid] += s_p0[tid + stride];
                s_p1[tid] += s_p1[tid + stride];
                s_p2[tid] += s_p2[tid + stride];
            }
            __syncthreads();
        }

        // Write the block result to global memory
        if (tid == 0) {
            d_block_results[blockIdx.x * 3] = s_p0[0];
            d_block_results[blockIdx.x * 3 + 1] = s_p1[0];
            d_block_results[blockIdx.x * 3 + 2] = s_p2[0];
        }
    }

    template<typename F>
    __global__
    void reduce_blocks(const F* __restrict__ d_block_results_src,
                             F* __restrict__ d_block_results_dst,
                             uint32_t num_src_blocks) {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t tid = threadIdx.x;

        // Arrange the shared memory
        extern __shared__ F s_data[];
        F* s_p0 = s_data;
        F* s_p1 = &s_data[blockDim.x];
        F* s_p2 = &s_data[2 * blockDim.x];

        // Load data into shared memory
        if(idx < num_src_blocks) {
            s_p0[tid] = d_block_results_src[idx * 3];
            s_p1[tid] = d_block_results_src[idx * 3 + 1];
            s_p2[tid] = d_block_results_src[idx * 3 + 2];
        } else {
            s_p0[tid] = F::zero();
            s_p1[tid] = F::zero();
            s_p2[tid] = F::zero();
        }
        __syncthreads();

        // Perform parallel reduction in shared memory
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_p0[tid] += s_p0[tid + stride];
                s_p1[tid] += s_p1[tid + stride];
                s_p2[tid] += s_p2[tid + stride];
            }
            __syncthreads();
        }

        // Write the block result to global memory
        if (tid == 0) {
            d_block_results_dst[blockIdx.x * 3] = s_p0[0];
            d_block_results_dst[blockIdx.x * 3 + 1] = s_p1[0];
            d_block_results_dst[blockIdx.x * 3 + 2] = s_p2[0];
        }
    }

    template<typename F, typename F_primitive>
    class SumcheckMultiLinearProdHelper {
    public:
        uint32_t nb_vars;
        uint32_t sumcheck_var_idx;
        uint32_t cur_eval_size;
        F* bookkeeping_f;
        F* bookkeeping_hg;
        const F* initial_v;

        // CUDA device memory
        bool gpuMode = false;
        F* d_r;
        F* d_src_v;
        F* d_bookkeeping_f;
        F* d_bookkeeping_hg_src;
        F* d_bookkeeping_hg_dst;
        F* d_block_results;
        F* d_blocks_reduce;
        bool d_blocks_reduce_malloced = false;

        // Assign the pointer from scratchpad
        void prepare(uint32_t nb_vars_, F* p1_evals, F* p2_evals, const F* v){
            nb_vars = nb_vars_;
            sumcheck_var_idx = 0;
            cur_eval_size  = 1 << nb_vars;
            bookkeeping_f  = p1_evals;
            bookkeeping_hg = p2_evals;
            initial_v = v;
        }

        void poly_eval_kernel_wrapper(const F*  __restrict__ src_v,
                                      F& p0,
                                      F& p1,
                                      F& p2,
                                      int evalSize,
                                      uint32_t& var_idx,
                                      TimingBreakdown& timer){
            auto start = std::chrono::high_resolution_clock::now();

            // Define CUDA parameters
            int num_thread = (evalSize >= 512) ? 512 : (evalSize <= 32 ? 32 : evalSize);
            int num_block_src = (evalSize + num_thread - 1) / num_thread;

            if(var_idx == 0){
                // Allocate memory for src_v
                cudaMalloc((void **)&d_src_v, 2 * evalSize * sizeof(F));
                cudaMemcpy(d_src_v,    src_v,    2 * evalSize * sizeof(F), cudaMemcpyHostToDevice);

                // Allocate memory for bookkeeping_hg
                cudaMalloc((void **)&d_bookkeeping_hg_src, 2 * evalSize * sizeof(F));
                cudaMalloc((void **)&d_bookkeeping_hg_dst, evalSize * sizeof(F));
                cudaMemcpy(d_bookkeeping_hg_src,    bookkeeping_hg,    2 * evalSize * sizeof(F), cudaMemcpyHostToDevice);

                // Allocate memory for block results
                cudaMalloc((void **)&d_block_results, num_block_src * 3 * sizeof(F));
            }

            auto end = std::chrono::high_resolution_clock::now();

            timer.pcie_time += (double) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            start = std::chrono::high_resolution_clock::now();
            // Calculate the size of shared memory
            size_t sharedMemSize = 3 * num_thread * sizeof(F);

            // Launch Kernel
            poly_eval_kernel<<<num_block_src, num_thread, sharedMemSize>>>(
                    d_src_v,
                    (var_idx % 2 == 0) ? d_bookkeeping_hg_src : d_bookkeeping_hg_dst,
                    d_block_results,
                    evalSize
            );

            // Reduce over block results
            bool choose_reduce = false;
            int num_block_old = num_block_src;
            while(num_block_src > 1){
                int num_block_dst = (num_block_src + num_thread - 1) / num_thread;
                if(!d_blocks_reduce_malloced){
                    cudaMalloc((void **)&d_blocks_reduce, num_block_dst * 3 * sizeof(F));
                    d_blocks_reduce_malloced = true;
                }
                reduce_blocks<<<num_block_dst, num_thread, sharedMemSize>>>(
                        choose_reduce ? d_blocks_reduce : d_block_results,
                        choose_reduce ? d_block_results : d_blocks_reduce,
                        num_block_src
                        );
                choose_reduce = !choose_reduce;
                num_block_src = num_block_dst;
            }

            // Allocate host memory for block results and copy from device
            F* h_block_results = (F*)malloc(3 * sizeof(F));
            cudaMemcpy(h_block_results, choose_reduce ? d_blocks_reduce : d_block_results, 3 * sizeof(F), cudaMemcpyDeviceToHost);

            // Do accumulation on host
            p0 = h_block_results[0];
            p1 = h_block_results[1];
            p2 = h_block_results[2];

            // Clean up and record time
            free(h_block_results);
            end = std::chrono::high_resolution_clock::now();
            auto total = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            timer.polyeval_time += (double) total.count();

            // Debug print
            if(verbose) printf("#block = %d, #thread = %d, time = %.1f us\n", num_block_old, num_thread, (float) total.count());
        }

        // Polynominal Evaluation
        void poly_eval_at(uint32_t var_idx, uint32_t degree, const bool *gate_exists, F* evals, TimingBreakdown& timer){
            F p0 = F::zero();
            F p1 = F::zero();
            F p2 = F::zero();
            const F* src_v = (var_idx == 0 ? initial_v : bookkeeping_f);
            int evalSize = 1 << (nb_vars - var_idx - 1);

            // Switch between GPU vs. CPU implementation
            if(useGPU){
                if(verbose) printf("CUDA: poly_eval_at : var_idx = %u, eval_size = %d, ", var_idx, evalSize);
                poly_eval_kernel_wrapper(src_v, p0, p1, p2, evalSize, var_idx, timer);
            }else{
                auto start = std::chrono::high_resolution_clock::now();
                if(verbose) printf("CPU: poly_eval_at : var_idx = %u, eval_size = %d\n", var_idx, evalSize);
                for (int i = 0; i < evalSize; i++){
                    if (!gate_exists[i * 2] && !gate_exists[i * 2 + 1]){ continue; }
                    auto f_v_0      = src_v[i * 2];
                    auto f_v_1      = src_v[i * 2 + 1];
                    auto hg_v_0 = bookkeeping_hg[i * 2];
                    auto hg_v_1 = bookkeeping_hg[i * 2 + 1];
                    p0 += f_v_0 * hg_v_0;
                    p1 += f_v_1 * hg_v_1;
                    p2 += (f_v_0 + f_v_1) * (hg_v_0 + hg_v_1);
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto total = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                timer.polyeval_time += (double) total.count();
            }

            // Compute final poly evaluation results
            p2 = p1 * F(6) + p0 * F(3) - p2 * F(2);
            evals[0] = p0;
            evals[1] = p1;
            evals[2] = p2;
        }

        // Receive Challenge of MLE Helper
        void receive_challenge(uint32_t var_idx,     // Index variable, nothing to do with computation
                               const F_primitive& r, // Random challenge
                               bool *gate_exists,     // Existence of gates
                               TimingBreakdown& timer
                               ){
            // Select the source
            auto* src_v = (var_idx == 0 ? initial_v : bookkeeping_f);

            // Sanity check
            assert(var_idx == sumcheck_var_idx && var_idx < nb_vars);

            // Define CUDA managed memory if it is the first iteration
            if(useGPU && var_idx == 0){
                gpuMode = true;
                // Memory Allocation on GPU
                cudaMalloc((void **)&d_r,                  sizeof(F));
                cudaMalloc((void **)&d_bookkeeping_f,      (cur_eval_size >> 1) * sizeof(F)); // write-only
            }

            // Switch between CUDA and CPU
            if(gpuMode){
                auto start = std::chrono::high_resolution_clock::now();

                // Memory copy from Host to Device
                cudaMemcpy(d_r,&r, sizeof(F), cudaMemcpyHostToDevice);

                // Launch Kernel
                int eval_size = cur_eval_size >> 1;
                int num_thread = (eval_size >= 512) ? 512 : (eval_size <= 32 ? 32 : eval_size);
                int num_block  = (eval_size + num_thread - 1) / num_thread;

                sumcheck_kernel<<<num_block, num_thread>>>(
                        d_r,
                        d_src_v,
                        d_bookkeeping_f,
                        (var_idx % 2 == 0) ? d_bookkeeping_hg_src : d_bookkeeping_hg_dst,
                        (var_idx % 2 == 0) ? d_bookkeeping_hg_dst : d_bookkeeping_hg_src,
                        eval_size
                );
                cudaDeviceSynchronize(); // No-need to make functional correct, but necessary for time measure

                // Copy result back
                cudaMemcpy(d_src_v,  d_bookkeeping_f, eval_size * sizeof(F),    cudaMemcpyDeviceToDevice);
                auto end = std::chrono::high_resolution_clock::now();
                auto total = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                timer.challenge_time += ((double) total.count());
                if(verbose) printf("CUDA: receive_chal : var_idx = %u, eval_size = %u, #block = %d, #thread = %d, time = %.1f us\n",
                                   var_idx, eval_size, num_block, num_thread, (float) total.count());
            }else{
                auto start = std::chrono::high_resolution_clock::now();
                for (uint32_t i = 0; i < (cur_eval_size >> 1); i++){
                    if (!gate_exists[2 * i] && !gate_exists[2 * i + 1]){
                        gate_exists   [i] = false;
                        bookkeeping_f [i] = src_v[2 * i]          + (src_v[2 * i + 1]          - src_v[2 * i]         ) * r;
                        bookkeeping_hg[i] = 0;
                    }else{
                        gate_exists   [i] = true;
                        bookkeeping_f [i] = src_v[2 * i]          + (src_v[2 * i + 1]          - src_v[2 * i]         ) * r;
                        bookkeeping_hg[i] = bookkeeping_hg[2 * i] + (bookkeeping_hg[2 * i + 1] - bookkeeping_hg[2 * i]) * r;
                    }
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto total = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                timer.challenge_time += ((double) total.count());
                if(verbose) printf("CPU: receive_chal : var_idx = %u, eval_size = %u, time = %.1f us\n", var_idx, cur_eval_size >> 1, (float) total.count());
            }

            // Turn off the CUDA if workload size if too small
            if(gpuMode && (cur_eval_size >> 1) == 1){
                gpuMode = false;
                // Copy back the final v claim
                cudaMemcpy(bookkeeping_f,  d_bookkeeping_f, sizeof(F),    cudaMemcpyDeviceToHost);
                // Free all CUDA memory
                cudaFree(d_r);
                cudaFree(d_src_v);
                cudaFree(d_bookkeeping_f);
                cudaFree(d_bookkeeping_hg_src);
                cudaFree(d_bookkeeping_hg_dst);
                cudaFree(d_block_results);
                if(d_blocks_reduce_malloced) {
                    d_blocks_reduce_malloced = false;
                    cudaFree(d_blocks_reduce);
                }
            }

            cur_eval_size >>= 1;
            sumcheck_var_idx++;
        }
    };

    template <typename  F_primitive>
    __global__
    void vecadd(const F_primitive* __restrict__ d_A,
                const F_primitive* __restrict__ d_B,
                F_primitive* __restrict__ d_C,
                uint32_t len){
        uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
        if(i < len) d_C[i] = d_A[i] + d_B[i];
    }

    template<typename F, typename F_primitive>
    class SumcheckGKRHelper{
    public:

        CircuitLayer<F, F_primitive> const* poly_ptr;
        F_primitive alpha, beta;
        GKRScratchPad<F, F_primitive>* pad_ptr;
        F_primitive rx[MAX_RESULT_LEN]; uint32_t rx_len = 0;
        F_primitive ry[MAX_RESULT_LEN]; uint32_t ry_len = 0;
        SumcheckMultiLinearProdHelper<F, F_primitive> x_helper, y_helper;
        uint32_t nb_input_vars;
        uint32_t nb_output_vars;

        // CUDA managed memory
        F_primitive*          d_eq_evals_rz1;
        F*                    d_hg_vals;
        bool*                 d_gate_exists;
        uint32_t *            d_lock_threadId;

        void _prepare_g_x_vals(
                const F_primitive* rz1, const uint32_t & rz1_len,
                const F_primitive* rz2, const uint32_t & rz2_len,
                const F_primitive& alpha,
                const F_primitive& beta,
                const SparseCircuitConnection<F_primitive, 2>& mul,
                const SparseCircuitConnection<F_primitive, 1>& add,
                const MultiLinearPoly<F>& vals,
                bool* gate_exists){
            F *hg_vals = pad_ptr->hg_evals;

            for(int i = 0; i < vals.evals_len; i++){ hg_vals[i] = 0; }
            for(int i = 0; i < vals.evals_len; i++){ gate_exists[i] = false; }

            // CPU @ 2^28, 1.45s
            // GPU (with cross prod eq) @ 2 ^ 28, 0.49s
            auto start = std::chrono::high_resolution_clock::now();
            _eq_evals_at(rz1, rz1_len, alpha, pad_ptr->eq_evals_at_rz1, pad_ptr -> eq_evals_first_half, pad_ptr -> eq_evals_second_half);
            _eq_evals_at(rz2, rz2_len, beta, pad_ptr->eq_evals_at_rz2, pad_ptr -> eq_evals_first_half, pad_ptr -> eq_evals_second_half);
            F_primitive * eq_evals_at_rz1 = pad_ptr->eq_evals_at_rz1;
            F_primitive const* eq_evals_at_rz2 = pad_ptr->eq_evals_at_rz2;
            auto end = std::chrono::high_resolution_clock::now();
            auto total = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "    - phase 1: two eq evals \t" << (float) total.count() / 1000.0 << "\ts" << std::endl;

            // CPU @ 2^28, 0.85s
            // GPU (with vecadd) @ 2 ^ 28, 0.289s
            start = std::chrono::high_resolution_clock::now();
            if(useGPU){
                // Prepare CUDA parameters
                uint32_t num_thread = 128;
                uint32_t num_block = ((1 << rz1_len) + num_thread - 1) / num_thread;
                // Malloc CUDA
                F_primitive* d_A;
                F_primitive* d_B;
                F_primitive* d_C;
                cudaMalloc((void **)&d_A, sizeof(F_primitive) * (1 << rz1_len));
                cudaMalloc((void **)&d_B, sizeof(F_primitive) * (1 << rz1_len));
                cudaMalloc((void **)&d_C, sizeof(F_primitive) * (1 << rz1_len));
                // Move to GPU
                cudaMemcpy(d_A, eq_evals_at_rz1, sizeof(F_primitive) * (1 << rz1_len), cudaMemcpyHostToDevice);
                cudaMemcpy(d_B, eq_evals_at_rz2, sizeof(F_primitive) * (1 << rz1_len), cudaMemcpyHostToDevice);
                // Launch kernel
                vecadd<<<num_block, num_thread>>>(d_A, d_B, d_C, 1 << rz1_len);
                // Copy back
                cudaMemcpy(eq_evals_at_rz1, d_C, sizeof(F_primitive) * (1 << rz1_len), cudaMemcpyDeviceToHost);
                // Free
                cudaFree(d_A);cudaFree(d_B);cudaFree(d_C);
            }else{
                for (int i = 0; i < (1 << rz1_len); ++i){
                    eq_evals_at_rz1[i] = eq_evals_at_rz1[i] + eq_evals_at_rz2[i];
                }
            }
            end = std::chrono::high_resolution_clock::now();
            total = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "    - phase 1: vec addition \t" << (float) total.count() / 1000.0 << "\ts" << std::endl;

            // CPU @ 2^28, 0.49s
            start = std::chrono::high_resolution_clock::now();
            if(useGPU){
                // Define CUDA kernel variables
                Gate<F_primitive, 2>* d_mult_sparse_evals;
                F*                    d_val_evals;
                // Malloc CUDA region
                cudaMalloc((void **)&d_mult_sparse_evals, sizeof(Gate<F_primitive, 2>) * mul.sparse_evals_len);
                cudaMalloc((void **)&d_val_evals,         sizeof(F) * vals.evals_len);
                cudaMalloc((void **)&d_eq_evals_rz1,      sizeof(F_primitive) * (1 << rz1_len));
                cudaMalloc((void **)&d_hg_vals, sizeof(F) * vals.evals_len);
                cudaMalloc((void **)&d_gate_exists, sizeof(bool) * vals.evals_len);
                cudaMalloc((void **)&d_lock_threadId, sizeof(uint32_t) * vals.evals_len);
                // Transfer data to GPU
                cudaMemcpy(d_mult_sparse_evals, mul.sparse_evals, sizeof(Gate<F_primitive, 2>) * mul.sparse_evals_len, cudaMemcpyHostToDevice);
                cudaMemcpy(d_val_evals, vals.evals, sizeof(F) * vals.evals_len, cudaMemcpyHostToDevice);
                cudaMemcpy(d_eq_evals_rz1, eq_evals_at_rz1, sizeof(F_primitive) * (1 << rz1_len), cudaMemcpyHostToDevice);
                cudaMemset(d_hg_vals, 0x00, sizeof(F) * vals.evals_len);              // HG init
                cudaMemset(d_gate_exists, 0x0, sizeof(bool) * vals.evals_len);        // gate exists init
                cudaMemset(d_lock_threadId, 0xFF, vals.evals_len * sizeof(uint32_t)); // Init locks
                // Start CUDA kernel
                uint32_t num_thread = 128;
                uint32_t num_block = (mul.sparse_evals_len + num_thread - 1) / num_thread;
                build_gx_mult<<<num_block, num_thread>>>(
                        d_mult_sparse_evals, d_val_evals, d_eq_evals_rz1,
                        d_hg_vals, d_gate_exists, d_lock_threadId,
                        mul.sparse_evals_len);
                cudaDeviceSynchronize();
                // Transfer result back to CPU
                cudaMemcpy(hg_vals, d_hg_vals, sizeof(F) * vals.evals_len, cudaMemcpyDeviceToHost);
                cudaMemcpy(gate_exists, d_gate_exists, sizeof(bool) * vals.evals_len, cudaMemcpyDeviceToHost);
                // Release and free GPU memory
                cudaFree(d_mult_sparse_evals);
                cudaFree(d_val_evals);
            }else{
                for(long unsigned int i = 0; i < mul.sparse_evals_len; i++){
                    // g(x) += eq(rz, z) * v(y) * coef
                    const Gate<F_primitive, 2> &gate = mul.sparse_evals[i];
                    uint32_t x = gate.i_ids[0];
                    uint32_t y = gate.i_ids[1];
                    uint32_t z = gate.o_id;
                    hg_vals[x] += vals.evals[y] * (gate.coef * eq_evals_at_rz1[z]);
                    gate_exists[x] = true;
                }
            }
            end = std::chrono::high_resolution_clock::now();
            total = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "    - phase 1: build gx(mult) \t" << (float) total.count() / 1000.0 << "\ts" << std::endl;

            // CPU @ 2^28, 1.2s
            start = std::chrono::high_resolution_clock::now();
            if(useGPU){
                // Define CUDA kernel variables
                Gate<F_primitive, 1>* d_add_sparse_evals;
                // Malloc CUDA region
                cudaMalloc((void **)&d_add_sparse_evals, sizeof(Gate<F_primitive, 1>) * add.sparse_evals_len);
                // Transfer data to GPU
                cudaMemcpy(d_add_sparse_evals, add.sparse_evals, sizeof(Gate<F_primitive, 1>) * add.sparse_evals_len, cudaMemcpyHostToDevice);
                // Start CUDA kernel
                uint32_t num_thread = 128;
                uint32_t num_block = (add.sparse_evals_len + num_thread - 1) / num_thread;
                build_gx_add<<<num_block, num_thread>>>(d_add_sparse_evals, d_eq_evals_rz1,
                                                        d_hg_vals, d_gate_exists, d_lock_threadId, add.sparse_evals_len);
                cudaDeviceSynchronize();
                // Transfer result back to CPU
                cudaMemcpy(hg_vals, d_hg_vals, sizeof(F) * vals.evals_len, cudaMemcpyDeviceToHost);
                cudaMemcpy(gate_exists, d_gate_exists, sizeof(bool) * vals.evals_len, cudaMemcpyDeviceToHost);
                // Release and free CUDA memory
                cudaFree(d_add_sparse_evals);
                cudaFree(d_eq_evals_rz1);
                cudaFree(d_hg_vals);
                cudaFree(d_gate_exists);
                cudaFree(d_lock_threadId);
            }else{
                for(long unsigned int i = 0; i < add.sparse_evals_len; i++){
                    // g(x) += eq(rz, x) * coef
                    const Gate<F_primitive, 1> &gate = add.sparse_evals[i];
                    uint32_t x = gate.i_ids[0];
                    uint32_t z = gate.o_id;
                    hg_vals[x] += gate.coef * eq_evals_at_rz1[z];
                    gate_exists[x] = true;
                }
            }
            end = std::chrono::high_resolution_clock::now();
            total = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "    - phase 1: build gx(add) \t" << (float) total.count() / 1000.0 << "\ts" << std::endl;
        }

        void _prepare_h_y_vals(
                const F& v_rx,
                const SparseCircuitConnection<F_primitive, 2>& mul,
                bool *gate_exists){
            auto start = std::chrono::high_resolution_clock::now();
            F *hg_vals = pad_ptr->hg_evals;
            // Reset hg_vals;
            for(int i = 0; i < (1 << rx_len); i++){
                hg_vals[i] = 0;
                gate_exists[i] = false;
            }

            F_primitive const* eq_evals_at_rz1 = pad_ptr->eq_evals_at_rz1; // already computed in g_x preparation
            _eq_evals_at(rx, rx_len, F_primitive::one(), pad_ptr->eq_evals_at_rx, pad_ptr -> eq_evals_first_half, pad_ptr -> eq_evals_second_half);
            F_primitive const* eq_evals_at_rx = pad_ptr->eq_evals_at_rx;

            if(useGPU){
                // Define CUDA kernel variables
                Gate<F_primitive, 2>* d_mult_sparse_evals;
                F_primitive*          d_eq_rx;
                F*                    d_v_rx;
                // Malloc CUDA region
                cudaMalloc((void **)&d_mult_sparse_evals,              sizeof(Gate<F_primitive, 2>) * mul.sparse_evals_len);
                cudaMalloc((void **)&d_eq_rx,             sizeof(F_primitive)          * (1 << rx_len));
                cudaMalloc((void **)&d_v_rx,              sizeof(F)                    * 1);
                cudaMalloc((void **)&d_eq_evals_rz1,      sizeof(F_primitive)          * (1 << rx_len));
                cudaMalloc((void **)&d_hg_vals,           sizeof(F)                    * (1 << rx_len));
                cudaMalloc((void **)&d_gate_exists,       sizeof(bool)                 * (1 << rx_len));
                cudaMalloc((void **)&d_lock_threadId,     sizeof(uint32_t)             * (1 << rx_len));
                // Transfer data to CUDA
                cudaMemcpy(d_mult_sparse_evals, mul.sparse_evals, sizeof(Gate<F_primitive, 2>) * mul.sparse_evals_len, cudaMemcpyHostToDevice);
                cudaMemcpy(d_eq_rx, eq_evals_at_rx, sizeof(F_primitive)          * (1 << rx_len), cudaMemcpyHostToDevice);
                cudaMemcpy(d_v_rx, &v_rx, sizeof(F), cudaMemcpyHostToDevice);
                cudaMemcpy(d_eq_evals_rz1, eq_evals_at_rz1, sizeof(F_primitive)          * (1 << rx_len), cudaMemcpyHostToDevice);
                cudaMemset(d_hg_vals, 0x00, sizeof(F) * (1 << rx_len) );              // HG init
                cudaMemset(d_gate_exists, 0x0, sizeof(bool) * (1 << rx_len) );        // gate exists init
                cudaMemset(d_lock_threadId, 0xFF, sizeof(uint32_t) * (1 << rx_len)); // Init locks
                // Start Kernel
                uint32_t num_thread = 128;
                uint32_t num_block = (mul.sparse_evals_len + num_thread - 1) / num_thread;
                build_hy_mult<<<num_block, num_thread>>>(
                        d_mult_sparse_evals, d_eq_rx, d_eq_evals_rz1, d_v_rx,
                        d_hg_vals, d_gate_exists,
                        d_lock_threadId,
                        mul.sparse_evals_len
                        );
                cudaDeviceSynchronize();
                // Transfer result back
                cudaMemcpy(hg_vals, d_hg_vals, sizeof(F) * (1 << rx_len), cudaMemcpyDeviceToHost);
                cudaMemcpy(gate_exists, d_gate_exists, sizeof(bool) * (1 << rx_len), cudaMemcpyDeviceToHost);
                // Cleanup CUDA memory region
                cudaFree(d_mult_sparse_evals);
                cudaFree(d_eq_rx);
                cudaFree(d_v_rx);
                cudaFree(d_eq_evals_rz1);
                cudaFree(d_hg_vals);
                cudaFree(d_gate_exists);
                cudaFree(d_lock_threadId);
            }else{
                for(int i = 0; i < mul.sparse_evals_len; i++){
                    const Gate<F_primitive, 2> &gate = mul.sparse_evals[i];
                    // g(y) += eq(rz, z) * eq(rx, x) * v(y) * coef
                    uint32_t x = gate.i_ids[0];
                    uint32_t y = gate.i_ids[1];
                    uint32_t z = gate.o_id;

                    hg_vals[y] += v_rx * (eq_evals_at_rz1[z] * eq_evals_at_rx[x] * gate.coef);
                    gate_exists[y] = true;
                }
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto total = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "    - phase 2: build hy(mult) \t" << (float) total.count() / 1000.0 << "\ts" << std::endl;
        }

        void _prepare_phase_two(){
            _prepare_h_y_vals(vx_claim(), poly_ptr->mul, pad_ptr->gate_exists);
            // TODO: may use the memory v_x_evals as long as the value vx_claim is saved
            y_helper.prepare(nb_input_vars, pad_ptr->v_evals, pad_ptr->hg_evals, poly_ptr->input_layer_vals.evals);
        }

        void prepare(
                const CircuitLayer<F, F_primitive>& poly,
                const F_primitive* rz1, const uint32_t & rz1_len,
                const F_primitive* rz2, const uint32_t & rz2_len,
                const F_primitive& alpha_,
                const F_primitive& beta_,
                GKRScratchPad<F, F_primitive>& scratch_pad){

            // Assign pointer
            nb_input_vars = poly.nb_input_vars;
            nb_output_vars = poly.nb_output_vars;
            alpha = alpha_;
            beta = beta_;
            poly_ptr = &poly;
            pad_ptr = &scratch_pad;

            // phase one
            _prepare_g_x_vals(rz1, rz1_len,
                              rz2, rz2_len,
                              alpha, beta,
                              poly.mul,poly.add,
                              poly.input_layer_vals,
                              pad_ptr->gate_exists);
            x_helper.prepare(nb_input_vars, pad_ptr->v_evals, pad_ptr->hg_evals, poly.input_layer_vals.evals);
        }

        void poly_evals_at(uint32_t var_idx, uint32_t degree, F* evals, TimingBreakdown& timer){
            if (var_idx < nb_input_vars){
                return x_helper.poly_eval_at(var_idx, degree, pad_ptr->gate_exists, evals, timer);
            }else{
                // When about the enter phase two, prepare the scratchpad
                return y_helper.poly_eval_at(var_idx - nb_input_vars, degree, pad_ptr->gate_exists, evals, timer);
            }
        }

        void receive_challenge(uint32_t var_idx, const F_primitive& r, TimingBreakdown& timer){
            if (var_idx < nb_input_vars){
                // Call x's sumcheck
                x_helper.receive_challenge(var_idx, r, pad_ptr->gate_exists, timer);
                assert(rx_len < MAX_RESULT_LEN);
                rx[rx_len] = r;
                rx_len += 1;
            }else{
                // Call y's sumcheck
                y_helper.receive_challenge(var_idx - nb_input_vars, r, pad_ptr->gate_exists, timer);
                assert(ry_len < MAX_RESULT_LEN);
                ry[ry_len] = r;
                ry_len += 1;
            }
        }

        F vx_claim(){
            return pad_ptr->v_evals[0];
        }

        F vy_claim(){
            return pad_ptr->v_evals[0];
        }
    };
} // namespace gkr
