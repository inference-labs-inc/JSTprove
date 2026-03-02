#pragma once

#include <vector>  // Only Verifier needs vector

#include "circuit/circuit.cuh"
#include "sumcheck_common.cuh"

namespace gkr{
    template<typename F, typename F_primitive>
    F degree_2_eval(const std::vector<F>& vals, const F_primitive& x){
        const F& c0 = vals[0];
        F c2 = F::INV_2 * (vals[2] - vals[1] * 2 + vals[0]);
        F c1 = vals[1] - vals[0] - c2;

        return c0 + (c2 * x + c1) * x;
    }


    template<typename F, typename F_primitive, uint32_t nb_input>
    F_primitive eval_sparse_circuit_connect_poly(
        const SparseCircuitConnection<F_primitive, nb_input>& poly,
        const F_primitive* rz1, const uint32_t & rz1_len,
        const F_primitive* rz2, const uint32_t & rz2_len,
        const F_primitive& alpha,
        const F_primitive& beta,
        const std::vector<std::vector<F_primitive>>& ris
        ){

        std::vector<F_primitive> eq_evals_at_rz1(1 << rz1_len);
        std::vector<F_primitive> eq_evals_at_rz2(1 << rz2_len);

        _eq_evals_at_primitive(rz1, rz1_len, alpha, eq_evals_at_rz1.data());
        _eq_evals_at_primitive(rz2, rz2_len, beta, eq_evals_at_rz2.data());

        std::vector<std::vector<F_primitive>> eq_evals_at_ris(nb_input);

        for (uint32_t i = 0; i < nb_input; i++){
            eq_evals_at_ris[i].resize(1 << (ris[i].size()));
            _eq_evals_at_primitive(ris[i].data(), ris[i].size(), F_primitive::one(), eq_evals_at_ris[i].data());
        }

        F_primitive v = F_primitive::zero();
        for (int g_id = 0; g_id < poly.sparse_evals_len; g_id++){
            Gate<F_primitive, nb_input> gate = poly.sparse_evals[g_id];
            auto prod = (eq_evals_at_rz1[gate.o_id] + eq_evals_at_rz2[gate.o_id]);
            for (uint32_t i = 0; i < nb_input; i++){
                prod *= eq_evals_at_ris[i][gate.i_ids[i]];
            }
            v += prod * gate.coef;
        }

        return v;
    }
}
