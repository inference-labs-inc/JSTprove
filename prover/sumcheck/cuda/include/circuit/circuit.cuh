#pragma once

#define MAX_NUM_LAYERS 10

namespace gkr{

    // Evaluate the MLE
    template<typename F, typename F_primitive>
    F eval_multilinear(const F* evals, const uint32_t& evals_len, const F_primitive* x, const uint32_t& x_len){
        assert((1UL << x_len) == evals_len);
        F* scratch = (F*) malloc(evals_len * sizeof(F));
        for(int i = 0; i < evals_len; i++){
            scratch[i] = evals[i];
        }
        uint32_t cur_eval_size = evals_len >> 1;
        for (int x_idx = 0; x_idx < x_len; x_idx++){
            F_primitive r = x[x_idx];
            for (uint32_t i = 0; i < cur_eval_size; i++){
                scratch[i] = scratch[(i << 1)] + (scratch[(i << 1) + 1] - scratch[(i << 1)]) * r;
            }
            cur_eval_size >>= 1;
        }
        F result = scratch[0];
        free(scratch);
        return result;
    }

    // Class of Multi-linear evaluation
    template<typename F>
    class MultiLinearPoly{
    public:
        uint32_t nb_vars = 0;
        F* evals = nullptr;
        uint32_t evals_len = 0;

        static MultiLinearPoly random(uint32_t nb_vars) {
            MultiLinearPoly poly;

            poly.nb_vars = nb_vars;
            uint32_t evals_len = 1 << nb_vars;
            poly.evals = (F*)malloc(evals_len * sizeof(F));
            poly.evals_len = evals_len;
            for (uint32_t i = 0; i < evals_len; i++){
                poly.evals[i] = F::random();
            }

            return poly;
        }
    };

    // One single gate
    template<typename F, uint32_t nb_input>
    class Gate{
    public:
        uint32_t     i_ids[nb_input];
        uint32_t     o_id;
        alignas(8) F coef;

        Gate(){}
        Gate(uint32_t o_id, uint32_t i_ids[nb_input], F coef) {
            this->o_id = o_id;
            for (uint32_t i = 0; i < nb_input; i++){
                this->i_ids[i] = i_ids[i];
            }
            this->coef = coef;
        }
    };

    // The sparse connection
    template<typename F, uint32_t nb_input>
    class SparseCircuitConnection{
    public:
        uint32_t nb_output_vars = 0;
        uint32_t nb_input_vars = 0;
        Gate<F, nb_input>* sparse_evals = nullptr;
        uint32_t sparse_evals_len = 0;

        static SparseCircuitConnection random(uint32_t nb_output_vars, uint32_t nb_input_vars){
            SparseCircuitConnection poly;
            poly.nb_input_vars = nb_input_vars;
            poly.nb_output_vars = nb_output_vars;
            uint32_t output_size = 1 << nb_output_vars;
            uint32_t input_size = 1 << nb_input_vars;
            poly.sparse_evals = (Gate<F, nb_input>*) malloc(output_size * sizeof(Gate<F, nb_input>));
            poly.sparse_evals_len = output_size;

            for (uint32_t i = 0; i < output_size; i++){
                // to make sure all o_gates are used
                uint32_t o_gate = i;
                uint32_t i_gates[nb_input];
                uint32_t i_gate = i;
                for (uint32_t j = 0; j < nb_input; j++){
                    i_gates[j] = i_gate % input_size;
                    i_gate = i_gate + output_size;
                }
                poly.sparse_evals[i] = Gate<F, nb_input> (o_gate, i_gates, F::one());
            }
            return poly;
        }
    };

    // One Layer of GKR circuit
    template<typename F, typename F_primitive>
    class CircuitLayer{
    public:
        uint32_t nb_output_vars;
        uint32_t nb_input_vars;
        MultiLinearPoly<F> input_layer_vals;
        MultiLinearPoly<F> output_layer_vals;

        SparseCircuitConnection<F_primitive, 1> add;
        SparseCircuitConnection<F_primitive, 2> mul;

        static CircuitLayer random(uint32_t nb_output_vars, uint32_t nb_input_vars){
            CircuitLayer poly;
            poly.nb_output_vars = nb_output_vars;
            poly.nb_input_vars = nb_input_vars;
            poly.input_layer_vals = MultiLinearPoly<F>::random(nb_input_vars);

            poly.mul = SparseCircuitConnection<F_primitive, 2>::random(nb_output_vars, nb_input_vars);
            poly.add = SparseCircuitConnection<F_primitive, 1>::random(nb_output_vars, nb_input_vars);
            return poly;
        }

        void evaluate(F* output, uint32_t output_len) const {

            for (int i = 0; i < mul.sparse_evals_len; i++){
                Gate<F_primitive, 2> gate = mul.sparse_evals[i];
                output[gate.o_id] +=
                        input_layer_vals.evals[gate.i_ids[0]] *
                        input_layer_vals.evals[gate.i_ids[1]] *
                        gate.coef;
            }

            for (int i = 0; i < add.sparse_evals_len; i++){
                Gate<F_primitive, 1> gate = add.sparse_evals[i];
                output[gate.o_id] +=
                        input_layer_vals.evals[gate.i_ids[0]] *
                        gate.coef;
            }
        }
    };

    // GKR Layered Circuit
    template<typename F, typename F_primitive>
    class Circuit{
    public:
        CircuitLayer<F, F_primitive> layers[MAX_NUM_LAYERS];
        uint32_t layers_len = 0;
        void add_layer(const CircuitLayer<F, F_primitive>& layer){
            assert(layers_len < MAX_NUM_LAYERS);
            layers[layers_len] = layer;
            layers_len = layers_len + 1;
        }
        void evaluate(){
            for (uint32_t i = 0; i < layers_len - 1; ++i){
                layers[i + 1].input_layer_vals.evals = layers[i].evaluate();
            }
            layers[layers_len-1].output_layer_vals.evals = layers[layers_len-1].evaluate();
        }
    };

} // namespace gkr
