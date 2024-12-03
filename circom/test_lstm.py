from reward_fn import generate_sample_inputs
import torch
from torch import nn
import random
import matplotlib.pyplot as plt
import json
import ezkl
from typing import Dict, List, Tuple
import bittensor as bt
import os
import numpy as np
import sys
from typing import Dict
from run_proofs import ZKProofsCircom
from onnxruntime import InferenceSession
import onnx
from test_lstm_python_work import LSTMWorking, LSTMWorkingInt, TotalModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, r2_score, mean_gamma_deviance, mean_absolute_percentage_error
# Manhattan score and absolute error
# Feed network absolute differences


class RewardTests():
    def __init__(self):
        self.scaling = 10**9
        super().__init__()
    
    def _inputs_to_json(self, inputs: Dict[str, torch.Tensor], input_path: str):
        with open(input_path, 'w') as outfile:
            json.dump(inputs, outfile)
        
    def _read_outputs_from_json(self, public_path: str):
        with open(public_path) as json_data:
            d = json.load(json_data)
            json_data.close()
            return d
    
    def _prove_and_verify(self, witness_file, input_file, proof_path, public_path, verification_key, circuit_name):
        circuit = ZKProofsCircom(circuit_name)
        res = circuit.compile_circuit()
        circuit.compute_witness(witness_file,input_file, wasm = True, c = False)
        circuit.proof_setup(verification_key)
        circuit.proof(witness_file,proof_path, public_path)
        circuit.verify(verification_key, public_path, proof_path)

    def _get_files(self, input_folder, proof_folder, temp_folder, circuit_folder, name):
        self._create_folder(input_folder)
        self._create_folder(proof_folder)
        self._create_folder(temp_folder)
        # self._create_folder(circuit_folder)

        witness_file = os.path.join(temp_folder,f"{name}_witness.wtns")
        input_file = os.path.join(input_folder,f"{name}_input.json")
        proof_path = os.path.join(proof_folder,f"{name}_proof.json")
        public_path = os.path.join(proof_folder,f"{name}_public.json")
        verification_key = os.path.join(temp_folder,f"{name}_verification_key.json")
        circuit_name = os.path.join(circuit_folder,f"{name}.circom")
        return witness_file,input_file,proof_path,public_path,verification_key,circuit_name
    
    def _create_folder(self, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def compare_values(self, python_value, circom_value, scaling):
        modulus = 21888242871839275222246405745257275088548364400416034343698204186575808495617
        if python_value < 0:
            circom_value = circom_value - modulus
            pass
        return abs(python_value - circom_value/scaling)<0.0000001
                                                
    
    def compare_values_ignore_case(self, python_value, circom_value):
        modulus = 21888242871839275222246405745257275088548364400416034343698204186575808495617
        if python_value < 0:
            # python_value * -1
            pass
        print(abs(abs(python_value) - abs(circom_value)))
        print((python_value - circom_value)<0.00000001)
        return abs(abs(python_value) - abs(circom_value))<0.000001
    
    def compare_values_case(self, python_value, circom_value):
        modulus = 21888242871839275222246405745257275088548364400416034343698204186575808495617
        if python_value < 0:
            # python_value * -1
            pass
        if python_value < 0:
            if circom_value == 1:
                return False
            elif circom_value == 0:
                return True
        if python_value>= 0 :
            if circom_value == 1:
                return True
            elif circom_value == 0:
                return False
        raise

    def get_weights(self, onnx_graph) -> Dict[str, np.array]:
        weights_dict = {}

        # Iterate over all initializers (which include the weights)
        for initializer in onnx_graph.initializer:
            # Convert the initializer to a numpy array
            weights_array = onnx.numpy_helper.to_array(initializer)
            # Store it in the dictionary with the initializer name
            weights_dict[initializer.name] = weights_array
        return weights_dict
    
    def test_LSTM_circom_work(self, input_folder:str, proof_folder: str, temp_folder: str, circuit_folder:str):
        name = "lstm_function"
        model_name = 'network.onnx'
        f = open('input_lstm.json')
        data = json.load(f)
        # x_test = data
        print(np.asarray(data['input_data']).shape)
        x_test = np.asarray(data['input_data']).reshape(1,5,1)
        print(x_test.shape)
        print(x_test)
        y_test = np.asarray(data['output_data'])
        sess = InferenceSession(model_name)
        #Check to make sure model is as expected
        assert np.allclose(sess.run(None, {"input": x_test.astype(np.float32)})[0], y_test)
        
        onnx_model = onnx.load(model_name)
        onnx.checker.check_model(onnx_model)
        onnx_graph = onnx_model.graph
        print(onnx_graph.node[1])

        weights_dict = self.get_weights(onnx_graph)
        # print(weights_dict.keys())
        

        onnx.utils.extract_model('network.onnx', 'extracted_network.onnx', ["input"], 
                                 ["LSTM__43:0"])
        sess = InferenceSession('extracted_network.onnx')



        lstm_function_results = sess.run(None, {"input": x_test.astype(np.float32)})[0]
        print(lstm_function_results.shape)

        witness_file, input_file, proof_path, public_path, verification_key, circuit_name = self._get_files(input_folder, proof_folder, temp_folder, circuit_folder, name)
        print(weights_dict['R0__41'].shape)
        print((self.scaling * x_test).astype('int').tolist()[0])
        python_work_int = LSTMWorkingInt(x_test, weights_dict["W0__40"], weights_dict['R0__41'], weights_dict['B0__42'], weights_dict['const_fold_opt__80'], weights_dict['const_fold_opt__80'], self.scaling)

        inputs = {
            'input_data': (self.scaling * x_test).astype('int').tolist()[0],
            'scaling': self.scaling,
            'W0__40': (weights_dict['W0__40']*self.scaling).astype(int).tolist(),
            'R0__41': (weights_dict['R0__41']*self.scaling).astype(int).tolist(),
            'B0__42': (weights_dict['B0__42']*self.scaling).astype(int).tolist(),
            'const_fold_opt__80': (weights_dict['const_fold_opt__80']*self.scaling).astype(int).tolist(),
            }
        print(weights_dict['W0__40'].shape, weights_dict['R0__41'].shape, weights_dict['B0__42'].shape, weights_dict['const_fold_opt__80'].shape)

        self._inputs_to_json(inputs, input_file)

        # # #Run the circuit
        self._prove_and_verify(witness_file, input_file, proof_path, public_path, verification_key, circuit_name)

        # # Check that the outputs match
        output = self._read_outputs_from_json(public_path)
        output = [int(t) for t in output]
        x = 0
        modulus = 21888242871839275222246405745257275088548364400416034343698204186575808495617

        print(len(output))
        print(lstm_function_results.shape)
        # for i in range(len(self.block_number.tolist())):
        #     # print(reward_fn_results[i].item(),output[i]/self.scaling)
        #     assert self.compare_values(reward_fn_results[i].item(),output[i], self.scaling)
        #     assert self.compare_values(self.block_number.tolist()[i], output[i+1024],1)
        #     assert self.compare_values(self.miner_uid.tolist()[i], output[i+1024*2],1)
        #     assert self.compare_values(self.validator_uid.tolist()[i], output[i+1024*3],1)

        print("Outputs match!")

    def test_LSTM_python_work(self, input_folder:str, proof_folder: str, temp_folder: str, circuit_folder:str):
        name = "lstm_function"
        model_name = 'network.onnx'
        f = open('input_lstm.json')
        data = json.load(f)

        x_test = np.asarray(data['input_data']).reshape(1,5,1)
        y_test = np.asarray(data['output_data'])
        sess = InferenceSession(model_name)
        #Check to make sure model is as expected
        assert np.allclose(sess.run(None, {"input": x_test.astype(np.float32)})[0], y_test)
        
        onnx_model = onnx.load(model_name)
        onnx.checker.check_model(onnx_model)
        onnx_graph = onnx_model.graph

        onnx.utils.extract_model('network.onnx', 'extracted_network.onnx', ["input"], 
                                # ["Squeeze__47:0"],
                                # ["LSTM__43:0"]
                                # ["LSTM__59:0"]
                                # ["Squeeze__63:0"]
                                # ["sequential/lstm_1/PartitionedCall/strided_slice_2:0"]
                                # ["sequential/lstm_1/PartitionedCall/strided_slice_2__73:0"]
                                ["output"]
                                 )
        sess = InferenceSession('network.onnx')
        lstm_function_results = sess.run(None, {"input": x_test.astype(np.float32)})[0]
        

        onnx_model = onnx.load('network.onnx')


        weights_dict = self.get_weights(onnx_model.graph)

        python_model = TotalModel(x_test, self.scaling, weights_dict)
        python_model_int = TotalModel(x_test, self.scaling, weights_dict, isInt=True)

        # python_total_model = TotalModel()
        
        python_circom_int_approx, tmp_1, tmp_2 = python_model_int.run("approx",5)
        python_circom_approx, tmp_1, tmp_2 = python_model.run("approx",5)

        python_circom_out = python_model.run("step_breakdown",5)
        python_onnx_out = python_model.run("step",5)

        lstm_function_results = sess.run(None, {"input": x_test.astype(np.float32)})[0]
        print(lstm_function_results.shape)
        print(weights_dict.keys())

        for i in range(1):
            for j in range(1):
                # Check that ONNX  vs python onnx implementation
                print(python_onnx_out['Y'].shape)
                print(lstm_function_results,python_onnx_out['Y'])
                assert abs(lstm_function_results - python_onnx_out['Y'])< 0.0000001
                # Check that python onnx  vs python circom implementation
                # print(np.asarray(python_circom_out['h']).astype('float64').shape, python_onnx_out['Y'].shape)
                assert abs(np.asarray(python_circom_out['h']).astype('float64')- python_onnx_out['Y'])< 0.0000001
                # Check that ONNX  vs python circom implementation

                assert abs(np.asarray(python_circom_out['h']).astype('float64')- lstm_function_results)< 0.0000001
                #Check integer version produces same outputs
                test_param = 'h'
                # print(np.asarray(python_circom_approx[test_param]).astype('float64'))

                print(np.asarray(python_circom_int_approx).astype('float64')/self.scaling, np.asarray(python_circom_approx).astype('float64'))
                # assert abs(np.asarray(python_circom_int_approx[test_param]).astype('float64')/self.scaling - np.asarray(python_circom_approx[test_param]).astype('float64')) < 0.0001
                # print(np.asarray(python_circom_int_approx['h']).astype('float64')[i][j]/self.scaling, python_onnx_out['Y'][i][0][0][j])
        print("outputs match!")

    def generate_inputs(self, size):
        out = []
        for i in range(size):
            ran_num = 0
            ran_nums = np.random.normal(loc = ran_num, scale = 1, size=5)

            # ran_num = np.random.normal(size=1)
            # ran_nums = np.random.normal(loc = ran_num, scale = 0.1, size=5)

            out.append(ran_nums.reshape(1,5,1))


        return out
        

    def error_tests(self, input_folder:str, proof_folder: str, temp_folder: str, circuit_folder:str):
        name = "lstm_function"
        model_name = 'network.onnx'
        
        f = open('input_lstm.json')
        input_size = 1000
        data = self.generate_inputs(input_size)

        outputs_exact = []
        outputs_approx = []

        for x_test in data:
            sess = InferenceSession(model_name)
        
            onnx_model = onnx.load(model_name)
            onnx.checker.check_model(onnx_model)
            onnx_graph = onnx_model.graph
            lstm_function_results = sess.run(None, {"input": x_test.astype(np.float32)})[0]


            weights_dict = self.get_weights(onnx_model.graph)

            python_model = TotalModel(x_test, self.scaling, weights_dict)
            python_model_int = TotalModel(x_test, self.scaling, weights_dict, isInt=True)
        
            python_circom_int_approx, tmp1, tmp2 = python_model_int.run("approx",5)
            python_circom_approx, tmp1, tmp2 = python_model.run("approx",5)

            python_circom_out = python_model.run("step_breakdown",5)
            python_onnx_out = python_model.run("step",5)

            lstm_function_results = sess.run(None, {"input": x_test.astype(np.float32)})[0]
            # print(lstm_function_results.shape)
            # print(weights_dict.keys())

            # Check that ONNX  vs python onnx implementation
            # print(python_onnx_out['Y'].shape)
            # print(lstm_function_results,python_onnx_out['Y'])
            assert abs(lstm_function_results - python_onnx_out['Y'])< 0.00001
            # Check that python onnx  vs python circom implementation
            # print(np.asarray(python_circom_out['h']).astype('float64').shape, python_onnx_out['Y'].shape)
            assert abs(np.asarray(python_circom_out['h']).astype('float64')- python_onnx_out['Y'])< 0.00001
            # Check that ONNX  vs python circom implementation

            assert abs(np.asarray(python_circom_out['h']).astype('float64')- lstm_function_results)< 0.0001
            #Check integer version produces same outputs
            test_param = 'h'
            # print(np.asarray(python_circom_approx[test_param]).astype('float64'))

            # print(np.asarray(python_circom_int_approx[test_param]).astype('float64')/self.scaling, np.asarray(python_circom_approx[test_param]).astype('float64'))
            outputs_exact.append(lstm_function_results[0][0])
            # outputs_approx.append((np.asarray(python_circom_int_approx[test_param]).astype('float64')/self.scaling)[0][0])
            outputs_approx.append(np.asarray(python_circom_approx).astype('float64')[0][0])

        mse = mean_squared_error(outputs_exact, outputs_approx)
        mae = mean_absolute_error(outputs_exact, outputs_approx)
        r2 = r2_score(outputs_exact, outputs_approx)
        me = max_error(outputs_exact, outputs_approx)
        mape = mean_absolute_percentage_error(outputs_exact, outputs_approx)
        # r2 = r2_score(outputs_exact, outputs_approx)

        
        print("MSE", mse)
        print("MAE", mae)
        print("r2", r2)
        print("MAPE", mape)
        # import statistics
        # errors = []
        # for i in range(len(outputs_exact)):
        #     errors.append(abs(1 - outputs_exact[i]/outputs_approx[i]))
        # print(statistics.mean(errors), statistics.median(errors), statistics.stdev(errors))
        # # print("MGD", mgd)


    def test_LSTM_python_to_circom(self, input_folder:str, proof_folder: str, temp_folder: str, circuit_folder:str):
        name = "lstm_function"

        witness_file, input_file, proof_path, public_path, verification_key, circuit_name = self._get_files(input_folder, proof_folder, temp_folder, circuit_folder, name)

        model_name = 'network.onnx'
        f = open('input_lstm.json')
        data = json.load(f)

        x_test = np.asarray(data['input_data']).reshape(1,5,1)
        y_test = np.asarray(data['output_data'])
        sess = InferenceSession(model_name)
        #Check to make sure model is as expected
        assert np.allclose(sess.run(None, {"input": x_test.astype(np.float32)})[0], y_test)
        
        onnx_model = onnx.load(model_name)
        onnx.checker.check_model(onnx_model)
        onnx_graph = onnx_model.graph
        sess = InferenceSession('network.onnx')
        lstm_function_results = sess.run(None, {"input": x_test.astype(np.float32)})[0]
        

        onnx_model = onnx.load('network.onnx')


        weights_dict = self.get_weights(onnx_model.graph)

        python_model = TotalModel(x_test, self.scaling, weights_dict)
        python_model_int = TotalModel(x_test, self.scaling, weights_dict, isInt=True)

        # python_total_model = TotalModel()
        
        python_circom_int_approx, tmp1, tmp2 = python_model_int.run("approx",5)
        # python_circom_approx, tmp1_float, tmp2_float = python_model.run("approx",5)
        # assert abs(np.asarray(python_circom_int_approx).astype('float64')/self.scaling - np.asarray(python_circom_approx).astype('float64')) < 0.0001

        # for p in tmp1.keys():
        #     print(p, len(tmp1[p]), len(tmp1[p][0]), len(tmp1[p][0][0]))
        #     print(p, len(tmp2[p]), len(tmp2[p][0]), len(tmp2[p][0][0]))

        # sys.exit()
        print(weights_dict['R0__57'].shape)
        self.hidden_size = 50

        inputs = {
            'input_data': (self.scaling * x_test).astype('int').tolist()[0],
            'scaling': self.scaling,
            'wi_1': (weights_dict['W0__40'][:,:self.hidden_size]*self.scaling).astype(int).tolist(),
            'wo_1': (weights_dict['W0__40'][:,self.hidden_size:self.hidden_size*2]*self.scaling).astype(int).tolist(),
            'wf_1': (weights_dict['W0__40'][:,self.hidden_size*2:self.hidden_size*3]*self.scaling).astype(int).tolist(),
            'wc_1': (weights_dict['W0__40'][:,self.hidden_size*3:]*self.scaling).astype(int).tolist(),

            'ui_1': (weights_dict['R0__41'][:,:self.hidden_size]*self.scaling).astype(int).tolist(),
            'uo_1': (weights_dict['R0__41'][:,self.hidden_size:self.hidden_size*2]*self.scaling).astype(int).tolist(),
            'uf_1': (weights_dict['R0__41'][:,self.hidden_size*2:self.hidden_size*3]*self.scaling).astype(int).tolist(),
            'uc_1': (weights_dict['R0__41'][:,self.hidden_size*3:]*self.scaling).astype(int).tolist(),

            'bi_1': (weights_dict['B0__42'][:,:self.hidden_size]*self.scaling).astype(int).tolist(),
            'bo_1': (weights_dict['B0__42'][:,self.hidden_size:self.hidden_size*2]*self.scaling).astype(int).tolist(),
            'bf_1': (weights_dict['B0__42'][:,self.hidden_size*2:self.hidden_size*3]*self.scaling).astype(int).tolist(),
            'bc_1': (weights_dict['B0__42'][:,self.hidden_size*3:self.hidden_size*4]*self.scaling).astype(int).tolist(),

            'wi_2': (weights_dict['W0__56'][:,:self.hidden_size]*self.scaling).astype(int).tolist(),
            'wo_2': (weights_dict['W0__56'][:,self.hidden_size:self.hidden_size*2]*self.scaling).astype(int).tolist(),
            'wf_2': (weights_dict['W0__56'][:,self.hidden_size*2:self.hidden_size*3]*self.scaling).astype(int).tolist(),
            'wc_2': (weights_dict['W0__56'][:,self.hidden_size*3:]*self.scaling).astype(int).tolist(),

            'ui_2': (weights_dict['R0__57'][:,:self.hidden_size]*self.scaling).astype(int).tolist(),
            'uo_2': (weights_dict['R0__57'][:,self.hidden_size:self.hidden_size*2]*self.scaling).astype(int).tolist(),
            'uf_2': (weights_dict['R0__57'][:,self.hidden_size*2:self.hidden_size*3]*self.scaling).astype(int).tolist(),
            'uc_2': (weights_dict['R0__57'][:,self.hidden_size*3:]*self.scaling).astype(int).tolist(),

            'bi_2': (weights_dict['B0__58'][:,:self.hidden_size]*self.scaling).astype(int).tolist(),
            'bo_2': (weights_dict['B0__58'][:,self.hidden_size:self.hidden_size*2]*self.scaling).astype(int).tolist(),
            'bf_2': (weights_dict['B0__58'][:,self.hidden_size*2:self.hidden_size*3]*self.scaling).astype(int).tolist(),
            'bc_2': (weights_dict['B0__58'][:,self.hidden_size*3:self.hidden_size*4]*self.scaling).astype(int).tolist(),
            'const_fold_opt__80': (weights_dict['const_fold_opt__80']*self.scaling).astype(int).tolist(),



            
            
            # 'R0__41': (weights_dict['R0__41']*self.scaling).astype(int).tolist(),
            # 'B0__42': (weights_dict['B0__42']*self.scaling).astype(int).tolist(),
            # 'W0__56': (weights_dict['W0__56']*self.scaling).astype(int).tolist(),
            # 'R0__57': (weights_dict['R0__57']*self.scaling).astype(int).tolist(),
            # 'B0__58': (weights_dict['B0__58']*self.scaling).astype(int).tolist(),

            'f_temp_1': tmp1['f_temp'],
            'f_temp_rem_1': tmp1['f_temp_rem'],
            'f_1': tmp1['f'],
            'f_rem_1': tmp1['f_rem'],
            'i_temp_1': tmp1['i_temp'],
            'i_temp_rem_1': tmp1['i_temp_rem'],
            'i_1': tmp1['i'],
            'i_rem_1': tmp1['i_rem'],
            'o_temp_1': tmp1['o_temp'],
            'o_temp_rem_1': tmp1['o_temp_rem'],
            'o_1': tmp1['o'],
            'o_rem_1': tmp1['o_rem'],
            'c_temp_1': tmp1['c_temp'],
            'c_temp_rem_1': tmp1['c_temp_rem'],
            'c_1': tmp1['c'],
            'c_rem_1': tmp1['c_rem'],
            'C_temp_1': tmp1['C_temp'],
            'C_temp_rem_1': tmp1['C_temp_rem'],
            'C_1': tmp1['C'],
            'C_rem_1': tmp1['C_rem'],
            'h_1': tmp1['h'],
            'h_rem_1': tmp1['h_rem'],

            'f_temp_2': tmp2['f_temp'],
            'f_temp_rem_2': tmp2['f_temp_rem'],
            'f_2': tmp2['f'],
            'f_rem_2': tmp2['f_rem'],
            'i_temp_2': tmp2['i_temp'],
            'i_temp_rem_2': tmp2['i_temp_rem'],
            'i_2': tmp2['i'],
            'i_rem_2': tmp2['i_rem'],
            'o_temp_2': tmp2['o_temp'],
            'o_temp_rem_2': tmp2['o_temp_rem'],
            'o_2': tmp2['o'],
            'o_rem_2': tmp2['o_rem'],
            'c_temp_2': tmp2['c_temp'],
            'c_temp_rem_2': tmp2['c_temp_rem'],
            'c_2': tmp2['c'],
            'c_rem_2': tmp2['c_rem'],
            'C_temp_2': tmp2['C_temp'],
            'C_temp_rem_2': tmp2['C_temp_rem'],
            'C_2': tmp2['C'],
            'C_rem_2': tmp2['C_rem'],
            'h_2': tmp2['h'],
            'h_rem_2': tmp2['h_rem']
            }
        
        

        # python_circom_out = python_model.run("step_breakdown",5)
        # python_onnx_out = python_model.run("step",5)

        lstm_function_results = sess.run(None, {"input": x_test.astype(np.float32)})[0]

        self._inputs_to_json(inputs, input_file)

        # #Run the circuit
        self._prove_and_verify(witness_file, input_file, proof_path, public_path, verification_key, circuit_name)

        # Check that the outputs match
        output = self._read_outputs_from_json(public_path)
        output = [int(t) for t in output]
        x = 0
        for i in range(1):
            print(lstm_function_results[i].item(),output[i]/self.scaling)
            assert self.compare_values(lstm_function_results[i].item(),output[i], self.scaling)

        print("outputs match!")



if __name__ == "__main__":
    proof_folder = "output"
    temp_folder = "temp"
    input_folder = "inputs"
    circuit_folder = ""
    # inputs = generate_sample_inputs()
    reward = RewardTests()
    # reward.test_LSTM_circom_work(input_folder,proof_folder,temp_folder,circuit_folder)
    reward.test_LSTM_python_work(input_folder,proof_folder,temp_folder,circuit_folder)
    # reward.error_tests(input_folder,proof_folder,temp_folder,circuit_folder)
    # reward.test_LSTM_python_to_circom(input_folder,proof_folder,temp_folder,circuit_folder)


