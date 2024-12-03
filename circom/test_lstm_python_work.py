from onnx.onnx_ml_pb2 import ModelProto
from typing import Dict, Tuple
import numpy as np
from numpy import tanh
import onnx
from math import log10, exp
import sys
from test_lstm_approximations import (zanh, zanh_int, tanh, sigmoid, zigmoid, zigmoid_int,
                                       piecewise_linear_sigmoid, piecewise_quadratic_sigmoid,
                                         minimax_sigmoid, bandish_sigmoid,rational_sigmoid, 
                                         algebraic_sigmoid, taylor_tanh, taylor_sigmoid,
                                         taylor_sigmoid_int, taylor_tanh_int)


def Gemm(weights_dict, A, scaling = 1):
    B = weights_dict['sequential/dense/MatMul/ReadVariableOp:0']
    C = weights_dict['sequential/dense/BiasAdd/ReadVariableOp:0']
    print("ASHAPE",A.shape)
    print("ASHAPE",B.shape)
    print("ASHAPE",C.shape)
    
    return  np.matmul(A,B) + C * scaling

def get_weights(onnx_graph) -> Dict[str, np.array]:
    weights_dict = {}

    # Iterate over all initializers (which include the weights)
    for initializer in onnx_graph.initializer:
        # Convert the initializer to a numpy array
        weights_array = onnx.numpy_helper.to_array(initializer)
        # Store it in the dictionary with the initializer name
        weights_dict[initializer.name] = weights_array
    return weights_dict


class LSTMWorking():
    def __init__(self, X, W, R, bh, wy, by, scaling, t = 1) -> None:
        n = 0
        # print(n)
        self.n = 0
        self.W = (W * float(10**n))#.astype(int)
        self.U = (R*float(10**n))#.astype(int)
        self.bh = (bh*float(10**(2*n)))#.astype(int)
        self.wy = R
        self.by = by[0][0]
        self.hidden_size = 50
        # print(self.hidden_size)
        # print('W = ', self.W.shape, ' U = ', self.U.shape, ' bh = ', self.bh.shape, ' wy =', self.wy.shape, 'by = ', self.by.shape)
        if t == 1:
            self.wi = self.W[:,:self.hidden_size].reshape(1,self.hidden_size)
            self.wo = self.W[:,self.hidden_size:self.hidden_size*2].reshape(1,self.hidden_size)
            self.wf = self.W[:,self.hidden_size*2:self.hidden_size*3].reshape(1,self.hidden_size)
            self.wc = self.W[:,self.hidden_size*3:].reshape(1,self.hidden_size)
        elif t == 2:
            self.wi = self.W[:,:self.hidden_size].reshape(self.hidden_size, self.hidden_size)
            self.wo = self.W[:,self.hidden_size:self.hidden_size*2].reshape(self.hidden_size,self.hidden_size)
            self.wf = self.W[:,self.hidden_size*2:self.hidden_size*3].reshape(self.hidden_size,self.hidden_size)
            self.wc = self.W[:,self.hidden_size*3:].reshape(self.hidden_size,self.hidden_size)
        # print(self.wi,self.wf,self.wc, self.wo)
        


        self.ui = np.transpose(self.U[:,:self.hidden_size]).reshape(1,self.hidden_size,self.hidden_size)
        self.uo = np.transpose(self.U[:,self.hidden_size:self.hidden_size*2]).reshape(1,self.hidden_size,self.hidden_size)
        self.uf = np.transpose(self.U[:,self.hidden_size*2:self.hidden_size*3]).reshape(1,self.hidden_size,self.hidden_size)
        self.uc = np.transpose(self.U[:,self.hidden_size*3:]).reshape(1,self.hidden_size,self.hidden_size)
        # print(self.ui,self.uf,self.uc, self.uo)


        self.bi = self.bh[:,:self.hidden_size].reshape(1,self.hidden_size)
        self.bo = self.bh[:,self.hidden_size:self.hidden_size*2].reshape(1,self.hidden_size)
        self.bf = self.bh[:,self.hidden_size*2:self.hidden_size*3].reshape(1,self.hidden_size)
        self.bc = self.bh[:,self.hidden_size*3:self.hidden_size*4].reshape(1,self.hidden_size)
        # print(self.bi,self.bf,self.bc, self.bo)

        # self.X = (X*float(10**n)).round().astype(int)[0]
        self.X = (X*float(10**n))[0]

        # print('X = ', X)

    def run_lstm_approx(self, t, fun1 = zigmoid, fun2 = zanh):
                # h = [np.zeros(self.hidden_size).astype(int)]
        # c = [np.zeros(self.hidden_size).astype(int)]
        h = [np.zeros(self.hidden_size)]
        c = [np.zeros(self.hidden_size)]
        f_out = []
        f_remainder = []
        f_zigmoid_out = []
        f_zigmoid_remainder = []
        i_out = []
        i_remainder = []
        i_zigmoid_out = []
        i_zigmoid_remainder = []
        o_out = []
        o_remainder = []
        o_zigmoid_out = []
        o_zigmoid_remainder = []
        candidate_out = []
        candidate_remainder = []
        candidate_zanh_out = []
        candidate_zanh_remainder = []
        c_out = []
        c_remainder = []
        c_zanh_out = []
        c_zanh_remainder = []
        h_out = []
        h_remainder = []
        H_t = np.zeros((1, 50), dtype=np.float32)
        ui,uo,uf,uc = np.split(np.transpose(self.U[0]), 4, -1)
        wi,wo,wf,wc = np.split(np.transpose(self.W[0]), 4, -1)
        for j in range(t):
            x = np.expand_dims(self.X[j],0)
            f = np.dot(x, wf) + np.dot(H_t,uf) + self.bf
            of = f 
            rf = f % 10**self.n
            f_out.append(of.astype(int).flatten().tolist())
            f_remainder.append(rf.astype(int).flatten().tolist())
            fz = [fun1(fo) for fo in of.flatten().tolist()]
            

            ofz = [zf for zf in fz]
            rfz = [zf % 10**(3*self.n) for zf in fz]
            ofz = np.array(ofz)
            rfz = np.array(rfz)
            f_zigmoid_out.append(ofz.astype(int).flatten().tolist())
            f_zigmoid_remainder.append(rfz.astype(int).flatten().tolist())

            i = np.dot(self.X[j], wi) + np.dot(H_t,ui) + self.bi
            oi = i 
            ri = i % 10**self.n
            i_out.append(oi.astype(int).flatten().tolist())
            i_remainder.append(ri.astype(int).flatten().tolist())

            iz = [fun1(io) for io in oi.flatten().tolist()]
            oiz = [zi for zi in iz]
            riz = [zi % 10**(3*self.n) for zi in iz]
            oiz = np.array(oiz)
            riz = np.array(riz)
            i_zigmoid_out.append(oiz.astype(int).flatten().tolist())
            i_zigmoid_remainder.append(riz.astype(int).flatten().tolist())   

            o = np.dot(self.X[j], wo) + np.dot(H_t,uo) + self.bo
            oo = o 
            ro = o % 10**self.n
            o_out.append(oo.astype(int).flatten().tolist())
            o_remainder.append(ro.astype(int).flatten().tolist())

            oz = [fun1(ooo) for ooo in oo.flatten().tolist()]
            ooz = [zo for zo in oz]
            roz = [zo % 10**(3*self.n) for zo in oz]
            ooz = np.array(ooz)
            roz = np.array(roz)
            o_zigmoid_out.append(ooz.astype(int).flatten().tolist())
            o_zigmoid_remainder.append(roz.astype(int).flatten().tolist())


            candidate = np.dot(self.X[j], wc) + np.dot(H_t,uc) + self.bc
            oca = candidate 
            rca = candidate % 10**self.n
            candidate_out.append(oca.astype(int).flatten().tolist())
            candidate_remainder.append(rca.astype(int).flatten().tolist())

            caz = [fun2(co) for co in oca.astype(float).flatten().tolist()]
            ocaz = [ca  for ca in caz]
            rcaz = [ca % 10**(3*self.n) for ca in caz]
            ocaz = np.array(ocaz)
            rcaz = np.array(rcaz)
            candidate_zanh_out.append(ocaz.astype(int).flatten().tolist())
            candidate_zanh_remainder.append(rcaz.astype(int).flatten().tolist())

            C = np.multiply(ofz, c[-1]) + np.multiply(oiz, ocaz)
            oc = C 
            rc = C % 10**self.n

            c.append(oc)
            c_out.append(oc.astype(int).flatten().tolist())
            c_remainder.append(rc.astype(int).flatten().tolist())

            cz = [fun2(co) for co in oc.astype(float).flatten().tolist()]
            ocz = [cz  for cz in cz]
            rcz = [cz % 10**(3*self.n) for cz in cz]
            ocz = np.array(ocz)
            rcz = np.array(rcz)
            c_zanh_out.append(ocz.astype(int).flatten().tolist())
            c_zanh_remainder.append(rcz.astype(int).flatten().tolist())


            H = np.multiply(ooz, ocz)
            H_t = np.expand_dims(H, 0)
            oh = H 
            rh = H % 10**self.n

            h.append(oh)
            h_out.append(oh.astype(int).flatten().tolist())
            h_remainder.append(rh.astype(int).flatten().tolist())

        out = (f_out, f_remainder, f_zigmoid_out, f_zigmoid_remainder, i_out, i_remainder, i_zigmoid_out, i_zigmoid_remainder, o_out, o_remainder, o_zigmoid_out, o_zigmoid_remainder, candidate_out, candidate_remainder, candidate_zanh_out, candidate_zanh_remainder, c_out, c_remainder, c_zanh_out, c_zanh_remainder, h_out, h_remainder)
        python_out_names = ('f_temp', 'f_temp_rem', 'f', 'f_rem','i_temp', 'i_temp_rem', 'i', 'i_rem','o_temp', 'o_temp_rem', 'o', 'o_rem', 'c_temp', 'c_temp_rem', 'c', 'c_rem','C_temp', 'C_temp_rem', 'C', 'C_rem', 'h', 'h_rem')
        output = dict(zip(python_out_names, out))
        return output

    def run_lstm(self, t, fun1 = sigmoid, fun2 = tanh):
        h = [np.zeros(self.hidden_size)]
        c = [np.zeros(self.hidden_size)]
        f_out = []
        f_remainder = []
        f_zigmoid_out = []
        f_zigmoid_remainder = []
        i_out = []
        i_remainder = []
        i_zigmoid_out = []
        i_zigmoid_remainder = []
        o_out = []
        o_remainder = []
        o_zigmoid_out = []
        o_zigmoid_remainder = []
        candidate_out = []
        candidate_remainder = []
        candidate_zanh_out = []
        candidate_zanh_remainder = []
        c_out = []
        c_remainder = []
        c_zanh_out = []
        c_zanh_remainder = []
        h_out = []
        h_remainder = []
        H_t = np.zeros((1, 50), dtype=np.float32)
        ui,uo,uf,uc = np.split(np.transpose(self.U[0]), 4, -1)
        wi,wo,wf,wc = np.split(np.transpose(self.W[0]), 4, -1)
        for j in range(t):
            x = np.expand_dims(self.X[j],0)
            
            f = np.dot(x, wf) + np.dot(H_t,uf) + self.bf
            of = f 
            rf = f % 10**self.n
            f_out.append(of.astype(str).flatten().tolist())
            f_remainder.append(rf.astype(str).flatten().tolist())
            fz = [fun1(fo) for fo in of.flatten().tolist()]
            # print("RUNLSTM",uf)
            

            ofz = [zf for zf in fz]
            rfz = [zf % 10**(3*self.n) for zf in fz]
            ofz = np.array(ofz)
            rfz = np.array(rfz)
            f_zigmoid_out.append(ofz.astype(str).flatten().tolist())
            f_zigmoid_remainder.append(rfz.astype(str).flatten().tolist())

            i = np.dot(self.X[j], wi) + np.dot(H_t,ui) + self.bi
            oi = i 
            ri = i % 10**self.n
            i_out.append(oi.astype(str).flatten().tolist())
            i_remainder.append(ri.astype(str).flatten().tolist())

            iz = [fun1(io) for io in oi.flatten().tolist()]
            oiz = [zi for zi in iz]
            riz = [zi % 10**(3*self.n) for zi in iz]
            oiz = np.array(oiz)
            riz = np.array(riz)
            i_zigmoid_out.append(oiz.astype(str).flatten().tolist())
            i_zigmoid_remainder.append(riz.astype(str).flatten().tolist())   

            o = np.dot(self.X[j], wo) + np.dot(H_t,uo) + self.bo
            oo = o 
            ro = o % 10**self.n
            o_out.append(oo.astype(str).flatten().tolist())
            o_remainder.append(ro.astype(str).flatten().tolist())

            oz = [fun1(ooo) for ooo in oo.flatten().tolist()]
            ooz = [zo for zo in oz]
            roz = [zo % 10**(3*self.n) for zo in oz]
            ooz = np.array(ooz)
            roz = np.array(roz)
            o_zigmoid_out.append(ooz.astype(str).flatten().tolist())
            o_zigmoid_remainder.append(roz.astype(str).flatten().tolist())


            candidate = np.dot(self.X[j], wc) + np.dot(H_t,uc) + self.bc
            oca = candidate 
            rca = candidate % 10**self.n
            candidate_out.append(oca.astype(str).flatten().tolist())
            candidate_remainder.append(rca.astype(str).flatten().tolist())

            caz = [fun2(co) for co in oca.astype(float).flatten().tolist()]
            ocaz = [ca  for ca in caz]
            rcaz = [ca % 10**(3*self.n) for ca in caz]
            ocaz = np.array(ocaz)
            rcaz = np.array(rcaz)
            candidate_zanh_out.append(ocaz.astype(str).flatten().tolist())
            candidate_zanh_remainder.append(rcaz.astype(str).flatten().tolist())

            C = np.multiply(ofz, c[-1]) + np.multiply(oiz, ocaz)
            oc = C 
            rc = C % 10**self.n

            c.append(oc)
            c_out.append(oc.astype(str).flatten().tolist())
            c_remainder.append(rc.astype(str).flatten().tolist())

            cz = [fun2(co) for co in oc.astype(float).flatten().tolist()]
            ocz = [cz  for cz in cz]
            rcz = [cz % 10**(3*self.n) for cz in cz]
            ocz = np.array(ocz)
            rcz = np.array(rcz)
            c_zanh_out.append(ocz.astype(str).flatten().tolist())
            c_zanh_remainder.append(rcz.astype(str).flatten().tolist())


            H = np.multiply(ooz, ocz)
            H_t = np.expand_dims(H, 0)
            oh = H 
            rh = H % 10**self.n

            h.append(oh)
            h_out.append(oh.astype(str).flatten().tolist())
            h_remainder.append(rh.astype(str).flatten().tolist())

        out = (f_out, f_remainder, f_zigmoid_out, f_zigmoid_remainder, i_out, i_remainder, i_zigmoid_out, i_zigmoid_remainder, o_out, o_remainder, o_zigmoid_out, o_zigmoid_remainder, candidate_out, candidate_remainder, candidate_zanh_out, candidate_zanh_remainder, c_out, c_remainder, c_zanh_out, c_zanh_remainder, h_out, h_remainder)
        python_out_names = ('f_temp', 'f_temp_rem', 'f', 'f_rem','i_temp', 'i_temp_rem', 'i', 'i_rem','o_temp', 'o_temp_rem', 'o', 'o_rem', 'c_temp', 'c_temp_rem', 'c', 'c_rem','C_temp', 'C_temp_rem', 'C', 'C_rem', 'h', 'h_rem')
        output = dict(zip(python_out_names, out))
        return output


    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        """This step function comes directly from onnx LSTM function, in order to try to match the outputs.
            The outputs of this function matches the output from onnxruntime

        Returns:
            Tuple[np.ndarray, np.ndarray]: output
        """        
        seq_length = 5
        hidden_size = 50
        batch_size = 1

        Y = np.empty([seq_length, 1, batch_size, hidden_size])
        h_list = []
        self.f = sigmoid
        self.g = tanh
        self.h = tanh

        [p_i, p_o, p_f] = [0,0,0]
        H_t = np.zeros((batch_size, hidden_size), dtype=np.float32)
        C_t = np.zeros((batch_size, hidden_size), dtype=np.float32)
        x_ = self.X

        self.num_directions = 1
        self. LAYOUT = 0
        index = 0
        print(x_)
        for x in np.split(x_, x_.shape[0], axis=0):
            gates = (
                np.dot(x, np.transpose(self.W[0]))
                + np.dot(H_t, np.transpose(self.U[0]))
                + np.add(*np.split(self.bh[0], 2))
            )
            # print(H_t.shape)
            # print(x.shape)
            # print(gates.shape)
            # print(np.dot(x, np.transpose(self.W[0]))[0][0:20])
            # print(np.add(*np.split(self.bh[0], 2))[0:20])
            # print(gates[0][0:20])

            # # print(gates[0][0:20])
            # sys.exit()
            
            i, o, f, c = np.split(gates, 4, -1)
            

            i = self.f(i + p_i * C_t)
            


            f = self.f(f + p_f * C_t)
            
            c = self.g(c)
            
            C = f * C_t + i * c
            
            o = self.f(o + p_o * C)
            
            C_tanh = self.h(C)
            
            H = o * C_tanh
            print(index, H.shape)
            # print(c)
            # print(gates[0][0:20])

            # print(gates[0][0:20])
            
            h_list.append(H)
            H_t = H
            C_t = C
            # if index >= 10:
            #     break
            index = index + 1
        concatenated = np.concatenate(h_list)
        if self.num_directions == 1:
            Y[:, 0, :, :] = concatenated.reshape(seq_length,1,hidden_size)

        if self.LAYOUT == 0:
            Y_h = Y[-1]
        else:
            Y = np.transpose(Y, [2, 0, 1, 3])
            Y_h = Y[:, :, -1, :]
        python_out_names = ('Y', 'Y_h', 'i', 'f', 'c', 'o', 'C', 'C_tanh', 'H', 'H_t', 'C_t', 'h_list')
        out = (Y, Y_h, i, f, c, o, C, C_tanh, H, H_t, C_t, h_list)
        output = dict(zip(python_out_names, out))

        return output

class LSTMWorkingInt():
    def __init__(self, X, W, R, bh, wy, by, scaling, t = 1) -> None:
        n = log10(scaling)

        self.n = n
        self.W = (W * float(10**n))#.astype(int)
        self.U = (R*float(10**n))#.astype(int)
        self.bh = (bh*float(10**(2*n)))#.astype(int)
        self.wy = R
        self.by = by[0][0]
        self.hidden_size = 50

        if t == 1:
            self.wi = self.W[:,:self.hidden_size].reshape(1,self.hidden_size)
            self.wo = self.W[:,self.hidden_size:self.hidden_size*2].reshape(1,self.hidden_size)
            self.wf = self.W[:,self.hidden_size*2:self.hidden_size*3].reshape(1,self.hidden_size)
            self.wc = self.W[:,self.hidden_size*3:].reshape(1,self.hidden_size)
        elif t == 2:
            self.wi = self.W[:,:self.hidden_size].reshape(1,self.hidden_size,self.hidden_size)
            self.wo = self.W[:,self.hidden_size:self.hidden_size*2].reshape(1,self.hidden_size,self.hidden_size)
            self.wf = self.W[:,self.hidden_size*2:self.hidden_size*3].reshape(1,self.hidden_size,self.hidden_size)
            self.wc = self.W[:,self.hidden_size*3:].reshape(1,self.hidden_size,self.hidden_size)


        self.ui = np.transpose(self.U[:,:self.hidden_size]).reshape(1,self.hidden_size,self.hidden_size)
        self.uo = np.transpose(self.U[:,self.hidden_size:self.hidden_size*2]).reshape(1,self.hidden_size,self.hidden_size)
        self.uf = np.transpose(self.U[:,self.hidden_size*2:self.hidden_size*3]).reshape(1,self.hidden_size,self.hidden_size)
        self.uc = np.transpose(self.U[:,self.hidden_size*3:]).reshape(1,self.hidden_size,self.hidden_size)


        self.bi = self.bh[:,:self.hidden_size].reshape(1,self.hidden_size)
        self.bo = self.bh[:,self.hidden_size:self.hidden_size*2].reshape(1,self.hidden_size)
        self.bf = self.bh[:,self.hidden_size*2:self.hidden_size*3].reshape(1,self.hidden_size)
        self.bc = self.bh[:,self.hidden_size*3:self.hidden_size*4].reshape(1,self.hidden_size)


        self.X = (X*float(10**n))[0]
        # print('X = ', X)

    def run_lstm_approx_int(self, t, fun1 = zigmoid_int, fun2 = zanh_int, polynomial_degree = 5):
        h = [np.zeros(self.hidden_size)]
        c = [np.zeros(self.hidden_size)]
        f_out = []
        f_remainder = []
        f_zigmoid_out = []
        f_zigmoid_remainder = []
        i_out = []
        i_remainder = []
        i_zigmoid_out = []
        i_zigmoid_remainder = []
        o_out = []
        o_remainder = []
        o_zigmoid_out = []
        o_zigmoid_remainder = []
        candidate_out = []
        candidate_remainder = []
        candidate_zanh_out = []
        candidate_zanh_remainder = []
        c_out = []
        c_remainder = []
        c_zanh_out = []
        c_zanh_remainder = []
        h_out = []
        h_remainder = []
        H_t = np.zeros((1, 50), dtype=np.float32)
        ui,uo,uf,uc = np.split(np.transpose(self.U[0]), 4, -1)
        wi,wo,wf,wc = np.split(np.transpose(self.W[0]), 4, -1)
        for j in range(t):
            # print(self.X[4])
            x = np.expand_dims(self.X[j],0)
            f = np.dot(x, wf) + np.dot(H_t,uf) + self.bf
            of = f // 10**self.n
            rf = f % 10**self.n
            f_out.append(of.astype(int).flatten().tolist())
            f_remainder.append(rf.astype(int).flatten().tolist())

            fz = [fun1(fo, self.n, polynomial_degree) for fo in of.astype(int).flatten().tolist()]
            ofz = [zf // 10**(polynomial_degree*self.n) for zf in fz]
            rfz = [zf % 10**(polynomial_degree*self.n) for zf in fz]
            ofz = np.array(ofz)
            rfz = np.array(rfz)
            f_zigmoid_out.append(ofz.copy().astype(int).flatten().tolist())
            f_zigmoid_remainder.append(rfz.astype(int).flatten().tolist())


            i = np.dot(x, wi) + np.dot(H_t,ui) + self.bi
            oi = i // 10**self.n
            ri = i % 10**self.n
            i_out.append(oi.astype(int).flatten().tolist())
            i_remainder.append(ri.astype(int).flatten().tolist())

            iz = [fun1(io, self.n, polynomial_degree) for io in oi.astype(int).flatten().tolist()]
            oiz = [zi // 10**(polynomial_degree*self.n) for zi in iz]
            riz = [zi % 10**(polynomial_degree*self.n) for zi in iz]
            oiz = np.array(oiz)
            riz = np.array(riz)
            i_zigmoid_out.append(oiz.astype(int).flatten().tolist())
            i_zigmoid_remainder.append(riz.astype(int).flatten().tolist())  

            o = np.dot(x, wo) + np.dot(H_t,uo) + self.bo
            oo = o // 10**self.n
            ro = o % 10**self.n
            o_out.append(oo.astype(int).flatten().tolist())
            o_remainder.append(ro.astype(int).flatten().tolist())

            oz = [fun1(ooo, self.n, polynomial_degree) for ooo in oo.astype(int).flatten().tolist()]
            ooz = [zo // 10**(polynomial_degree*self.n) for zo in oz]
            roz = [zo % 10**(polynomial_degree*self.n) for zo in oz]
            ooz = np.array(ooz)
            roz = np.array(roz)
            o_zigmoid_out.append(ooz.astype(int).flatten().tolist())
            o_zigmoid_remainder.append(roz.astype(int).flatten().tolist())

            candidate = np.dot(x, wc) + np.dot(H_t,uc) + self.bc
            oca = candidate // 10**self.n
            rca = candidate % 10**self.n
            candidate_out.append(oca.astype(int).flatten().tolist())
            candidate_remainder.append(rca.astype(int).flatten().tolist())

            caz = [fun2(co, self.n, polynomial_degree) for co in oca.astype(int).flatten().tolist()]
            ocaz = [ca // 10**(polynomial_degree*self.n)for ca in caz]
            rcaz = [ca % 10**(polynomial_degree*self.n) for ca in caz]
            ocaz = np.array(ocaz)
            rcaz = np.array(rcaz)
            candidate_zanh_out.append(ocaz.astype(int).flatten().tolist())
            candidate_zanh_remainder.append(rcaz.astype(int).flatten().tolist())

            C = np.multiply(ofz, c[-1]) + np.multiply(oiz, ocaz)
            oc = C // 10**self.n
            rc = C % 10**self.n

            c.append(oc)
            c_out.append(oc.astype(int).flatten().tolist())
            c_remainder.append(rc.astype(int).flatten().tolist())

            cz = [fun2(co, self.n, polynomial_degree) for co in oc.astype(int).flatten().tolist()]
            ocz = [cz // 10**(polynomial_degree*self.n) for cz in cz]
            rcz = [cz % 10**(polynomial_degree*self.n) for cz in cz]
            ocz = np.array(ocz)
            rcz = np.array(rcz)
            c_zanh_out.append(ocz.astype(int).flatten().tolist())
            c_zanh_remainder.append(rcz.astype(int).flatten().tolist())


            H = np.multiply(ooz, ocz)
            oh = H // 10**self.n
            rh = H % 10**self.n
            H_t = np.expand_dims(oh, 0)


            h.append(oh)
            h_out.append(oh.astype(int).flatten().tolist())
            h_remainder.append(rh.astype(int).flatten().tolist())

        out = (f_out, f_remainder, f_zigmoid_out, f_zigmoid_remainder, i_out, i_remainder, i_zigmoid_out, i_zigmoid_remainder, o_out, o_remainder, o_zigmoid_out, o_zigmoid_remainder, candidate_out, candidate_remainder, candidate_zanh_out, candidate_zanh_remainder, c_out, c_remainder, c_zanh_out, c_zanh_remainder, h_out, h_remainder)
        python_out_names = ('f_temp', 'f_temp_rem', 'f', 'f_rem','i_temp', 'i_temp_rem', 'i', 'i_rem','o_temp', 'o_temp_rem', 'o', 'o_rem', 'c_temp', 'c_temp_rem', 'c', 'c_rem','C_temp', 'C_temp_rem', 'C', 'C_rem', 'h', 'h_rem')
        output = dict(zip(python_out_names, out))
        # print(output['f_temp'])
        # sys.exit()
        return output

class Squeeze():
    def __init__(self) -> None:
        pass

class TotalModel():
    def __init__(self, x, scaling, weights_dict, isInt: bool = False) -> None:
        self.isInt = isInt
        self.weights_dict = weights_dict
        if isInt:
            self.lstm1 = LSTMWorkingInt(x, weights_dict["W0__40"], weights_dict['R0__41'], weights_dict['B0__42'], weights_dict['const_fold_opt__80'], weights_dict['const_fold_opt__80'], scaling)
            self.lstm2 = LSTMWorkingInt(x, weights_dict["W0__56"], weights_dict['R0__57'], weights_dict['B0__58'], weights_dict['const_fold_opt__80'], weights_dict['const_fold_opt__80'], scaling, t = 2)
        else:
            self.lstm1 = LSTMWorking(x, weights_dict["W0__40"], weights_dict['R0__41'], weights_dict['B0__42'], weights_dict['const_fold_opt__80'], weights_dict['const_fold_opt__80'], scaling)
            self.lstm2 = LSTMWorking(x, weights_dict["W0__56"], weights_dict['R0__57'], weights_dict['B0__58'], weights_dict['const_fold_opt__80'], weights_dict['const_fold_opt__80'], scaling, t = 2)


    def run(self, version, t):

        if version == "approx":
            if self.isInt:
                fun1 = taylor_sigmoid_int
                fun2 = taylor_tanh_int
                polynomial_degree = 5
                x = self.lstm1.run_lstm_approx_int(t, fun1 = fun1, fun2 = fun2, polynomial_degree=polynomial_degree)
                y = x.copy()
                shape = np.asarray(x["h"]).shape
                x["h"] = np.asarray(x["h"]).reshape(shape[0],1,shape[1]).astype(float).astype(int)
                self.lstm2.X = x['h']
                # print(self.lstm2.X)
                x = self.lstm2.run_lstm_approx_int(t, fun1 = fun1, fun2 = fun2,polynomial_degree = polynomial_degree)
                shape = np.asarray(x["h"]).shape
                out = np.asarray(x["h"]).reshape(shape[0],1,shape[1]).astype(float).astype(int)[-1,:,:]
                out = Gemm(self.weights_dict, out,scaling = 10**9)
                return out, y, x
            else:
                f1 = taylor_sigmoid
                f2 = taylor_tanh
                x = self.lstm1.run_lstm_approx(t, fun1=f1, fun2 = f2)
                y = x.copy()
                shape = np.asarray(x["h"]).shape
                x["h"] = np.asarray(x["h"]).reshape(shape[0],1,shape[1]).astype(float)
                self.lstm2.X = x['h']
                x = self.lstm2.run_lstm_approx(t, fun1=f1, fun2 = f2)
                shape = np.asarray(x["h"]).shape
                out = np.asarray(x["h"]).reshape(shape[0],1,shape[1]).astype(float)[-1,:,:]
                out = Gemm(self.weights_dict, out)
                return out, y, x
        elif version == "step" and not self.isInt:
            x = self.lstm1.step()
            shape = x["Y"].shape
            x["Y"] = x["Y"].reshape(shape[0],1,shape[3]).astype(float)
            self.lstm2.X = x['Y']
            print("SHAPE", x["Y"].shape)
            x = self.lstm2.step()
            shape = x["Y"].shape
            print("SHAPE2", shape)
            x["Y"] = x["Y"].reshape(shape[0],1,shape[3]).astype(float)[-1,:,:]
            print(x["Y"])
            x["Y"] = Gemm(self.weights_dict, x["Y"])
            return x
        elif version == "step_breakdown" and not self.isInt:
            x = self.lstm1.run_lstm(t)
            shape = np.asarray(x["h"]).shape
            x["h"] = np.asarray(x["h"]).reshape(shape[0],1,shape[1]).astype(float)
            self.lstm2.X = x['h']
            x = self.lstm2.run_lstm(5)
            shape = np.asarray(x["h"]).shape
            x["h"] = np.asarray(x["h"]).reshape(shape[0],1,shape[1]).astype(float)[-1,:,:]
            x["h"] = Gemm(self.weights_dict, x["h"])
            return x
        else:
            raise NotImplementedError
        
