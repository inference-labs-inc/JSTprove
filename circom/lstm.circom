pragma circom 2.0.0;

include "./circomlib-ml/circuits/circomlib-matrix/matElemMul.circom";
include "./circomlib-ml/circuits/Zanh.circom";
include "./circomlib-ml/circuits/Zigmoid.circom";
include "./integerDivision.circom";


// include "./circomlib-ml/circuits/Zigmoid.circom";


// Input x and scaling, output out + remainder
function Zigmoid_int(x){
    var t_1 = 502073021 * (10**(9*3));
    var t_2 = x * 198695283 * (10**(9*2));
    var t_3 = (x * x * 1570683) * (10**(9*1));
    var t_4 = (x * x * x * 4001354);

    var out = t_1 + t_2 - t_3 - t_4;
    return out;
}
// 6769816 + 554670504 * x - 9411195 * x**2 - 14187547 * x**3
function Zanh_int(x){
    var t_1 = 6769816 * (10**(9*3));
    var t_2 = x * 554670504 * (10**(9*2));
    var t_3 = (x * x * 9411195) * (10**(9*1));
    var t_4 = (x * x * x * 14187547);

    var out = t_1 + t_2 - t_3 - t_4;
    return out;
}



template LSTM (t, h, i) { // n is 10 to the number of decimal places


    // signal input step_in[6]; // hidden states + cell states
    signal output out[t][1][1][h];

    // private inputs
    signal input in[1][t]; 
    signal input scaling;

    signal input W0[1][4*h][i]; // (i,o,f,g)
    signal input R0[1][4*h][h]; //(iofg)
    signal input B0[1][8*h]; //W(iofg), R(iofg)
    signal input const_fold_opt[1][1][h];

    signal input f_temp_1[t][h];
    signal input f_temp_rem_1[t][h];
    signal input f_1[t][h];
    signal input f_rem_1[t][h];

    signal input i_temp_1[t][h];
    signal input i_temp_rem_1[t][h];
    signal input i_1[t][h];
    signal input i_rem_1[t][h];

    signal input o_temp_1[t][h];
    signal input o_temp_rem_1[t][h];
    signal input o_1[t][h];
    signal input o_rem_1[t][h];

    signal input c_temp_1[t][h];
    signal input c_temp_rem_1[t][h];
    signal input c_1[t][h];
    signal input c_rem_1[t][h];

    signal input C_temp_1[t][h];
    signal input C_temp_rem_1[t][h];
    signal input C_1[t][h];
    signal input C_rem_1[t][h];

    signal input h_1[t][h];
    signal input h_rem_1[t][h];

    signal temp_h_0[t+1];

    temp_h_0[0] <== 0;


    // for (var k=0; k<h; k++) {
    //         // First
    //         // it = fun1(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
    //         i_temp[0][k][0] <== in[0][0]*W0__40[0][k] + B0__42[0][k + 4*h];
    //         i_temp[0][k][1] <== const_fold_opt__80[0][0][k]*R0__41[0][k][k] + B0__42[0][k];
    //         i_temp[0][k][2] <== i_temp[0][k][0] + i_temp[0][k][1];
    //         zigmoid_int_i[0][k] <-- Zigmoid_int(i_temp[0][k][2]);
    //         zigmoid_quot_i[0][k] <-- zigmoid_int_i[0][k] \ 10**(3*9);
    //         zigmoid_rem_i[0][k] <-- zigmoid_int_i[0][k] % 10**(3*9);

    //         zigmoid[0][k][0] = Zigmoid(10**9);
    //         zigmoid[0][k][0].in <== i_temp[0][k][2];
    //         zigmoid[0][k][0].out <== zigmoid_quot_i[0][k];
    //         zigmoid[0][k][0].remainder <== zigmoid_rem_i[0][k];

    //         c_t[0][k] <-- 0;
    //         h_t[0][k] <-- 0;


    // }

    for (var j=1; j<t; j++) {
        
        for (var k=0; k<h; k++) {
            // First
            // it = fun1(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
            // const_fold_opt__80[0][0][k] is a placeholder here for now
            // i_temp[j][k][0] <== in[0][j]*W0__40[0][k] + B0__42[0][k + 4*h];
            // i_temp[j][k][1] <== const_fold_opt__80[0][0][k]*R0__41[0][k][k] + B0__42[0][k];
            // i_temp[j][k][2] <== i_temp[j][k][0] + i_temp[j][k][1];// scaled up by scaling**2
            
            // zigmoid_int_i[j][k] <-- Zigmoid_int(i_temp[j][k][2]);
            // zigmoid_quot_i[j][k] <-- zigmoid_int_i[j][k] \ 10**(3*9);
            // zigmoid_rem_i[j][k] <-- zigmoid_int_i[j][k] % 10**(3*9);

            // zigmoid[j][k][0] = Zigmoid(10**9);
            // zigmoid[j][k][0].in <== i_temp[j][k][2];
            // zigmoid[j][k][0].out <== zigmoid_quot_i[j][k];
            // zigmoid[j][k][0].remainder <== zigmoid_rem_i[j][k];

            // i_t[j][k] <== zigmoid_quot_i[j][k];
            out[j][0][0][k] <== i_temp_1[j][k];

            // // ft = fun1(Xt*(Wf^T) + Ht-1*(Rf^T) + Wbf + Rbf)
            // f_temp[j][k][0] <== in[0][j]*W0__40[0][k+h*2] + B0__42[0][k + 6*h];
            // f_temp[j][k][1] <== const_fold_opt__80[0][0][k]*R0__41[0][k+2*h][k] + B0__42[0][k+2*h];
            // f_temp[j][k][2] <== f_temp[j][k][0] + f_temp[j][k][1];// scaled up by scaling**2

            // zigmoid_int_f[j][k] <-- Zigmoid_int(f_temp[j][k][2]);
            // zigmoid_quot_f[j][k] <-- zigmoid_int_f[j][k] \ 10**(3*9);
            // zigmoid_rem_f[j][k] <-- zigmoid_int_f[j][k] % 10**(3*9);

            // zigmoid[j][k][1] = Zigmoid(10**9);
            // zigmoid[j][k][1].in <== f_temp[j][k][2];
            // zigmoid[j][k][1].out <== zigmoid_quot_f[j][k];
            // zigmoid[j][k][1].remainder <== zigmoid_rem_f[j][k];

            // f_t[j][k] <== zigmoid_quot_f[j][k];

            // // ot = fun1(Xt*(Wo^T) + Ht-1*(Ro^T) + Wbo + Rbo)
            // o_temp[j][k][0] <== in[0][j]*W0__40[0][k+h] + B0__42[0][k + 5*h];
            // o_temp[j][k][1] <== const_fold_opt__80[0][0][k]*R0__41[0][k+h][k] + B0__42[0][k+h];
            // o_temp[j][k][2] <== o_temp[j][k][0] + o_temp[j][k][1];// scaled up by scaling**2

            // zigmoid_int_o[j][k] <-- Zigmoid_int(o_temp[j][k][2]);
            // zigmoid_quot_o[j][k] <-- zigmoid_int_o[j][k] \ 10**(3*9);
            // zigmoid_rem_o[j][k] <-- zigmoid_int_o[j][k] % 10**(3*9);

            // zigmoid[j][k][2] = Zigmoid(10**9);
            // zigmoid[j][k][2].in <== o_temp[j][k][2];
            // zigmoid[j][k][2].out <== zigmoid_quot_o[j][k];
            // zigmoid[j][k][2].remainder <== zigmoid_rem_o[j][k];

            // o_t[j][k] <== zigmoid_quot_o[j][k];

            // // gt = fun2(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
            // g_temp[j][k][0] <== in[0][j]*W0__40[0][k+3*h] + B0__42[0][k + 7*h];
            // g_temp[j][k][1] <== const_fold_opt__80[0][0][k]*R0__41[0][k+3*h][k] + B0__42[0][k+3*h];
            // g_temp[j][k][2] <== o_temp[j][k][0] + o_temp[j][k][1];// scaled up by scaling**2

            // zanh_int_g[j][k] <-- Zanh_int(g_temp[j][k][2]);
            // zanh_quot_g[j][k] <-- zanh_int_g[j][k] \ 10**(3*9);
            // zanh_rem_g[j][k] <-- zanh_int_g[j][k] % 10**(3*9);

            // zanh[j][k][0] = Zanh(10**9);
            // zanh[j][k][0].in <== g_temp[j][k][2];
            // zanh[j][k][0].out <== zanh_quot_g[j][k];
            // zanh[j][k][0].remainder <== zanh_rem_g[j][k];

            // g_t[j][k] <== zanh_quot_g[j][k];
            
            // // ct = ft (.) ct-1 + it (.) gt

            // // hadamard[j][k] = matElemMul()
            // c_temp[j][k][0] <== f_t[j][k] * c_t[j-1][k];
            // c_temp[j][k][1] <== i_t[j][k] * g_t[j][k];

            // c_t[j][k] <== c_temp[j][k][0] + c_temp[j][k][1];
            // // MUST ADD INTEGER DIVISION, TO REDUCE THE SCALING

            // // THINGS TO DO, LINEUP scaling with integer divisions
            // // Incorporate negatives
            

            // // Ht = ot (.) fun3(Ct)
            // zanh_int_h[j][k] <-- Zanh_int(c_t[j][k]);
            // zanh_quot_h[j][k] <-- zanh_int_h[j][k] \ 10**(3*9);
            // zanh_rem_h[j][k] <-- zanh_int_h[j][k] % 10**(3*9);

            // zanh[j][k][1] = Zanh(10**9);
            // zanh[j][k][1].in <== c_t[j][k];
            // zanh[j][k][1].out <== zanh_quot_h[j][k];
            // zanh[j][k][1].remainder <== zanh_rem_h[j][k];

            // h_t[j][k] <== zanh_quot_h[j][k] * o_t[j][k];



            // // out[j][0][0][k] <== h_t[j][k];
        }
    }
}



// component main { public [step_in] } = LSTM(10**9);