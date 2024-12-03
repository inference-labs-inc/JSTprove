pragma circom 2.0.0;

/*This circuit template checks that c is the multiplication of a and b.*/  

include "./lstm.circom";


// Reshape layer with that accepts a 1D input
template Reshape2D (nRows, nCols, nChannels) {
    signal input in[nRows*nCols*nChannels];
    signal output out[nRows][nCols][nChannels];
    signal one;
    one <-- 1;

    for (var i=0; i<nRows; i++) {
        for (var j=0; j<nCols; j++) {
            for (var k=0; k<nChannels; k++) {
                out[i][j][k] <== in[i*nCols*nChannels + j*nChannels + k]*one;
            }
        }
    }
}


template LstmFunctionPythonWork () {  
    // Declaration of signals.  
    signal input input_data[1][5];  
    signal input scaling;
    // // Weights for LSTM1
    // // signal input W0__40[1][200]; // (i,o,f,g)
    // // signal input R0__41[1][200][50]; //(iofg)
    // // signal input B0__42[1][400]; //W(iofg), R(iofg)

    signal input wi_1[1][50];
    signal input wo_1[1][50];
    signal input wf_1[1][50];
    signal input wc_1[1][50];

    signal input ui_1[1][50][50];
    signal input uo_1[1][50][50];
    signal input uf_1[1][50][50];
    signal input uc_1[1][50][50];

    signal input bi_1[1][50];
    signal input bo_1[1][50];
    signal input bf_1[1][50];
    signal input bc_1[1][50];

    signal input wi_2[1][50][50];
    signal input wo_2[1][50][50];
    signal input wf_2[1][50][50];
    signal input wc_2[1][50][50];

    signal input ui_2[1][50][50];
    signal input uo_2[1][50][50];
    signal input uf_2[1][50][50];
    signal input uc_2[1][50][50];

    signal input bi_2[1][50];
    signal input bo_2[1][50];
    signal input bf_2[1][50];
    signal input bc_2[1][50];



    signal input const_fold_opt__80[1][1][50];

    // signal input W0__56[1][200][50]; // (i,o,f,g)
    // signal input R0__57[1][200][50]; //(iofg)
    // signal input B0__58[1][400]; //W(iofg), R(iofg)

    // // Input LSTM intermediary calculations First LSTM:

    signal input f_temp_1[5][50];
    signal input f_temp_rem_1[5][50];
    signal input f_1[5][50];
    signal input f_rem_1[5][50];

    signal input i_temp_1[5][50];
    signal input i_temp_rem_1[5][50];
    signal input i_1[5][50];
    signal input i_rem_1[5][50];

    signal input o_temp_1[5][50];
    signal input o_temp_rem_1[5][50];
    signal input o_1[5][50];
    signal input o_rem_1[5][50];

    signal input c_temp_1[5][50];
    signal input c_temp_rem_1[5][50];
    signal input c_1[5][50];
    signal input c_rem_1[5][50];

    signal input C_temp_1[5][50];
    signal input C_temp_rem_1[5][50];
    signal input C_1[5][50];
    signal input C_rem_1[5][50];

    signal input h_1[5][50];
    signal input h_rem_1[5][50];

    // Second LSTM Inputs

    signal input f_temp_2[5][50];
    signal input f_temp_rem_2[5][50];
    signal input f_2[5][50];
    signal input f_rem_2[5][50];

    signal input i_temp_2[5][50];
    signal input i_temp_rem_2[5][50];
    signal input i_2[5][50];
    signal input i_rem_2[5][50];

    signal input o_temp_2[5][50];
    signal input o_temp_rem_2[5][50];
    signal input o_2[5][50];
    signal input o_rem_2[5][50];

    signal input c_temp_2[5][50];
    signal input c_temp_rem_2[5][50];
    signal input c_2[5][50];
    signal input c_rem_2[5][50];

    signal input C_temp_2[5][50];
    signal input C_temp_rem_2[5][50];
    signal input C_2[5][50];
    signal input C_rem_2[5][50];

    signal input h_2[5][50];
    signal input h_rem_2[5][50];





    signal output out[1][5];


    component Reshape2D;
    // (1, 200, 1) (1, 200, 50) (1, 400) (1, 1, 50)
    component LSTM1;

    // Step 1, reshape the inputs

    Reshape2D = Reshape2D(1,5,1);
    Reshape2D.in <== input_data[0];
    // out <== Reshape2D.out;


    // LSTM1 = LSTM(5, 50, 1);
    // LSTM1.in <== input_data;
    // LSTM1.scaling <== scaling;
    // LSTM1.W0 <== W0__40;
    // LSTM1.R0 <== R0__41;
    // LSTM1.B0 <== B0__42;
    // LSTM1.const_fold_opt <== const_fold_opt__80;

    
    out <== input_data;
    // Step 2, LSTM
}

component main = LstmFunctionPythonWork();