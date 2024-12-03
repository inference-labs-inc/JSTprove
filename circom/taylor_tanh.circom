pragma circom 2.0.0;

// Polynomial approximation for the sigmoid layer
// if p == 5:
//     z =  0 * 10**(5*n) + 1000000000 * x * 10**(4*n) + 0 * x**2 * 10**(3*n) - 333333333 * x**3 * 10**(2*n) + 0 * x**4 * 10**(1*n) + 133333333 * x**5
// elif p == 3:    
//     z =  0 * 10**(3*n) + 1000000000 * x * 10**(2*n) - 333333333 * x**3
// n = 10 to the power of the number of decimal places
// out and remainder are from division by n**2 so out has the same number of decimal places as in
template TaylorTanh5 (n) {
    signal input in;
    signal input out;
    signal input remainder;
    
    assert(remainder < (n**2) * (10**9));

    signal tmp[2];

    tmp[0] <== 133333333 * in * in; // x**2 * 208333

    tmp[1] <== 1000000000 * n**4 - 333333333 * in * in * n**2 + tmp[0] * in * in; // 250000000 * 10**(4*n) - 20833333 * x**2 * 10**(2*n) + 2083333 * x**4
    // log(502073021 * n**3 + in * tmp);
    out * (n**4) * (10**9) + remainder === in * tmp[1];
}

template TaylorTanh3 (n) {
    signal input in;
    signal input out;
    signal input remainder;
    
    assert(remainder < (n**2) * (10**9));

    signal tmp;

    tmp <== 1000000000 * n**2 - 333333333 * in * in;
    // log(502073021 * n**3 + in * tmp);
    out * (n**2) * (10**9) + remainder ===  in * tmp;
}

// component main { public [ out ] } = Zigmoid(10**9);
