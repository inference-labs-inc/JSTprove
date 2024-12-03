pragma circom 2.0.0;

/*This circuit template checks that c is the multiplication of a and b.*/  

template MultiplierTensor (n) {  
    var i;

    // Declaration of signals.  
    signal input a[n];
    signal input b[n];
    signal output out[n];

    // Constraints.  
    for (i=0; i<n; i++) {
        out[i] <== a[i] * b[i];
    }
}

component main = MultiplierTensor(1024);