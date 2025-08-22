# FAQ

## Do I need to specify a circuit class?
**No.** JSTProve defaults to **GenericModelONNX**.

## Can I run only witness/prove/verify without compile?
**Yes**, as long as you already have the **circuit** and **quantized ONNX** produced by a prior compile.

## Where does the CLI put reshaped inputs?
It writes a local `*_reshaped.json` in your **current working directory** during witness/verify.

## What exactly is proven?
That the **quantized model**, when evaluated on your input, produces the stated output — and that the **circuit constraints hold** (via GKR/sumcheck in Expander).
