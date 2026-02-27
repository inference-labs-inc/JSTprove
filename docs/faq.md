# FAQ

## Do I need to specify a circuit class?
**No.** JSTprove uses the `Circuit` type by default.

## Can I run only witness/prove/verify without compile?
**Yes**, as long as you already have the **compiled circuit** (msgpack bundle) produced by a prior compile.

## What exactly is proven?
That the model, when evaluated on your input, produces the stated output -- and that the **circuit constraints hold** (via GKR/sumcheck in Expander, or via Remainder_CE for the remainder backend).
