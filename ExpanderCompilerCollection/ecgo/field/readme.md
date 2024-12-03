# field

Within gnark's builder, the Rank-1 Constraint System (R1CS) is used, such as in `gnark/constraint/bn254`. R1CS implements arithmetic over a field.

Since the R1CS implementation is private and there is a need to support other fields, an independent library for field arithmetic was created.

Currently, the supported fields include `bn254` and `m31`, where the modulus for `m31` is $2^{31}-1$.