# Generate test vectors for the field implementations using SageMath
# Usage: sage test_vectors.sage

# M31 Field
p = 2**31 - 1
print("M31 Field")
print(f"p = {p}")
F = GF(p)
a = F(3)
print(f"a = {a}")
print(f"a^(-1) = {a^(-1)}")
print(f"a^(11) = {a^(11)}")

# Degree 3 extension
R.<x> = F[]
K.<a> = F.extension(x^3 - 5)
print("M31 Degree 3 Extension")
b = 1 + 2*a + 3*a^2
c = 4 + 5*a + 6*a^2
print(f"b = {b}")
print(f"c = {c}")
print(f"b*c = {b*c}")
print(f"b^(-1) = {b^(-1)}")
print(f"b^(11) = {b^(11)}")
