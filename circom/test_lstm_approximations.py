import numpy as np
def zanh(x):
    # x = x/
    y =  0.006769816 + 0.554670504 * x - 0.009411195 * x**2 - 0.014187547 * x**3
    # y = tanh(x) 
    return y

def zanh_int(x, n):
    # print(x/(10**n))
    z = 6769816 * 10**(3*n) + 554670504 * x * 10**(2*n) - 9411195 * x**2 * 10**n - 14187547 * x**3
    return z

def zigmoid(x):
    # y =  1/ (1 + exp(-x))
    z = 0.502073021 + 0.198695283 * x - 0.001570683 * x**2 - 0.004001354 * x**3
    return z

def zigmoid_int(x, n): 
    z =  502073021 * 10**(3*n) + 198695283 * x * 10**(2*n) - 1570683 * x**2 * 10**n - 4001354 * x**3
    # print(x, y)
    return z

def taylor_tanh(x):
    return x - (x**3)/3 + (2*(x**5))/15

def taylor_sigmoid(x):
    return 0.5 + x/4 - (x**3)/48 + (x**5)/480

def taylor_tanh_int(x, n, p):
    # n = 9
    if p == 5:
        z =  0 * 10**(5*n) + 1000000000 * x * 10**(4*n) + 0 * x**2 * 10**(3*n) - 333333333 * x**3 * 10**(2*n) + 0 * x**4 * 10**(1*n) + 133333333 * x**5
    elif p == 3:    
        z =  0 * 10**(3*n) + 1000000000 * x * 10**(2*n) + 0 * x**2 * 10**(1*n) - 333333333 * x**3
    return z

def taylor_sigmoid_int(x, n, p):
    # n = 9
    if p ==5:
        z = 500000000 * 10**(5*n) + 250000000 * x * 10**(4*n) + 0 * x**2 * 10**(3*n) - 20833333 * x**3 * 10**(2*n) + 0 * x**4 * 10**(n) + 2083333 * x**5
    elif p==3:
        z = 500000000 * 10**(3*n) + 250000000 * x * 10**(2*n) + 0 * x**2 * 10**(n) - 20833333 * x**3 
    return z

##### Activation Functions #####
def sigmoid(input, derivative = False):
    if derivative:
        return input * (1 - input)
    
    return 1 / (1 + np.exp(-input))

def tanh(input, derivative = False):
    if derivative:
        return 1 - input ** 2
    
    return np.tanh(input)

# 1. Piecewise Linear Approximation
def piecewise_linear_sigmoid(x):
    if x < -3:
        return 0
    elif x > 3:
        return 1
    else:
        return 0.5 + 0.15 * x

# 2. Mini-max Approximation (Degree 3)
def minimax_sigmoid(x):
    return 0.5 + 0.19751 * x + -0.0033288 * x**3

# 3. Bandish Approximation
def bandish_sigmoid(x):
    return 0.5 + x / (2 * (1 + abs(x)))

# 4. Algebraic Approximation
def algebraic_sigmoid(x):
    return x / (1 + abs(x))

# 5. Piecewise Quadratic Approximation
def piecewise_quadratic_sigmoid(x):
    if x < -2.5:
        return 0
    elif x > 2.5:
        return 1
    else:
        return 0.5 + 0.19 * x + 0.0045 * x**2

# 6. Rational Approximation
def rational_sigmoid(x):
    return 0.5 * (x / (1 + abs(x)) + 1)

# # Test the approximations
# x = np.linspace(-5, 5, 100)
# true_sigmoid = sigmoid(x)

# for func in [piecewise_linear_sigmoid, minimax_sigmoid, bandish_sigmoid, 
#              algebraic_sigmoid, piecewise_quadratic_sigmoid, rational_sigmoid]:
#     approx = np.vectorize(func)(x)
#     mse = np.mean((true_sigmoid - approx)**2)
#     print(f"{func.__name__} MSE: {mse}")