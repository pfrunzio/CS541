# using code to check my work

import sympy as sp

# Define the variables
n = sp.symbols('n', integer=True)  # The upper limit of summation
x = sp.symbols('x1:5')  # Define a vector x with components x1, x2, ..., xn (here n = 4 for example)

# Define the function g(x_i), for example g(x_i) = x_i^2
def g(x_i):
    return x_i**2  # Example function, replace with your own if needed

# Define the sigma notation function
def sigma_function(x, n):
    return sp.summation(g(x[i]), (i, 0, n-1))

# Compute the gradient
gradient = [sp.diff(sigma_function(x, n), x_i) for x_i in x]

# Display the gradient
for i, grad in enumerate(gradient):
    print(f"∂f/∂x{i+1}: {grad}")