"""
 - Select A, a, beta
 - Find c for given parameters
 - Generate initial conditions from c
 - Run simulation
 - Report actual time that neuron 0 fires
"""

import numpy
from time import time

import find_c_theory
import profile_generator
import time_stepper

def find_c(A, a, beta):
    c = find_c_theory.find_c(A, a, beta, v_th - I)
    voltage, synapse = profile_generator.make(I, A, a, beta, c, x_dim, x_points)

    limit = 5*(x_dim/c)
    model_time = time_stepper.get_zero_time(v_th, v_r, I, A, a, beta,
                                            voltage, synapse, dx, limit)
    if model_time >= limit:
        print("Numerical wave couldn't reach boundary in expected time.")
    c_numeric = (x_dim/2) / model_time
    return c, c_numeric

#------------------------------------------

stopwatch = time()

x_dim = 20
x_points = 500
dx = x_dim/x_points

v_th = 1
v_r = -10
I = 0.9

b = 1.001

#A_list = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
#a_list = [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1]
#beta_l = [b, b, b, b, b, b, b, b, b, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3]
A_list = [1, 2, 3, 1, 2, 3]
a_list = [2, 2, 2, 3, 3, 3]
beta_l = [3, 3, 3, 3, 3, 3]

#a_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,]
#A_list = [1, 2, 3, 4, 5, 6, 7, 8, 9,10, 1, 2, 3, 4, 5, 6, 7, 8, 9,10]
#beta_l = [b, b, b, b, b, b, b, b, b, b, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

a_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,]
beta_l = [b, 2, 3, 4, 5, 6, 7, 8, 9,10, b, 2, 3, 4, 5, 6, 7, 8, 9,10]
A_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

print("A \t a \t beta \t c_theory \t c_numeric")

for n in range(len(A_list)):
    A = A_list[n]
    a = a_list[n]
    beta = beta_l[n]

    #A = A**0.5
    beta = beta**0.5
    
    c_theory, c_numeric = find_c(A, a, beta)
    print("{}\t {}\t {}\t {}\t {}".format(A, a, beta, c_theory, c_numeric))


print("Time taken: ", time() - stopwatch)



