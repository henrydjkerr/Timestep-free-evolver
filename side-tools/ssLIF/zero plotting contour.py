"""
Plots the zeros of the real and imaginary parts of the stability function
for the purpose of finding eigenvalues.
Uses contour plotting rather than linearly interpolated scatter plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from math import e, erf, pi, cos, sin, log
from scipy import optimize
from scipy import linalg
from time import time
from numpy import arctan


v_th = 1
v_r = 0

beta = 6
A = 2
a = 1
B = 2
b = 2


D = 1

c, t, R = 0.8905755048475765,3.076090951557198,0.33381108153704564

taus = [0, t]


p = 0.5*(D+1)
q = 0.5*( (D-1)**2 -4*R )**0.5
if type(q) != type(1j):
    print("q is", q)
    print("q should be imaginary for this to work")
    raise TypeError
abs_q = abs(q)
q2 = -abs_q**2

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def w(d):
    return w_part(d, A, a) - w_part(d, B, b)

def w_part(d, Z, z):
    return (Z / (z * (2*pi)**0.5)) * e**(-d**2 / (2 * z**2))


#------------------------------------------------------------------------------
stopwatch = time()

diff = np.zeros((len(taus), len(taus)))
for j in range(len(taus)):
    for k in range(len(taus)):
        diff[j][k] = taus[j] - taus[k]
        
integral_steps = 100
integral_end = 10 / min(p, beta, 1)
increment = integral_end / integral_steps

def F_entry(j, k, root):
    d = diff[j][k]
    #First integral
    integral = 0
    for step in range(integral_steps):
        T = -increment * (step + 0.5) #Clunky for now
        coeff = e**(p * T) * (cos(abs_q * T) + ((1-p)/abs_q)*sin(abs_q * T))
        part_1 = e**(root * T) * w(c*(T - d))
        #Second integral
        part_2 = 0
        for step2 in range(integral_steps):
            r = T - increment * (step2 + 0.5)
            part_2 += e**(beta*(r - T) + root * r) * w(c*(r - d))
        part_2 *= increment * beta
        integral += increment * coeff * (part_1 - part_2)

    integral *= beta / 2
    if j < k:
        reset = (v_th - v_r) * (1/(2*c)) * e**(d*(p + root))
        reset *= cos(abs_q * d) - ((p**2 - q2 - p)/abs_q) * sin(abs_q * d)
    else:
        reset = 0
    return integral + reset

def make_F(root):
    for k in range(len(taus)):
        for j in range(len(taus)):
            F[j,k] = F_entry(j, k, root)

def get_det(root):
    make_F(root)
    matrix = F - G
    return np.linalg.det(matrix)

def get_det2(root_parts):
    root = root_parts[0] + root_parts[1] * 1j
    det = get_det(root)
    return (np.real(det), np.imag(det))

def get_det_mag(root_parts):
    r, i = get_det2(root_parts)
    return (r**2 + i**2)**0.5

def get_real(root_parts):
    return get_det2(root_parts)[0]

def get_imag(root_parts):
    return get_det2(root_parts)[1]

def PAC_ver(root_parts):
    global func
    global guess
    global tangent
    return [func(root_parts), np.dot((root_parts - guess), tangent)]

#----------------------------------------------------------------------------

#Constructing F matrix
F = np.zeros((len(taus), len(taus)), dtype=complex)

#Constructing G matrix
G = np.zeros((len(taus), len(taus)), dtype=complex)
for k in range(len(taus)):
    for j in range(len(taus)):
        G[k,k] += F_entry(j, k, 0)


#------------------------------------------------------------------------------

#Used for adjusting step sizes
base_epsilon = 0.2

solns_r = []
solns_i = []
solns_both = [solns_r, solns_i]

r_arr = np.arange(-2, 2, base_epsilon)
i_arr = np.arange(0, 8, base_epsilon)
v_arr = np.ndarray((2, len(i_arr), len(r_arr)))

#Pick your root function
funcs_both = [get_real, get_imag]

for x, r in enumerate(r_arr):
    for y, i in enumerate(i_arr):
        value = get_det2((r, i))
        v_arr[0,y,x] = value[0]
        v_arr[1,y,x] = value[1]
    
plt.figure()#figsize=(5,5))
#plt.figure(figsize=(4, 4), dpi=800)
plt.axvline(x=0, c="gray", linestyle="dashed")
##cols = ((0.8, 0, 0.5), (0, 0.8, 0.5))
plt.contour(r_arr, i_arr, v_arr[0])
plt.contour(r_arr, -1 * i_arr, v_arr[0])
plt.contour(r_arr, i_arr, v_arr[1])
plt.contour(r_arr, -1 * i_arr, v_arr[1])

plt.xlabel("Real part of $\lambda$")
plt.ylabel("Imaginary part of $\lambda$")
plt.title("Zeros of $F(\lambda) - G$")

print("Time:", time() - stopwatch)
plt.show()






