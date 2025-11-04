import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from math import e, erf, pi, cos, sin, log
from scipy import optimize
from scipy import linalg
from time import time
from numpy import arctan

#array([1.44194622, 2.07471737, 0.73825848])

v_th = 1
v_r = 0
v_rest = 0.9

beta = 6
A = 2
a = 1
B = 2
b = 2


D = 1

#c, t, R = 0.8905755048475765,3.076090951557198,0.33381108153704564

#taus = [0, t]


def update_R(new_R, new_D):
    global R
    global D
    global I
    global p
    global q2
    global abs_q

    R = new_R
    D = new_D
    I = v_rest * (R + D) / D
    #Derived values
    p = 0.5*(D+1)
    interior = (D-1)**2 -4*R
    if interior <= 0:
        abs_q = 0.5 * (-interior)**0.5
        q2 = -abs_q**2
    else:
        print("q is 0.5 *", interior, "** 0.5")
        print("q should be imaginary for this to work")
        raise TypeError
    q2 = -abs_q**2
    return

#update_R(R, D)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def w(d):
    return w_part(d, A, a) - w_part(d, B, b)

def w_part(d, Z, z):
    return (Z / (z * (2*pi)**0.5)) * e**(-d**2 / (2 * z**2))


#------------------------------------------------------------------------------
stopwatch = time()


        
integral_steps = 300

def F_entry(j, k, root):
    integral_end = 10 / min(p, beta, 1)
    increment = integral_end / integral_steps
    
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

    
    

#----------------------------------------------------------------------------


#Format (real part, imaginary part, R)
##id_guesses = []
##lambda_guesses = [0,    #1
##                  0,    #2
##                  (0.5, 4.0),   #3
##                  (0.7, 3.0),   #4
##                  (1.0, 2.5),   #5
##                  (1.2, 2.0),   #6
##                  (1.5, 2.0),   #7
##                  (1.7, 1.5),   #8
##                  (1.7, 1.5),   #9
##                  (1.8, 1.5),   #10
##                  (1.9, 1.2),   #11
##                  (2.0, 1.0),   #12
##                  (2.0, 0.8),   #13
##                  (2.0, 0.2),   #14 #Image starts to get confused here...
##                  (1.5, 0.2),   #15 #Have they turned into real roots?
##                  (1.5, 0.2),   #16
##                  ]
guesses = [0,   #1
           0,   #2
           (-0.021318458033664025,4.186680788335193,0.44999999999999996),   #3
           (-0.020754554671381868,5.332219109564421,0.7500000000000001),    #4
           (-0.02044464907074433,4.4300575724837294,0.8500000000000002),    #5
           (-0.00023008902638912436,5.257310991389755,1.0000000000000002),  #6
           (-0.00584408542888216,4.743260918244081,1.1000000000000003),     #7
           (-0.005706690462837507,5.285589043939644,1.1500000000000004),    #8
           (-0.013618890004832317,5.057839064321797,1.3000000000000005),    #9
           (-0.0014216362728892218,4.595288207419991,1.3000000000000005),   #10
           (-0.003348251604835507,5.096009332092257,1.3500000000000005),    #11
           (-0.005809474461923271,4.913253015048935,1.4500000000000006),    #12
           (-0.0008144542802543692,5.050076706246563,1.3500000000000005),   #13
           ]
solutions = []


stopwatch = time()

outfile = open("stability refined data.txt", mode="w")
outfile.close()
def fileout(data):
    outfile = open("stability refined data.txt", mode="a")
    line = ""
    for item in data:
        line += str(item) + ","
    outfile.write(line[:-1] + "\n")
    outfile.close()

max_spikes = 16
fileout(["n/a"])
fileout(["n/a"])
for spikes in range(3, 14):
    print(time() - stopwatch)
    print(spikes)

    #Load candidates
    waves = []
    infile = open(str(spikes) + "spike.txt")
    for line in infile:
        data = line.strip("\n").split(",")
        waves.append([])
        for item in data:
            waves[-1].append(float(item))
    infile.close()
    #guess = initial_guesses[spikes]
    guess = guesses[spikes - 1]
    wave_id = round(guess[2] / 0.05)
    guess_lambda = guess[:2]
    old_soln = guess_lambda[:]
    while True:
        guess_wave = waves[wave_id]
        c = guess_wave[0]
        R = guess_wave.pop()
        update_R(R, D)
        taus = guess_wave
        taus[0] = 0
        #Constructing blank matrices
        F = np.zeros((len(taus), len(taus)), dtype=complex)
        G = np.zeros((len(taus), len(taus)), dtype=complex)
        diff = np.zeros((len(taus), len(taus)))
        #Reconstruct matrix G and the firing-time-difference matrix
        for j in range(len(taus)):
            for k in range(len(taus)):
                diff[j][k] = taus[j] - taus[k]
        for k in range(len(taus)):
            for j in range(len(taus)):
                G[k,k] += F_entry(j, k, 0)
        soln = optimize.root(get_det2, old_soln, tol=0.00001)
        if soln.success:
            print("{} has root at {}".format(wave_id, soln.x))
            if soln.x[0] > 0:
                #Not there yet
                #Assume we start at the unstable side
                wave_id += 1
                old_soln = soln.x[:]
            else:
                print("R bracketed")
                solutions.append([soln.x[0], soln.x[1], R])
                break
        else:
            print("Something went wrong")
            solutions.append([0])
            break
    fileout(solutions[-1])

print(time() - stopwatch)









