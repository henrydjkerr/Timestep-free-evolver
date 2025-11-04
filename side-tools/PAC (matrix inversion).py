"""PAC using matrix inversion for tracking solutions as R changes."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from math import e, erf, pi, cos, sin, log
from scipy import optimize
from scipy import linalg
from scipy import differentiate
import time


#Model parameters
beta = 6
A = 2
a = 1
B = 2
b = 2

R = 3 #Placeholder
D = 1
v_rest = 0.9

v_th = 1
v_r = 0

#Solver settings
base_epsilon = 0.005        #Controls PAC step size
jacobian_epsilon = 0.0001   #For estimating derivatives
rootfinder_tolerance = 0.0000005 #For deciding when a root has converged
steps = 150                 #Maximum number of PAC steps for each starting point
N = 0                       #File name index start point

#Picking initial conditions:
#Program iterates across start_points to produce each branch
#The first half of each entry is a list [c, t_1, ..., t_n, R]
#The second is an initial direction for the tangent vector (same variables)
#For beta = 6, spikes = 2:
spikes = 2
start_points = [
    ([2.9382511993738913,1.5420917498971292,3.2301131679240407], [0, 0, -1]),
    ([2.9382511993738913,1.5420917498971292,3.2301131679240407], [0, 0, 1]),
    ]

spikes = 1
start_points = [
    ([3.032528409489903,3.0], [0, -1]),
    ([3.032528409489903,3.0], [0, 1]),
    ]
    


#Alt: for importing start points from file
##start_points = []
##fwd = [0, 0, 0, 1]
##bwd = [0, 0, 0, -1]
##infile = open("3 spike data curated small.csv")
##for line in infile:
##    values = line.strip("\n").split(",")
##    point = []
##    for value in values:
##        point.append(float(value))
##    start_points.append((point[:], fwd[:]))
##    #start_points.append((point[:], bwd[:]))
##infile.close()



def update_R(new_R):
    global R
    global I
    global p
    global q2
    global abs_q

    R = new_R
    I = v_rest * (R + D) / D
    #Derived values
    p = 0.5*(D+1)
    interior = (D-1)**2 - 4*R
    if interior <= 0:
        abs_q = 0.5 * (-interior)**0.5
        q2 = -abs_q**2
    else:
        print("q is 0.5 *", interior, "** 0.5")
        print("q should be imaginary for this to work")
        raise TypeError
    q2 = -abs_q**2
    return

update_R(R)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#General integration function

def one(x):
    return 1

divisions = 800
stdevs = 4
def calc_integral(t, mu, sigma, param, param2 = None,
                  func_name = None, lower = None, upper = None):
    #Parse the optional inputs
    if func_name == "sine":
        func = sin
    elif func_name == "cosine":
        func = cos
    else:
        func = one
    if param2 == None:
        param2 = param
    if upper == None:
        upper = t
        
    #Window of integration
    lower_temp = min(upper, mu - sigma**2 * param2) - stdevs * sigma
    if lower == None:    
        lower = upper - stdevs * sigma
    lower = max(lower_temp, lower)
    
    dT = (upper - lower) / divisions
    total = 0
    for x in range(divisions):
        T = lower + dT * (x + 0.5)
        total += func(abs_q * (t - T))  \
                 * e**(-0.5 * ((T - mu)/sigma)**2 - param*t + param2*T)
    total *= dT / (sigma * (2*pi)**0.5)
    return total

#-----------------------------------------------------------------------------
#Calculations for s

def s(t):
    return beta * (part_s(A, a, t) - part_s(B, b, t))

def part_s(Z, z, t):
    total = 0
    sigma = z / c
    for mu in firing_times:
        total += calc_integral(t, mu, sigma, beta)
    total *= Z
    return total

#-----------------------------------------------------------------------------
#Calculations for v

def v(t, t_old = None, u_old = None):
    coeff_cos = (2*p - beta - 1) / ((p - beta)**2 - q2)
    coeff_sin = (p**2 + q2 - p*(beta + 1) + beta) / (((p - beta)**2 - q2)*abs_q)
    
    part_I = I * (2*p - 1) / (p**2 - q2)
    part_init = 0
    if t_old != None:
        part_I *= 1 - e**(-p * (t - t_old)) * cos(abs_q * (t - t_old))
        part_I -= I * (p**2 + q2 - p) / ((p**2 - q2) * abs_q)           \
                  * e**(-p * (t - t_old)) * sin(abs_q * (t - t_old))
        part_init = e**(-p * (t - t_old))       \
                    * (v_r * cos(abs_q * (t - t_old))
                       - ((v_r*(1 - p) + u_old) / abs_q)
                          * sin(abs_q * (t - t_old)))
    part_s = s(t) * coeff_cos

    coeff_cos = (2*p - beta - 1) / ((p - beta)**2 - q2)
    coeff_sin = (p**2 + q2 - p*(beta + 1) + beta) / (((p - beta)**2 - q2)*abs_q)
    part_t_old = 0
    if t_old != None:
        part_t_old = s(t_old) * e**(-p * (t - t_old))       \
                     * (  coeff_cos * cos(abs_q * (t - t_old))
                        + coeff_sin * sin(abs_q * (t - t_old)))

    return part_I + part_init + part_s - part_t_old \
           + part_v(A, a, t, t_old) - part_v(B, b, t, t_old)

def part_v(Z, z, t, t_old):
    coeff_cos = (2*p - beta - 1) / ((p - beta)**2 - q2)
    coeff_sin = (p**2 + q2 - p*(beta + 1) + beta) / (((p - beta)**2 - q2)*abs_q)

    sigma = z / c
    part_cos = 0
    part_sin = 0
    for mu in firing_times:
        part_cos += calc_integral(t, mu, sigma, p,
                                  func_name = "cosine", lower = t_old)
        part_sin += calc_integral(t, mu, sigma, p,
                                  func_name = "sine", lower = t_old)
    part_cos *= coeff_cos
    part_sin *= coeff_sin
    return -beta * Z * (part_cos + part_sin)

#-----------------------------------------------------------------------------
#Calculations for u

def u(t, t_old = None, u_old = None):
    coeff_cos = 1 / ((p - beta)**2 - q2)
    coeff_sin = (p - beta) / (((p - beta)**2 - q2) * abs_q)
    
    part_I = R * I / (p**2 - q2)
    part_init = 0
    if t_old != None:
        part_I *= 1 - e**(-p * (t - t_old)) * cos(abs_q * (t - t_old))
        part_I -= R * I * (p / ((p**2 - q2) * abs_q))   \
                  * e**(-p * (t - t_old)) * sin(abs_q * (t - t_old))
        part_init = e**(-p * (t - t_old))       \
                    * (u_old * cos(abs_q * (t - t_old))
                       + ((R * v_r + u_old * (1 - p)) / abs_q)
                          * sin(abs_q * (t - t_old)))
    part_s = s(t) * R * coeff_cos

    part_t_old = 0
    if t_old != None:
        part_t_old = s(t_old) * e**(-p * (t - t_old))       \
                     * R * (  coeff_cos * cos(abs_q * (t - t_old))
                            + coeff_sin * sin(abs_q * (t - t_old)))
    return part_I + part_init + part_s - part_t_old \
           + part_u(A, a, t, t_old) - part_u(B, b, t, t_old)

def part_u(Z, z, t, t_old):
    coeff_cos = 1 / ((p - beta)**2 - q2)
    coeff_sin = (p - beta) / (((p - beta)**2 - q2) * abs_q)

    sigma = z / c
    part_cos = 0
    part_sin = 0
    for mu in firing_times:
        part_cos += calc_integral(t, mu, sigma, p,
                                   func_name = "cosine", lower = t_old)
        part_sin += calc_integral(t, mu, sigma, p,
                                   func_name = "sine", lower = t_old)
    part_cos *= coeff_cos
    part_sin *= coeff_sin
    return -R * beta * Z * (part_cos + part_sin)
    

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def root_function(inputs):
    global c
    global firing_times
    global R
    global base_estimate
    global tangent
    
    c = inputs[0]
    R = inputs[-1]
    update_R(R)
    
    firing_times = np.zeros(spikes)
    for n in range(1, spikes):
        firing_times[n] = inputs[n]
    
    outputs = np.zeros(len(inputs))
    #Roots for firing times
    for n in range(spikes):
        t = firing_times[n]
        if n == 0:
            outputs[n] = v(t) - v_th
            u_old = u(t)
        else:
            outputs[n] = v(t, t_old, u_old) - v_th
            u_old = u(t, t_old, u_old)
        t_old = t
    #Roots for being correct distance away
    outputs[-1] = np.dot((inputs - base_estimate), tangent)
    return outputs

def root_for_jacobian(inputs):
    #It's how the scipy docs say to do it
    return np.apply_along_axis(root_function, axis=0, arr=inputs)

##def jacobian_like(vector, tangent, h):
##    assert h > 0
##    global c
##    global firing_times
##    global R
##    c = vector[0] 
##    firing_times_raw = np.zeros(spikes)
##    firing_times = np.zeros(spikes)
##    for n in range(1, spikes):
##        firing_times_raw[n] = vector[n]
##        firing_times[n] = vector[n]
##    R = vector[-1]
##    update_R(R)
##    
##    matrix = np.zeros((spikes+1, spikes+1))
##    #Finite difference (centred) to approximate derivative
##    for col in range(spikes+1):
##        firing_times = firing_times_raw[:]
##        #Set up to calculate the +h part
##        if col == 0:
##            c = vector[0] + h
##        elif col == spikes:
##            R = vector[-1] + h
##            update_R(R)
##        else:
##            firing_times[col] += h
##        #Calculate the +h part
##        for row in range(spikes):
##            t = firing_times[row]
##            if row == 0:
##                matrix[row, col] = v(t)
##                u_old = u(t)
##            else:
##                matrix[row, 0] = v(t, t_old, u_old)
##                u_old = u(t, t_old, u_old)
##            t_old = t
##        #Set up to calculate the -h part
##        if col == 0:
##            c = vector[0] - h
##        elif col == spikes:
##            R = vector[-1] - h
##            update_R(R)
##        else:
##            firing_times[col] -= 2*h
##        #Calculate the -h part
##        for row in range(spikes):
##            t = firing_times[row]
##            if row == 0:
##                matrix[row, col] -= v(t)
##                u_old = u(t)
##            else:
##                matrix[row, 0] -= v(t, t_old, u_old)
##                u_old = u(t, t_old, u_old)
##            matrix[row, 0] /= 2*h
##            t_old = t
##
##    #Then just fill in the tangent
##    row = spikes
##    for col in range(spikes+1):
##        matrix[spikes, col] = tangent[col]
##
##    return matrix

#------------------------------------------------------------------------------

#Used in matrix inversion step
static_vector = np.zeros(spikes+1)
static_vector[-1] = 1

points = []

stopwatch = time.time()

for seed in start_points:
    points.append([])
    print("")
    print(seed)
    print(time.time() - stopwatch)
    guess = seed[0]
    tangent = np.array(seed[1])
    for n in range(steps):
        #NR to find a point on the curve
        base_estimate = guess[:]
        old_soln = guess[:]
        count = 0
        epsilon = base_epsilon
        count_break = 10
        while count < count_break:
            try:
                soln = optimize.root(root_function, guess, tol=rootfinder_tolerance)
            except TypeError:
                count = count_break + 1
                break
            if soln.success:
                #Record solution
                points[-1].append(soln.x)
                #Estimate the tangent at that point
                #This is an approach of using matrix inversion
                try:
                    J_like = differentiate.jacobian(root_for_jacobian, soln.x,
                                                    maxiter=1)
                    z_vector = linalg.solve(J_like.df, static_vector)
                    print("Found z_vector")
                    #Normalise z_vector
                    dot_product = np.dot(z_vector, tangent)
                    sign = abs(dot_product) / dot_product
                    z_vector *= sign / linalg.norm(z_vector, ord = 2)
                    tangent = z_vector[:]
                except np.linalg.LinAlgError:
                    print("LinAlgError for n =", n)
                    #Thus last tangent vector is retained
                except TypeError:
                    print("Out-of-bounds R in Jacobian determination")
                #Create next guess
                old_soln = soln.x[:]
                epsilon = base_epsilon
                guess = old_soln + epsilon * tangent
                break
            else:
                epsilon /= 2
                guess = old_soln + epsilon * tangent
                count += 1
        if count == count_break:
            print("NR failed...")
            print("n =", n)
            
            break
        elif count == count_break + 1:
            print("R gone wrong?")
            print("R =", R)
            print("q = 0.5 *", (D-1)**2 - 4*R, "** 0.5")
            print("n =", n)
            print("epsilon =", epsilon)
            break

print(time.time() - stopwatch)

#Save branches to file
for branch in points:
    filename = "branch_{}.txt".format(str(N))
    N += 1
    outfile = open(filename, "w")
    for point in branch:
        string = ""
        for entry in point:
            string += str(entry) + ","
        string = string[:-1] + "\n"
        outfile.write(string)
    outfile.close()

plt.figure(figsize=(5,5))
#plt.figure(figsize=(4, 4), dpi=800)
for branch in points:
    plot_c = []
    plot_t = []
    plot_R = []
    for point in branch:
        plot_c.append(point[0])
        plot_t.append(point[-2])
        plot_R.append(point[-1])
    #plt.scatter(plot_b, plot_c, s=0.4)
    #plt.plot(plot_b, plot_c, c="#007d69")
    plt.plot(plot_R, plot_c)
    #plt.plot(plot_c, plot_t, c="#007d69")
plt.xlabel("$R$")
plt.ylabel("$c$")
#plt.xlabel("Wave speed $c$")
#plt.ylabel("Time between firing events $T$")
plt.title("Tracking solutions")
#plt.xlim(0,3)
#plt.ylim(0,3)
#plt.savefig("PAC output tuned.png")
plt.show()
