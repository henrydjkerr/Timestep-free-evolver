import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from math import e, erf, pi, cos, sin, log
from scipy import optimize
from scipy import linalg

beta = 6
A = 2
a = 1
B = 2
b = 2

C = 3
D = 1
I_raw = 0.9

v_th = 1
v_r = 0

C_min = 0.001
C_max = 3.001
C_steps = 30
x_count = 6
y_count = 12

def update_C(new_C):
    global C
    global I
    global p
    global q2
    global abs_q

    #C = C_min + (C_max - C_min) * (k/C_steps)
    C = new_C
    I = I_raw * (C + D) / D
    #Derived values
    p = 0.5*(D+1)
    interior = (D-1)**2 -4*C
    if interior <= 0:
        abs_q = 0.5 * (-interior)**0.5
        q2 = -abs_q**2
    else:
        print("q is 0.5 *", interior, "** 0.5")
        print("q should be imaginary for this to work")
        raise TypeError
    q2 = -abs_q**2
    return

update_C(C)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#General integration function

def one(x):
    return 1

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
    if lower == None:
        #lower = upper - 3*sigma
        lower = upper - 5
    else:
        #lower = max(upper - 3*sigma, lower)
        lower = max(upper - 5, lower)
    divisions = 100
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
    
    part_I = C * I / (p**2 - q2)
    part_init = 0
    if t_old != None:
        part_I *= 1 - e**(-p * (t - t_old)) * cos(abs_q * (t - t_old))
        part_I -= C * I * (p / ((p**2 - q2) * abs_q))   \
                  * e**(-p * (t - t_old)) * sin(abs_q * (t - t_old))
        part_init = e**(-p * (t - t_old))       \
                    * (u_old * cos(abs_q * (t - t_old))
                       + ((C * v_r + u_old * (1 - p)) / abs_q)
                          * sin(abs_q * (t - t_old)))
    part_s = s(t) * C * coeff_cos

    part_t_old = 0
    if t_old != None:
        part_t_old = s(t_old) * e**(-p * (t - t_old))       \
                     * C * (  coeff_cos * cos(abs_q * (t - t_old))
                            + coeff_sin * sin(abs_q * (t - t_old)))
##
##    print("u_old", u_old)
##    print("part_I", part_I)
##    print("part_init", part_init)
##    print("part_s", part_s)
##    print("part_t_old", part_t_old)
##    print("part_A", part_u(A, a, t, t_old))
##    print("part_B", part_u(B, b, t, t_old))
##        
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
    return -C * beta * Z * (part_cos + part_sin)
    

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


#def root_function(c_try, t_1_try):
def root_function(inputs):
    global c
    global firing_times
    global C
    global base_estimate
    global tangent
    
    c = inputs[0]
    firing_times = np.zeros(spikes)
    for n in range(1, spikes):
        firing_times[n] = inputs[n]
    C = inputs[-1]
    update_C(C)
    
    outputs = np.zeros(spikes + 1)
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


def jacobian_like(vector, tangent, h):
    assert h > 0
    global c
    global firing_times
    global C
    c = vector[0] 
    firing_times_raw = np.zeros(spikes)
    firing_times = np.zeros(spikes)
    for n in range(1, spikes):
        firing_times_raw[n] = vector[n]
        firing_times[n] = vector[n]
    C = vector[-1]
    update_C(C)
    
    matrix = np.zeros((spikes+1, spikes+1))
    #Finite difference (centred) to approximate derivative
    #This is real hacky but a consequence of how it's currently set up
    for col in range(spikes+1):
        firing_times = firing_times_raw[:]
        #Set up to calculate the +h part
        if col == 0:
            c = vector[0] + h
        elif col == spikes:
            C = vector[-1] + h
            update_C(C)
        else:
            firing_times[col] += h
        #Calculate the +h part
        for row in range(spikes):
            t = firing_times[row]
            if row == 0:
                matrix[row, col] = v(t)
                u_old = u(t)
            else:
                matrix[row, 0] = v(t, t_old, u_old)
                u_old = u(t, t_old, u_old)
            t_old = t
        #Set up to calculate the -h part
        if col == 0:
            c = vector[0] - h
        elif col == spikes:
            C = vector[-1] - h
            update_C(C)
        else:
            firing_times[col] -= 2*h
        #Calculate the -h part
        for row in range(spikes):
            t = firing_times[row]
            if row == 0:
                matrix[row, col] -= v(t)
                u_old = u(t)
            else:
                matrix[row, 0] -= v(t, t_old, u_old)
                u_old = u(t, t_old, u_old)
            matrix[row, 0] /= 2*h
            t_old = t

    #Then just fill in the tangent
    row = spikes
    for col in range(spikes+1):
        matrix[spikes, col] = tangent[col]

    return matrix

#------------------------------------------------------------------------------

spikes = 2

#[c, t_1, R]
start_points = [
    ([2, 0.5, 3], [0, 0, -1]),  #Middle branch, starts where it says
    ([0.5, 1, 3], [0, 0, -1]),  #Bottom branch, goes all the way to zero
    ([1, 2.2, 3], [0, 0, -1]),  #Starts lower right, connects to bend
    ([1.7, 4, 0.1], [0, 0, 1]), #Top branch
    ([3, 2, 3], [0, 0, -1]),    #Also top branch, but other end
    ([2, 2, 0.5], [0, 0, 1]),   #Goes around the curve... or used to
    ([2, 2, 0.5], [0, 0, -1]),  #Doesn't help much, terminates quickly
    ([0.5, 7, 0.1], [0, -1, 0]),    #Aiming for "unstable" split wave branch
    ##([2, 7, 0.1], [0, -1, 0]),  #Aiming for "stable" split wave branch (top b.)
    ]

##start_points = [
##    ([0.7, 1.5, 0.1], [0, 0, -1])
##    ]

static_vector = np.zeros(spikes+1)
static_vector[-1] = 1
#Make up an initial "tangent"
tangent_original = np.zeros(spikes+1)
tangent_original[-1] = 1

base_epsilon = 0.05
points = []

for seed in start_points:
    print("")
    print(seed)
    guess = seed[0]
    tangent = seed[1]
    #tangent[-1] = seed[1]
    #tangent = tangent_original[:] * seed[1]
    #print("tangent", tangent)
    #print("guess", guess)
    #print("static_vector", static_vector)
    
    for n in range(60):
        #NR to find a point on the curve
        base_estimate = guess[:]
        #if n == 0: print("base_estimate", base_estimate)

        count = 0
        epsilon = base_epsilon
        count_break = 10
        while count < count_break:
            try:
                soln = optimize.root(root_function, guess, tol=0.0001)
                #print(base_estimate)
            except TypeError:
                count = count_break + 1
                break
            if soln.success:
                #if n == 0: print("soln.x", soln.x)
                #if n == 0: print("tangent", tangent)
                #Record solution
                points.append(soln.x)
                #Estimate the tangent at that point
                J_like = jacobian_like(soln.x, tangent, 0.0001)
                #if n == 0: print("J_like", J_like)
                #if n == 0: print("static_vector", static_vector)
                inv_J = linalg.inv(J_like)
                z_vector = np.dot(inv_J, static_vector)
                #Normalise z_vector
                dot_product = np.dot(z_vector, tangent)
                #if n == 0: print("dot_product", dot_product)
                sign = abs(dot_product) / dot_product 
                #z_vector *= sign / linalg.norm(z_vector, ord = np.inf)
                                                    #Works better than ord = 2?
                z_vector *= sign / linalg.norm(z_vector, ord = 2)
                #if n == 0: print("z_vector", z_vector)
                tangent = z_vector[:]
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
            print("C gone wrong?")
            print("C =", C)
            print("q = 0.5 *", (D-1)**2 - 4*C, "** 0.5")
            print("n =", n)
            print("epsilon =", epsilon)
            break

#for point in points:
#    print(point)

plot_c = []
plot_t = []
plot_R = []
for point in points:
    plot_c.append(point[0])
    plot_t.append(point[1])
    plot_R.append(point[-1])

##plt.figure()
##plt.scatter(plot_R, plot_c, s=0.4)
##plt.xlabel("$R$")
##plt.ylabel("Wave speed $c$")
##plt.title("Wave solutions by speed for given $R$ (formerly $C$)")
##plt.xlim(0,3)
##plt.ylim(0,3)
##plt.show()


plt.figure()
ax = plt.axes(projection = "3d")
ax.scatter3D(plot_R, plot_c, plot_t, s=0.4)
ax.set_xlabel("$R$")
ax.set_ylabel("Wave speed $c$")
ax.set_zlabel("Second firing time")
plt.title("Wave solutions by speed for given $R$ (formerly $C$)")
ax.set_xlim(0,3)
ax.set_ylim(0,3)
plt.show()





