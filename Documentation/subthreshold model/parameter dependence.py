import numpy
import matplotlib.pyplot as plt

from math import e, erf, pi, cos, sin, factorial, log
import cmath

#Shouldn't be 1 for this program, just use 1.0001 or similar
beta = 6
A = 2
a = 1
B = 2
b = 2

C = 0.0001
D = 1

I = 0.90

is_2D = False
target = 1    #v_th

#If in 2D:
if is_2D:
    A = A * a * ((2*pi)**0.5)
    B = B * b * ((2*pi)**0.5)

#Derived values
def pq_stuff(C, D):
    p = 0.5*(D+1)
    q = 0.5 * cmath.sqrt( (D-1)**2 -4*C )
    if type(q) != type((-1)**0.5):
        print("q is", q)
        print("q should be imaginary for this to work")
        raise TypeError
    abs_q = abs(q)
    q2 = -abs_q**2
    return p, abs_q, q2


#-----------------------------------------------------------------------------
#Redoing equations entirely

def v_alt(c):
    return (I * (2*p - 1) / (p**2 - q2)) \
           + part_v_alt(A, a, c) - part_v_alt(B, b, c)

def part_v_alt(Z, z, c):
    coeff_cos = (2*p - beta - 1) / ( (p - beta)**2 - q2 )
    coeff_sin = (p**2 + q2 - p*(beta + 1) + beta) / ( (p - beta)**2 - q2 )
    coeff_sin /= abs_q

    part_p = -coeff_cos * calc_integral(z, c, p, "cosine")
    part_p += coeff_sin * calc_integral(z, c, p, "sine")

    part_beta = coeff_cos * calc_integral(z, c, beta, None)
    return beta * Z * (part_p + part_beta)

def one(x):
    return 1

def calc_integral(z, c, param, func_id):
    if func_id == "sine":
        func = sin
    elif func_id == "cosine":
        func = cos
    else:
        func = one

    sigma = z/c
    mu = sigma**2 * param

    length = 5 #3*sigma
    divisions = 5000
    dT = length/divisions
    total = 0
    t = 0
    for x in range(divisions):
        T = - (x+0.5)*dT  #Midpoint method
##        value = func(abs_q * (T - t))     \
##                * (1 / (z * (2*pi)**0.5)) \
##                * e**( -(c**2 / (2*z**2)) * T**2 + param * (T - t))
        value = func(abs_q * T) * (c / (z * (2*pi)**0.5)) \
                 * e**(-(c**2 / (2*z**2)) * (T + t)**2 + param * T)
        total += dT * value
    return total
    
#------------------------------------------------------------------------------

#Assume the function is decreasing
#This is probably going to be a bit of a hack for now

def get_c():
    c = 1
    step = 0.5
    margin = 0.00001
    max_iter = 50
    for count in range(max_iter):
        test = v_alt(c)
        if test > target:
            if test - target < margin:
                break
            c += step
        else:
            step *= 0.5
            c -= step

    if count == max_iter - 1:
        print("Operation timed out")
        return None
    print(count, c)
    return c

mode = "beta"

if mode == "AB":
    points = 40
    s_lim = [1.3, 3.9]
    ss_lim = [1.7, 3]
elif mode == "ab":
    points = 2
    s_lim = [0.1, 10]
    ss_lim = [0.1, 10]
elif mode == "b":
    points = 40
    s_lim = [1.57, 4.5]
    ss_lim = [1.43, 2.75]
elif mode == "beta":
    points = 40
    s_lim = [1.4, 50]
    ss_lim = [4, 35]

xs_values = numpy.linspace(s_lim[0], s_lim[1], points)
xss_values = numpy.linspace(ss_lim[0], ss_lim[1], points)
s_values = numpy.zeros(points)
ss_values = numpy.zeros(points)

if mode == "ab":
    s_factor = 1.6942138671875
    ss_factor = 2.7125732421875
    for n in range(points):
        s_values[n] = s_factor * xs_values[n]
        ss_values[n] = ss_factor * xss_values[n]

if mode != "ab":    
    C = 0.0001
    I = 0.9 * (C + D) / D
    p, abs_q, q2 = pq_stuff(C, D)
    for n, x in enumerate(xs_values):
        if mode == "AB":
            A = x
            B = x
        elif mode == "b":
            b = x
        elif mode == "beta":
            beta = x
        s_values[n] = get_c()

    C = 2
    I = 0.9 * (C + D) / D
    p, abs_q, q2 = pq_stuff(C, D)
    for n, x in enumerate(xss_values):
        if mode == "AB":
            A = x
            B = x
        elif mode == "b":
            b = x
        elif mode == "beta":
            beta = x
        ss_values[n] = get_c()


plt.figure(figsize=(5, 3))
if mode == "AB":
    plt.xlabel("$A$, $B$")
    plt.title("Scaling $A$ and $B$, with $A = B$")
elif mode == "ab":
    plt.xlabel("$a$, $0.5b$")
    plt.title("Scaling $a$ and $b$, with $2a = b$")
elif mode == "b":
    plt.title("Scaling $b$")
    plt.xlabel("$b$")
elif mode == "beta":
    plt.xlabel(r"$\beta$")
    plt.title(r"Scaling $\beta$")
#plt.xlabel("C, with I(C+D)/D")

plt.ylabel("$c$")

if mode == "b":
    ss_values[0] = 1.2
plt.plot(xs_values, s_values, c="#000000")
plt.plot(xss_values, ss_values, c="#9569be")

if mode != "ab":
    for n, y in enumerate(s_values):
        if not numpy.isnan(y):
            plt.scatter(xs_values[n], y,
                        s = 50, marker = "x", c = "#000000")
            print(xs_values[n], y)
            break
if mode == "AB" or mode == "b":
    plt.scatter(xs_values[points-1], s_values[points-1],
                s = 50, marker="o", c="#000000")

if mode != "ab":
    for n, y in enumerate(ss_values):
        if not numpy.isnan(y):
            mark = "x"
            if mode == "beta":
                mark = "o"
            plt.scatter(xss_values[n], y,
                        s = 50, marker = mark, c = "#9569be")
            print(xss_values[n], y)
            break
if mode != "ab":
    plt.scatter(xss_values[points-1], ss_values[points-1],
                s = 50, marker="o", c="#9569be")

plt.tight_layout()


##points = 200
##x_values = numpy.linspace(c - 0.2, c + 0.2, points)
##x_values = numpy.linspace(0.2, 5, points)
##y_values = numpy.zeros(points)
##for key in range(points):
##    #y_values[key] = v(x_values[key])
##    y_values[key] = v_alt(x_values[key])



#plt.plot(x_values, l_values)
#plt.axhline(y = target, c="#000000", linestyle="dotted")

#plt.title("Speed finder, $(A, a, B, b)$ = ({}, {}, {}, {}), $\\beta$ = {}".format(
#    A, a, B, b, beta))

plt.show()
        
