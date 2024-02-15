import numpy
import matplotlib.pyplot as plt

from math import e, erf, pi, cos, sin, factorial

#Shouldn't be 1 for this program, just use 1.0001 or similar
beta = 2
A = 2
a = 1
B = 2
b = 2

C = 2
D = 2

I = 1.8

is_2D = False
target = 1    #v_th

#If in 2D:
if is_2D:
    A = A * a * ((2*pi)**0.5)
    B = B * b * ((2*pi)**0.5)

#Derived values
p = 0.5*(D+1)
q = 0.5*( (D-1)**2 -4*C )**0.5
if type(q) != type((-1)**0.5):
    print("q is", q)
    print("q should be imaginary for this to work")
    raise TypeError
abs_q = abs(q)
q2 = -abs_q**2


#-----------------------------------------------------------------------------
#The follows is based on formula 8.40 and thereon in running review - edit

##def new_m(m_minus_1, m_minus_2, mu, sigma, k, n, gaussian):
##    #Get next value in recursive scheme for the integrals
##    #Eqn 8.48
##    value = m_minus_1 * mu * ((k - 1)/k) \
##            + m_minus_2 * sigma**2 * (n - 1) \
##            - (-mu/k)**(n-1) * sigma**2 * gaussian
##    return value

##def calc_integral(z, c, cos_or_sin):
##    sigma = z/c
##    mu = sigma**2 * p
##    gaussian = (1 / (sigma * (2*pi)**0.5)) * e**((-mu**2)/(2*sigma**2))
##    if cos_or_sin == "sine":
##        derivs = [sin(mu), cos(mu), -sin(mu), -cos(mu)]
##    elif cos_or_sin == "cosine":
##        derivs = [cos(mu), -sin(mu), -cos(mu), sin(mu)]
##    value = 0
##    m = []
##    for n in range(40):
##        #See Eqn 8.44
##        if n == 0:
##            m.append(0.5*(1 + erf(-mu/(sigma*2**0.5))))
##        elif n == 1:
##            m.append(new_m(m[0], 0, mu, sigma, abs_q, n, gaussian))
##        else:
##            m.append(new_m(m[-1], m[-2], mu, sigma, abs_q, n, gaussian))
##        value += (derivs[n%4] * abs_q**n / factorial(n)) * m[-1]
##    return value

##def part_v(Z, z, c):
##    coeff_cos = (2*p - beta - 1) / ((p - beta**2) - q2)
##    coeff_sin = (p**2 + q2 - p*(beta + 1) + beta) / ((p - beta**2) - q2)
##    coeff_sin /= abs_q
##    
##    part_p =  coeff_cos * calc_integral(z, c, "cosine")
##    part_p += coeff_sin * calc_integral(z, c, "sine")
##    part_p *= e**((z*p/c)**2 / 2)
##    
##    part_beta = 0.5 * (1 + erf(-(z * beta)/(c * 2**0.5)))
##    part_beta *= e**((z*beta/c)**2 / 2) * coeff_cos
##    return Z * beta * (part_p + part_beta)
##
##def v(c):
##    return (I * (2*p - 1) / (p**2 - q2)) + part_v(A, a, c) - part_v(B, b, c)
##
##def part_u(Z, z, c):
##    coeff_sin = (beta - p) / abs_q
##    coeff_total = C * beta * Z / ((p - beta**2) - q2)
##    
##    part_p = calc_integral(z, c, "cosine") \
##             + coeff_sin * calc_integral(z, c, "sine")
##    part_p *= e**((z*p/c)**2 / 2)
##
##    part_beta = 0.5 * (1 + erf(-(z * beta)/(c * 2**0.5))) \
##                * e**((z*beta/c)**2 / 2)
##    return coeff_total * (part_p + part_beta)
##    
##def u(c):
##    return (-C*I / (p**2 - q2)) + part_u(A, a, c) - part_u(B, b, c)

#------------------------------------------------------------------------------
#Testing the convergence/divergence effects of the integration method

##def calc_integral_alt(z, c, cos_or_sin):
##    sigma = z/c
##    mu = sigma**2 * p
##
##    length = 5
##    divisions = 500
##    dT = length/divisions
##    total = 0
##    for x in range(divisions):
##        T = -dT * x
##        if cos_or_sin == "sine":
##            value = sin(abs_q * T)
##        elif cos_or_sin == "cosine":
##            value = cos(abs_q * T)
##        value *= (1 / (sigma * (2*pi)**0.5)) * e**(-((T - mu)/sigma)**2 / 2)
##        total += dT * value
##    return total
##
##def part_v_alt(Z, z, c):
##    coeff_cos = (2*p - beta - 1) / ((p - beta**2) - q2)
##    coeff_sin = (p**2 + q2 - p*(beta + 1) + beta) / ((p - beta**2) - q2)
##    coeff_sin /= abs_q
##    
##    part_p =  coeff_cos * calc_integral_alt(z, c, "cosine")
##    part_p += coeff_sin * calc_integral_alt(z, c, "sine")
##    part_p *= e**((z*p/c)**2 / 2)
##    
##    part_beta = 0.5 * (1 + erf(-(z * beta)/(c * 2**0.5)))
##    part_beta *= e**((z*beta/c)**2 / 2) * coeff_cos
##    return Z * beta * (part_p + part_beta)
##
##def v_alt(c):
##    return (I * (2*p - 1) / (p**2 - q2)) \
##           + part_v_alt(A, a, c) - part_v_alt(B, b, c)

#------------------------------------------------------------------------------
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
    divisions = 500
    dT = length/divisions
    total = 0
    t = 0
    for x in range(divisions):
        T = t - (x+0.5)*dT  #Midpoint method
        value = func(abs_q * (T - t))     \
                * (1 / (z * (2*pi)**0.5)) \
                * e**( -(c**2 / (2*a**2)) * T**2 + param * (T - t))
        total += dT * value
    return total
    
#------------------------------------------------------------------------------

#Assume the function is decreasing
#This is probably going to be a bit of a hack for now
c = 1
step = 0.5
margin = 0.00001
for count in range(100):
    test = v_alt(c)
    if test > target:
        if test - target < margin:
            break
        c += step
    else:
        step *= 0.5
        c -= step

if count == 99:
    print("Operation timed out")
print(count)
print(c)

points = 200
x_values = numpy.linspace(c - 0.2, c + 0.2, points)
x_values = numpy.linspace(0.5, 10, points)
y_values = numpy.zeros(points)
y_alt_values = numpy.zeros(points)  #"Manual" version
for key in range(points):
    #y_values[key] = v(x_values[key])
    y_alt_values[key] = v_alt(x_values[key])



        

plt.figure()
#plt.plot(x_values, y_values, c="#dd0000")
plt.plot(x_values, y_alt_values, c="#00dddd")
plt.axhline(y = target, c="#000000", linestyle="dotted")

#plt.title("Speed finder, $(A, a, B, b)$ = ({}, {}, {}, {}), $\\beta$ = {}".format(
#    A, a, B, b, beta))
plt.title("Placeholder title")
plt.xlabel("Speed $c$")
plt.ylabel("$v_{th}$")
plt.show()
        
