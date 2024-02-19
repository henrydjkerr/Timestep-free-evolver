from math import e, erf, pi, cos, sin

beta = 2
A = 2
a = 1
B = 2
b = 2

I = 1.8

C = 2
D = 2

c = 1.02

dx = 0.001
neurons_number = 2000


##is_2D = False
##
###If in 2D:
##if is_2D:
##    A = A * a * ((2*pi)**0.5)
##    B = B * b * ((2*pi)**0.5)

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
#The s calculations are nothing new

def sub_part(t, gamma, maybe_beta):
    coeff = e**(-maybe_beta * t) * e**((maybe_beta * gamma)**2)
    part_erf = 0.5 * (1 + erf((t / (2*gamma)) - maybe_beta * gamma))
    return coeff * part_erf

def part_s(Z, z, t):
    gamma = z / (c * 2**0.5)
    return Z * sub_part(t, gamma, beta)
    
def s(t):
    return beta * (part_s(A, a, t) - part_s(B, b, t))

#-----------------------------------------------------------------------------
#Redoing

def v(c):
    return (I * (2*p - 1) / (p**2 - q2)) \
           + beta * (part_v(A, a, t) - part_v(B, b, t))

def part_v(Z, z, t):
    coeff_cos = (2*p - beta - 1) / ( (p - beta)**2 - q2 )
    coeff_sin = (p**2 + q2 - p*(beta + 1) + beta) / ( (p - beta)**2 - q2 )
    coeff_sin /= abs_q

    part_p = -coeff_cos * calc_integral(z, t, p, "cosine")
    part_p += coeff_sin * calc_integral(z, t, p, "sine")

    part_beta = coeff_cos * calc_integral(z, t, beta, None)
    return Z * (part_p + part_beta)


def u(c):
    return C*I / (p**2 - q2) \
           + C * beta * (part_u(A, a, t) - part_u(B, b, t))

def part_u(Z, z, t):
    coeff_cos = 1 / ( (p - beta)**2 - q2 )
    coeff_sin = (p - beta) / ( (p - beta)**2 - q2 )
    coeff_sin /= abs_q

    part_p = -coeff_cos * calc_integral(z, t, p, "cosine")
    part_p += coeff_sin * calc_integral(z, t, p, "sine")

    part_beta = coeff_cos * calc_integral(z, t, beta, None)
    return Z * (part_p + part_beta)


def s(c):
    return beta * (part_s(A, a, t) - part_s(B, b, t))

def part_s(Z, z, c):
    return Z * calc_integral(z, t, beta, None)

#And then the new integral solver (numerical midpoint method)

def one(x):
    return 1

def calc_integral(z, t, param, func_id):
    if func_id == "sine":
        func = sin
    elif func_id == "cosine":
        func = cos
    else:
        func = one

    #sigma = z/c
    #mu = sigma**2 * param

    length = 5 #3*sigma
    divisions = 500
    dT = length/divisions
    total = 0
    for x in range(divisions):
        T = t - (x+0.5)*dT  #Midpoint method
        value = func(abs_q * (T - t))     \
                * (1 / (z * (2*pi)**0.5)) \
                * e**( -(c**2 / (2*z**2)) * T**2 + param * (T - t))
        total += dT * value
    return total

#-----------------------------------------------------------------------------


v_file = open("v-import-ssLIF.txt", "w")
u_file = open("u-import-ssLIF.txt", "w")
s_file = open("s-import-ssLIF.txt", "w")
line_end = "\n"

for n in range(neurons_number):
    t = -(n - neurons_number//2) * dx / c
    if t <= 0:
        voltage = v(t)
        wigglage = u(t)
        synapse = s(t)
    else:
        voltage = -10
        wigglage = 10
        synapse = s(t)
    if n == neurons_number - 1:
        line_end = ""
    v_file.write(str(voltage) + line_end)
    u_file.write(str(wigglage) + line_end)
    s_file.write(str(synapse) + line_end)
v_file.close()
u_file.close()
s_file.close()
    
