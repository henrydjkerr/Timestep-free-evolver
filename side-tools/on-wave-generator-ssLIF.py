from math import e, erf, pi, cos, sin

beta = 4
A = 2
a = 1
B = 2
b = 2

I = 3.0

C = 2
D = 2

c = 7.36

dx = 0.04
neurons_number = 500


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

def part_v(Z, z, t):
    coeff_cos = (2*p - beta - 1) / ((p - beta**2) - q2)
    coeff_sin = (p**2 + q2 - p*(beta + 1) + beta) / ((p - beta**2) - q2)
    coeff_sin /= abs_q
    
    part_p =  coeff_cos * calc_integral(z, t, "cosine")
    part_p += coeff_sin * calc_integral(z, t, "sine")
    part_p *= e**(-p * t) * e**((z*p/c)**2 / 2)
    #part_p *= e**((z*p/c)**2 / 2)
    
    part_beta = 0.5 * (1 + erf((c/(z*2**0.5)*t) - ((z*beta)/(c*2**0.5))))
    part_beta *= e**(-beta * t) * e**((z*beta/c)**2 / 2) * coeff_cos
    #part_beta *= e**((z*beta/c)**2 / 2) * coeff_cos
    return Z * beta * (part_p + part_beta)

def v(t):
    return (I / (p**2 - q2)) + part_v(A, a, t) - part_v(B, b, t)

def part_u(Z, z, t):
    coeff_sin = (beta - p) / abs_q
    coeff_total = C * beta * Z / ((p - beta**2) - q2)
    
    part_p = calc_integral(z, t, "cosine") \
             + coeff_sin * calc_integral(z, t, "sine")
    part_p *= e**(-p * t) * e**((z*p/c)**2 / 2)

    part_beta = 0.5 * (1 + erf(-(z * beta)/(c * 2**0.5))) \
                * e**(-beta * t) * e**((z*beta/c)**2 / 2)
    return coeff_total * (part_p + part_beta)
    
def u(t):
    return (-C*I / (p**2 - q2)) + part_u(A, a, t) - part_u(B, b, t)

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
#Replacement part since the fancy integration method likes to diverge

def calc_integral(z, t, cos_or_sin):
    sigma = z/c
    mu = sigma**2 * p - t

    length = 5
    divisions = 500
    dT = length/divisions
    total = 0
    for x in range(divisions):
        T = -dT * x
        if cos_or_sin == "sine":
            value = sin(abs_q * T)
        elif cos_or_sin == "cosine":
            value = cos(abs_q * T)
        value *= (1 / (sigma * (2*pi)**0.5)) * e**(-((T - mu)/sigma)**2 / 2)
        total += dT * value
    return total

def calc_integral(z, t, cos_or_sin):
    sigma = z/c
    mu = sigma**2 * p - t

    length = 5
    divisions = 500
    dT = length/divisions
    total = 0
    for x in range(divisions):
        T = -dT * x
        if cos_or_sin == "sine":
            value = sin(abs_q * T)
        elif cos_or_sin == "cosine":
            value = cos(abs_q * T)
        value *= (1 / (sigma * (2*pi)**0.5)) * e**(-((T - mu)/sigma)**2 / 2)
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
        voltage = 0
        wigglage = 0
        synapse = s(t)
    if n == neurons_number - 1:
        line_end = ""
    v_file.write(str(voltage) + line_end)
    s_file.write(str(synapse) + line_end)
v_file.close()
s_file.close()
    
