from math import e, erf

beta = 2
A = 2
a = 1
B = 2
b = 2

c = 0.8

I = 0.9
dx = 0.04
neurons_number = 500

#-----------------------------------------------------------------------------

def sub_part(t, gamma, maybe_beta):
    coeff = e**(-maybe_beta * t) * e**((maybe_beta * gamma)**2)
    part_erf = 0.5 * (1 + erf((t / (2*gamma)) - maybe_beta * gamma))
    return coeff * part_erf

def part_v(Z, z, t):
    gamma = z / (c * 2**0.5)
    return Z * (sub_part(t, gamma, beta) - sub_part(t, gamma, 1))

def v(t):
    value = I + (beta / (1 - beta)) * (part_v(A, a, t) - part_v(B, b, t))
    return value

def part_s(Z, z, t):
    gamma = z / (c * 2**0.5)
    return Z * sub_part(t, gamma, beta)
    
def s(t):
    return beta * (part_s(A, a, t) - part_s(B, b, t))

#-----------------------------------------------------------------------------

v_file = open("v-import.txt", "w")
s_file = open("s-import.txt", "w")
line_end = "\n"

for n in range(neurons_number):
    t = -(n - neurons_number//2) * dx / c
    if t <= 0:
        voltage = v(t)
        synapse = s(t)
    else:
        voltage = 0
        synapse = s(t)
    if n == neurons_number - 1:
        line_end = ""
    v_file.write(str(voltage) + line_end)
    s_file.write(str(synapse) + line_end)
v_file.close()
s_file.close()
    
