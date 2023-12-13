from math import e, erf, pi, cos

beta = 2
A = 0.8
a = 1
B = 0.4
b = 2

#Remember to ALSO apply these adjustments to wave speed calculation!
#Use the above values in parameters.txt
A = A * (a * ((2*pi)**0.5))
B = B * (b * ((2*pi)**0.5))

c = 0.8

I = 0.9
dx = 0.1
neurons_number = 100
neurons_y = 100
wiggle_amplitude = -0.1
wiggle_number = 1

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


for y in range(neurons_y):
    epsilon = wiggle_amplitude * cos(wiggle_number * (y/neurons_y - 1/2) * 2*pi)
    for n in range(neurons_number):
        t = -(n - neurons_number//2) * dx / c
        t += epsilon
        if t <= 0:
            voltage = v(t)
            synapse = s(t)
        else:
            voltage = I * (e**(- 3 * n / neurons_number))
            synapse = s(t)
        if (n == neurons_number - 1) and (y == neurons_y - 1):
            line_end = ""
        v_file.write(str(voltage) + line_end)
        s_file.write(str(synapse) + line_end)
v_file.close()
s_file.close()
    
