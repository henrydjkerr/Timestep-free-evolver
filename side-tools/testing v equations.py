import numpy
import matplotlib.pyplot as plt

from math import pi, e, erf

I = 0.9
beta = 1.001
A = 1
a = 3
B = 0
b = 1
Delta = 1
sqrt2 = 2**0.5
c = 7.51

#AaBb, beta = 2
#1101: c =  3.48    All 4 lines agree
#2101: c =  5.35    All 4 lines agree
#1201: c =  6.97    All 4 lines agree
#2201: c = 10.71    All 4 lines agree

#beta = 3
#1101: c =  4.18    All 4 lines agree
#beta = 4
#1101: c =  4.72    All 4 lines agree
#beta = 5
#1101: c =  5.17    All 4 lines agree
#beta = 1.001
#1101: c =  2.50    Surprisingly, all 4 lines agree
#2101: c =  3.82    "
#3101: c =  4.83    "
#4101: c =  5.68    "
#1201: c =  5.00    "
#1301: c =  7.51    "
#At this point, I can probably stop checking with this

def part_v(Z, z, t):
    coeff = Z*z * e**((z*beta / (c * sqrt2))**2)
    part_one = e**((1 - beta)*t) * 0.5 \
               * (1 + erf((c/(z*sqrt2)) * (t - (z**2 * beta / c**2))))
    part_two = e**(0.5*(1 - beta**2) * (z/c)**2) * 0.5 \
               * (1 + erf((c/(z*sqrt2)) * (t - (z**2 / c**2))))
    return coeff * (part_one - part_two)

def v(t):
    value = I + (beta / (1 - beta)) * Delta * e**(-t) * (2*pi)**0.5 * (1/c) \
            * (part_v(A, a, t) - part_v(B, b, t))
    return value

def part_s(Z, z, t):
    coeff = Z*z * e**((z*beta / (c * sqrt2))**2)
    value = 0.5 * (1 + erf((c / (z * sqrt2)) * (t - z**2 * beta / c**2)))
    return coeff * value

def s(t):
    value = beta * Delta * e**(-beta * t) * (2*pi)**0.5 * (1/c) * \
            (part_s(A, a, t) - part_s(B, b, t))
    return value

def part_connect(Z, z, t):
    #value = (Z / ((2*pi)**0.5 * z)) * e**(-0.5 * (c*t / z)**2)
    value = Z * e**(-0.5 * (c*t / z)**2)
    return value

def connect(t):
    return part_connect(A, a, t) - part_connect(B, b, t)

def part_signal(Z, z, d):
    return (Z / (z * (2*pi)**0.5)) * e**(-0.5 * (z/d)**2)

def signal(d):
    return part_signal(A, a, d) - part_signal(B, b, d)

points = 500
lower_bound = -2
step = abs(lower_bound)/points

t_values = numpy.linspace(lower_bound, 0, points)
v_analytic = numpy.zeros(points)
v_numeric = numpy.zeros(points)
v_very_numeric = numpy.zeros(points)
v_REALLY_numeric = numpy.zeros(points)
s_temp = 0
s_temp_2 = 0
for k in range(points):
    #v_analytic
    v_analytic[k] = v(t_values[k])
    
    #v_numeric
    if k == 0:
        v_numeric[k] = I
    else:
        v_numeric[k] = v_numeric[k - 1] \
                       + step * (I - v_numeric[k - 1] + s(t_values[k - 1]))
        
    #v_very_numeric
    if k == 0:
        v_very_numeric[k] = I
    else:
        #s_temp *= e**(-beta * step)
        #s_temp += step * beta * connect(t_values[k])
        s_temp += step * beta * (connect(t_values[k]) - s_temp)
        v_very_numeric[k] = v_very_numeric[k - 1] \
                            + step * (I - v_very_numeric[k - 1] + s_temp)

    #v_REALLY_numeric
    if k == 0:
        v_REALLY_numeric[k] = I
    else:
        input_sum = 0
        count = 100
        width = 10
        for i in range(count):
            #Iterating acro
            y_offset = width*((i + 0.5)/count - 0.5)
            distance = ((c*t_values[k - 1])**2 + y_offset**2)**0.5
            input_sum += beta * (width/count) * signal(distance)
        s_temp_2 += step * beta *(input_sum - s_temp_2)
        v_REALLY_numeric[k] = v_REALLY_numeric[k - 1] \
                              + step * (I - v_REALLY_numeric[k - 1] + s_temp)
    

plt.figure()
plt.plot(t_values, v_analytic)
plt.plot(t_values, v_numeric)
plt.plot(t_values, v_very_numeric)
plt.plot(t_values, v_REALLY_numeric)
plt.axhline(y = 0)
plt.axhline(y = 1)
plt.axvline(x = 0)
plt.xlabel("t (= x/c)")
plt.ylabel("v(t)")
plt.show()
