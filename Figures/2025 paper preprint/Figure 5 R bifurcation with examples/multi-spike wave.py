import numpy as np
import matplotlib.pyplot as plt
from math import e, erf, pi, cos, sin, ceil

v_rest = 0.9

A = 2
a = 1
B = 2
b = 2

beta = 6
R = 3
D = 1

example = 4
match example:
    case 1:
        c, t, R = 2.69362964354985,2.158687482521653,2.0
        xmin = -2.5
        xmax = 5
        tickstep = 2
    case 2:
        c, t, R = 2.4026914673993933,1.5756989411145332,2.0
        xmin = -2.5
        xmax = 5
        tickstep = 2
    case 3:
        c, t, R = 1.42639261429302,0.744605815516075,1.02418171720662
        xmin = -2.5
        xmax = 5
        tickstep = 2
    case 4:
        c, t, R = 0.74951470275535,3.41352164974942,1.02940536528184
        xmin = -4
        xmax = 10
        tickstep = 4
    case 5:
        c, t, R = 0.188744175975244,2.89519819431064,0.194375021448766
        xmin = -2.5
        xmax = 9
        tickstep = 2
    case 6:
        c, t, R = 0.206395022004725,2.27544859859534,0.413806130543091
        xmin = -2.5
        xmax = 9
        tickstep = 2
        

firing_times = [0, t]

u_at_firing_times = firing_times[:]

v_r = 0


I = v_rest * (R + D) / D
#Derived values
p = 0.5*(D+1)
q = 0.5*( (D-1)**2 -4*R )**0.5
if type(q) != type((-1)**0.5):
    print("q is", q)
    print("q should be imaginary for this to work")
    raise TypeError
abs_q = abs(q)
q2 = -abs_q**2

#-----------------------------------------------------------------------------
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
    return -R * beta * Z * (part_cos + part_sin)
    
#-----------------------------------------------------------------------------

steps = 501
t_values = np.linspace(xmin, xmax, steps)
#t_values = np.linspace(2.4, 2.7, steps)
#voltage = np.zeros(steps)
#wigglage = np.zeros(steps)
#synapse = np.zeros(steps)
t_list = []
v_list = []
u_list = []
s_list = []

#Finding values of u at firing times
for k, t in enumerate(firing_times):
    if k == 0:
        u_at_firing_times[k] = u(t)
    else:
        u_at_firing_times[k] = u(t, firing_times[k-1], u_at_firing_times[k-1])

print(v(0))
if len(firing_times) > 1:
    for n in range(len(firing_times) - 1):
        print(v(firing_times[n+1], firing_times[n], u_at_firing_times[n]))



fire_index = -1
t_last = None
u_last = None
for n in range(steps):
    t = t_values[n]
    for k, firing_time in enumerate(firing_times):
        if firing_time < t:
            if fire_index < k:
                fire_index = k
                t_last = firing_time
                u_last = u_at_firing_times[k]
                #Break the voltage line
                t_list.append(t)
                v_list.append(None)
                u_list.append(u(t, t_last, u_last))
                s_list.append(s(t))
    t_list.append(t)
    v_list.append(v(t, t_last, u_last))
    u_list.append(u(t, t_last, u_last))
    s_list.append(s(t))
    #voltage[n] = v(t, t_last, u_last)
    #wigglage[n] = u(t, t_last, u_last)
    #synapse[n] = s(t)


#plt.figure(figsize=(6,4), dpi=200)
#plt.figure(figsize=(5, 3.5))
plt.figure(figsize=(5, 3.5))
#plt.figure(figsize=(5, 4), dpi=800)
plt.axhline(0, linestyle="dotted", c="#999999", label="$v_\\text{r}$")
plt.axhline(1, linestyle="dashed", c="#999999", label="$v_\\text{th}$")
plt.axvline(0, c="#999999")

#vcol = "#007d69"
#ucol = "#9569be"
#scol = "#ff7f41"

vcol = (0.0, 0.1, 0.3)
ucol = (0.1, 0.5, 0.1)
scol = (0.9, 0.5, 0.75)

plt.plot(t_list, s_list, c=scol, label="$s$")
plt.plot(t_list, u_list, c=ucol, label="$u$")
plt.plot(t_list, v_list, c=vcol, label="$v$")

plt.gca().invert_xaxis()

#plt.title("""\n
#Travelling wave profile for two-spike wave, $c$ = {}
#""".format(str(c)[:4]))
plt.title("""\n
Travelling wave profile, $c$ = {}
""".format(str(c)[:4]))
plt.xlabel("Co-moving coordinate $t - x/c$")
plt.ylabel("Value")

##thismax = 0
##for entry in v_list:
##    if entry != None:
##        if entry > thismax:
##            thismax = entry
##print(thismax)
##plt.xlim((2.4, 2.7))
##plt.ylim((0.9, 1.1))
#plt.ylim((-0.5, 2))
plt.xticks(np.arange(ceil(xmin), xmax+0.1, tickstep))

#plt.tight_layout()
#plt.ylim(-0.8, 3.4)
plt.margins(x=0)
if example == 1:
    plt.legend(loc = "lower left", reverse = True)
plt.savefig("profile_{}.png".format(example))
plt.savefig("profile_{}.pdf".format(example))
plt.show()
#plt.tight_layout()
#plt.savefig("profile_R={}.png".format(str(R)[:4]))

