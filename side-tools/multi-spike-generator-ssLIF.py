import numpy as np
import matplotlib.pyplot as plt
from math import e, erf, pi, cos, sin

dx = 0.02
neurons_number = 2000

c, t, C = 0.206395022004725,2.27544859859534,0.413806130543091

beta = 6
A = 2
a = 1
B = 2
b = 2

D = 1


####beta = 6
####A = 2
####a = 1
####B = 2
####b = 2
####
####I = 1.74
####
####C = 2
####D = 2
####
####c = 0.31
####firing_times = [0, 1.12, 2.04, 2.903, 3.74, 4.63, 5.665]
##
##beta = 6
##A = 2
##a = 1
##B = 2
##b = 2
##
##C = 0.31857354
##D = 1
##
##c = 0.197
##
##t = 2.51
##C = 1
##[c, t, beta] = [0.98448429, 2.91757188, 3.20595434]
##
##C = 2
##c = 2.7
##beta = 6
##t = 2.1
##
##
##
###Checking diagram stability
##beta = 6
##D = 1
###Branch 0
##c, t, C = 1.6355594006744427,0.6494444294319965,1.5028341125875362 #Stable
##c, t, C = 1.151497610857992,0.9220566892348047,0.5026268292366598 #Stable
##c, t, C = 0.94605057429821,1.1188822252594435,0.1957607232630025 #Stable
##c, t, C = 1.7641707735007448,0.6017417932434962,1.843552677719024 #Stable
##c, t, C = 1.7489270197141489,0.6070318594445155,1.8013752711555457 #Stable
###Branch 1
##c, t, C = 0.2099064967858099,2.1890596885619917,0.4564747409694321 #Nope
##c, t, C = 0.20491597051836852,2.3006340614732563,0.4025761146540789 #Nope
##c, t, C = 0.18307206447232272,3.260896916143769,0.1007147409171978 #Loses 1
##c, t, C = 0.17892924767384613,3.6899197383003286,0.001871577772208256 #Loses 1
##c, t, C = 0.18878253523317823,2.8728137764105317,0.20104676456517231 #Loses 1
##c, t, C = 0.195851527752025,2.5653918947916696,0.2973134033135687 #Nope
##c, t, C = 0.19195904355769403,2.7186264921271937,0.24672552699627062 #Loses 1
##c, t, C = 0.19029092662174704,2.795618999196856,0.22335759566913052 #Loses 1
##c, t, C = 0.19380573452303032,2.641870814183079,0.27131586481563635 #Loses 1
##c, t, C = 0.19480240253146333,2.6035940312031722,0.28412638388221795 #Nope
###Branch 2
##c, t, C = 0.7616822489872278,3.3242426645188083,1.1553535697127242 #Loses 1
##c, t, C = 0.7449972390820433,3.434563474130204,1.0063684434135414 #Loses 1
##c, t, C = 0.7277116085691531,3.562356533566378,0.831112206634724 #Loses 1
##c, t, C = 0.7243509749647453,3.5899623824032147,0.7907741004282746 #Different
##c, t, C = 0.7214086249063827,3.6155460205680443,0.7515052676112505 #Loses 1
##c, t, C = 0.7147929754069632,3.692393503936773,0.5999379003670868 #Loses 1
##c, t, C = 0.7441647270789213,3.609706082008726,0.38535519131759804 #Loses 1
##c, t, C = 0.7167204092121393,3.7056473048636174,0.520119681971411 #Loses 1
###Should come back to some of that stuff later
###Branch 3
##c, t, C = 1.7788324318965791,6.716629804037007,0.1 #Too wide to fit in
##c, t, C = 1.854641598919716,5.196631307866318,0.2031859378966146 #Still NG
##c, t, C = 2.048855581816334,3.796984062460166,0.5048763646738162 #Fine-ish
##c, t, C = 2.3001770967856516,2.924703921162332,0.9851563159084588 #Fine
##c, t, C = 2.698469392124488,2.1533989103703184,1.992297021029906 #Fine
##c, t, C = 2.896641324335779,1.648568279903316,2.8028942111260324 #Fine
###Branch 4
##c, t, C = 1.1767574882543224,2.4257520859495,0.5 #Loses 1
####c, t, C = 2.07889047387089,1.6573759520331421,1.5069120685942785 #Slowly part
####c, t, C = 2.405444095194004,1.5734023669669235,1.9930580439243404 #Slowly part
####c, t, C = 2.925144182244824,1.6476291758648887,2.862639238900777 #Ditto?
#####Branch 5
####c, t, C = 1.1767574882543224,2.4257520859495,0.5 #Loses 1
####c, t, C = 0.8398652793951251,3.2416568328471387,0.3263799906016608 #Loses 1
####c, t, C = 0.738919029656508,3.630755253858613,0.4000013384569679
##
##c, t, C = 1.9942178507900454,1.6911926566685478,1.3906397793509873
##c, t, C = 2.476625969598888,2.5331687532794542,1.3906397793509873
##
##c, t, C = 2.564903466390308,2.371611994787835,1.6167364443913748
###c, t, C = 2.1622749490296975,1.6290329906190448,1.6254072225067215
###c, t, C = 1.423904483324293,0.7464146459060151,1.0134835713459653
###c, t, C = 0.7214086249063827,3.6155460205680443,0.7515052676112505
###c, t, C = 0.1993424623541326,2.451295588100865,0.3394051211672402
###c, t, C = 0.1855993450510718,3.066498093917295,0.1489643577921857
###c, t, C = 2.9107382886794855,1.7292257991690314,2.7705192934043623
##
##C = 2.7
##c, t, beta = 3.2447080682208482,1.3697630330812054,8.50150162135644
##c, t, beta = 3.5381578702586123,1.3699847775645984,10.0

I = 0.9 * (D + C) / D
print("I =", I)

firing_times = [0, t]

##c = 2.71
##firing_times = [0, 5]
##
### - - -
##c = 0.05
##firing_diff = [1.06, 0.80, 0.71, 0.65, 0.61, 0.61,
##               0.58, 0.56, 0.54, 0.53, 0.53, 0.53,
##               0.52, 0.51, 0.50, 0.49, 0.48, 0.48,
##               0.47, 0.47, 0.47, 0.46, 0.46, 0.46, 
##               0.46, 0.45, 0.45, 0.44, 0.44,
##               0.43, 0.43, 0.42, 0.42, 0.41,
##               0.41, 0.40, 0.40, 0.40, 0.40, 0.39, 
##               0.39, 0.39, 0.39, 0.39, 0.39,
##               0.39, 0.39, 0.39, 0.39, 0.39, 
##               0.39, 0.39, 0.39, 0.39, 0.39,
##               0.40, 0.40, 0.40, 0.40,
##               0.40, 0.40, 0.40, 0.40, 0.40, 
##               0.40, 0.40, 0.40, 0.40, 0.41,
##               0.41, 0.41, 0.41, 0.41, 0.41,
##               0.42, 0.42, 0.42, 0.42, 0.43,
##               0.43, 0.43, 0.43, 
##               0.44, 0.44, 0.45, 0.45, 0.46,
##               0.46, 0.46, 0.46, 0.46, 0.46,
##               0.47, 0.47,
##               0.48, 0.48, 0.49, 0.50, 0.51,
##               0.51, 0.51, 0.51, 
##               0.52, 0.53, 0.53, 0.53, 0.54,
##               0.55, 0.56, 0.58, 0.62, 0.63, 0.64,
##               0.66, 0.68, 0.70, 0.71, 0.75, 0.75]
##
##firing_times = [0]
##for inc in firing_diff:
##    firing_times.append(firing_times[-1] + inc)
### - - -



u_at_firing_times = firing_times[:]

v_r = 0

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
    
#-----------------------------------------------------------------------------

v_file = open("v-import-ssLIF.txt", "w")
u_file = open("u-import-ssLIF.txt", "w")
s_file = open("s-import-ssLIF.txt", "w")
line_end = "\n"

#Finding values of u at firing times
for k, t in enumerate(firing_times):
    if k == 0:
        u_at_firing_times[k] = u(t)
    else:
        u_at_firing_times[k] = u(t, firing_times[k-1], u_at_firing_times[k-1])

for n in range(neurons_number):
    t = -(n - neurons_number//2) * dx / c
    #print(n, t)
    t_last = None
    u_last = None
    for k, firing_time in enumerate(firing_times):
        if firing_time < t:
            t_last = firing_time
            u_last = u_at_firing_times[k]
    voltage = v(t, t_last, u_last)
    wigglage = u(t, t_last, u_last)
    synapse = s(t)
    if n == neurons_number - 1:
        line_end = ""
    v_file.write(str(voltage) + line_end)
    u_file.write(str(wigglage) + line_end)
    s_file.write(str(synapse) + line_end)
v_file.close()
u_file.close()
s_file.close()


