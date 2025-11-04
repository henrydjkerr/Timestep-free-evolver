import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from math import e, pi
from numpy import sin, cos, log, arctan


R = 2
D = 1
omega = 1

def update_RD(new_R, new_D):
    global R
    global D
    global I
    global p
    global q
    global abs_q
    global q2
    global lamb1
    global lamb2

    R = new_R
    D = new_D
    p = 0.5*(D+1)
    q = 0.5 * np.emath.sqrt( (D-1)**2 -4*R )
    if type(q) != type(np.emath.sqrt(-1)):
        #print("q is", q)
        #print("q should be imaginary for this to work")
        raise TypeError
    abs_q = abs(q)
    q2 = -abs_q**2

    lamb1 = -p -q
    lamb2 = -p +q

update_RD(R, D)



#------------------------------------------------------------------------------

def derived(func, lamb1, lamb2, coeff1, coeff2):
    return derived_part(func, lamb1, coeff1) - derived_part(func, lamb2, coeff2)

def derived_part(func, lamb, coeff):
    return (coeff / (lamb**2 + omega**2)**0.5) * func(arctan(lamb/omega))

#------------------------------------------------------------------------------



points = 200
R_start = 1
R_length = 19
R_values = np.linspace(R_start, R_start + R_length, points)
D_start = 1
D_length = 7
D_values = np.linspace(D_start, D_start + D_length, points)

results = np.zeros((points, points))

o_points = 600
o_start = 0
o_length = 6
omega_values = np.linspace(o_start, o_start + o_length, o_points)

for y in range(points):
    for x in range(points):
        results[y, x] = None

for y in range(points):
    for x in range(points):
        forced_freq = 0
        try:
            update_RD(R_values[y], D_values[x])
            #Get natural frequency
            nat_freq = abs_q
            #Find resonant forced frequency
            top_amp = 0
            for o_point in range(o_points):
                omega = omega_values[o_point]
                V_1 = derived(cos, lamb1, lamb2, lamb2 + 1, lamb1 + 1)
                V_2 = -derived(sin, lamb1, lamb2, lamb2 + 1, lamb1 + 1)
                #U_1 = -derived(cos, lamb1, lamb2, 1, 1)
                #U_2 = derived(sin, lamb1, lamb2, 1, 1)
                v_amp = 0.5 * ((V_1**2 + V_2**2)/q2)**0.5
                if v_amp > top_amp:
                    top_amp = v_amp
                    forced_freq = omega
                else:
                    break
        except TypeError:
            pass
        if forced_freq == 0:
            freq = None
            break
        else:
            #freq = forced_freq
            freq = forced_freq / abs_q
        results[y, x] = freq

level_select = np.arange(0.5, 5.00, 0.5)
#level_select = np.arange(0.1, 1.25, 0.1)

fig, ax = plt.subplots(figsize=(5,4), dpi=200)

if False:
    img = ax.contour(D_values, R_values, results,
                     level_select,
                     cmap="PiYG",
                     norm=matplotlib.colors.CenteredNorm(vcenter=1))
else:
    img = ax.imshow(results, origin="lower",
                    extent=(D_start, D_start + D_length,
                            R_start, R_start + R_length),
                    aspect = D_length/R_length, cmap="PiYG",
                    norm=matplotlib.colors.CenteredNorm(vcenter=1))
fig.colorbar(img, ax=ax)

boundary = np.zeros(points)
for n in range(points):
    boundary[n] = 0.25 * (D_values[n]-1)**2
plt.plot(D_values, boundary, color="#999999", linestyle="dashed",
         label = "$4R = (D - 1)^2$")

plt.xlabel("$D$ (Ion channel activity decay rate)")
plt.ylabel("$R$ (Ion channel response rate)")
#plt.title("Resonant frequencies")
plt.legend(loc = "lower right")
plt.title("Resonant frequencies as $ω/|q|$")
plt.xlim(D_start, D_start + D_length)
plt.ylim(R_start, R_start + R_length)
plt.show()






