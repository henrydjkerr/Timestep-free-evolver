import numpy as np
import matplotlib.pyplot as plt

blue1 = np.array((0.2, 0.8, 0.9))
blue2 = np.array((0.15, 0.55, 0.7))
blue3 = np.array((0.1, 0.2, 0.3))
gold = np.array((0.85, 0.75, 0.2))
white = np.array((1.0, 1.0, 1.0))
fade = 0.67

print_mode = False
if print_mode:
    plt.figure(figsize=(4, 4), dpi=400)
else:
    plt.figure(figsize=(4,4))

max_view = 3
min_view = -3
count = 301

#Stable/unstable boundary
stable_R = np.linspace(min_view, max_view, count)
stable_D = np.linspace(min_view, max_view, count)
for n in range(count):
    stable_D[n] = max(-1, -stable_R[n])

#Focus/node boundary: 4*R = (D - 1)**2
osc_R = np.linspace(min_view, max_view, count)
osc_D = np.linspace(min_view, max_view, count)
for n in range(count):
    osc_R[n] = 0.25 * (osc_D[n] - 1)**2

plt.xlim(min_view, max_view)
plt.ylim(min_view, max_view)

plt.axhline(0, c = "#999999", linestyle = "dotted")
plt.axvline(0, c = "#999999", linestyle = "dotted")

plt.plot(stable_D, stable_R, c = "#000000")
plt.fill_betweenx(stable_R, stable_D, -4, color = "#dddddd")

plt.plot(osc_D, osc_R, color = blue1)#, linestyle = "dashed")
x_plain = np.linspace(min_view, max_view, count)
plt.fill_between(x_plain[100:], osc_R[100:], -stable_R[100:],
                 color = "#ffffff", hatch = "///",
                 edgecolor = "#dddddd")

plt.xlabel("$D$ (Ion channel activity decay rate)")
plt.ylabel("$R$ (Ion channel response rate)")
plt.title("Single-neuron stability")

##plt.legend(loc="lower right")

plt.tight_layout()
if print_mode:
    plt.savefig("1neuron_stability.pdf")
    plt.savefig("1neuron_stability.png")
else:
    plt.show()





