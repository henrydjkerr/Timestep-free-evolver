import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

print_mode = True
if print_mode:
    plt.figure(figsize=(4,4), dpi=400)
else:
    plt.figure(figsize=(4,4))
blue1 = np.array((0.2, 0.8, 0.9))
blue2 = np.array((0.15, 0.55, 0.7))
blue3 = np.array((0.1, 0.2, 0.3))
gold = np.array((0.85, 0.75, 0.2))
white = np.array((1.0, 1.0, 1.0))
fade = 0.67

green = np.array((0.5, 0.8, 0.2))
purple = np.array((0.9, 0.3, 0.55))

green = np.array((0.5, 0.7, 0.1))
purple = np.array((0.9, 0.5, 0.75))

branches = {
    "1spike_graze.txt" : {
        "colour" : gold,
        "linestyle" : "solid",
        "label" : "Graze, 1-spike wave"},
    "1spike_graze_pt2.txt" : {
        "colour" : gold,
        "linestyle" : "solid"},
    "1spike_graze_pt2_unstable.txt" : {
        "colour" : gold,
        "linestyle" : "dotted"},
    "2spike_graze.txt" : {
        "colour" : blue2,
        "linestyle" : "solid",
        "label" : "Graze, 2-spike wave"},
    "2spike_graze_pt2.txt" : {
        "colour" : blue2,
        "linestyle" : "solid"},
    "2spike_graze_pt2_unstable.txt" : {
        "colour" : blue2,
        "linestyle" : "dotted"},
    "3spike_graze.txt" : {
        "colour" : purple,
        "linestyle" : "solid",
        "label" : "Graze, 3-spike wave"},
    "3spike_graze_pt2.txt" : {
        "colour" : purple,
        "linestyle" : "solid"},
    "3spike_graze_pt2_unstable.txt" : {
        "colour" : purple,
        "linestyle" : "dotted"},
    "4spike_graze.txt" : {
        "colour" : green,
        "linestyle" : "solid",
        "label" : "Graze, 4-spike wave"},
    "4spike_graze_unstable.txt" : {
        "colour" : green,
        "linestyle" : "dotted"},
    }

for name in branches:
    infile = open(name)
    plot_R = []
    plot_beta = []
    for line in infile:
        part = line.strip("\n").split(",")
        plot_R.append(float(part[-2]))
        plot_beta.append(float(part[-1]))
    infile.close()
    if "label" in branches[name]:
        plt.plot(plot_beta, plot_R, c = branches[name]["colour"],
                 linestyle = branches[name]["linestyle"],
                 label = branches[name]["label"])
    else:
        plt.plot(plot_beta, plot_R, c = branches[name]["colour"],
                 linestyle = branches[name]["linestyle"])


#plt.tight_layout()
plt.xlabel("$\\beta$ (Synaptic response rate)")
#plt.xlabel("Wave speed $c$")
plt.ylabel("$R$ (Ion channel response rate)")
#plt.ylabel("Inter-spike time $t_1$")
#plt.title("Two-spike wave solutions, D = 1")
#plt.title("Wave solutions for given $R$; $D$ = 1")
plt.title("Grazes for $m$-spike wave solutions, $D$ = 1")
plt.xlim(0,150)
plt.ylim(0,8.5)
plt.legend(loc = "lower right")

if print_mode:
    plt.savefig("Rbeta_mspike.pdf")
    plt.savefig("Rbeta_mspike.png")
else:
    plt.show()





