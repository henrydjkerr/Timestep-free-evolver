import numpy as np
import matplotlib.pyplot as plt

print_mode = False
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

blue1fade = (1 - fade) * blue1 + fade * white
blue2fade = (1 - fade) * blue2 + fade * white
goldfade  = (1 - fade) * gold + fade * white

branches = {
    "slow_pregraze.txt" : {
        "colour" : blue2,
        "linestyle" : "solid",
        "graze" : True},
    "slow_postgraze.txt" : {
        "colour" : blue2fade,
        "linestyle" : "solid"},
    "slow_unstable.txt" : {
        "colour" : blue2fade,
        "linestyle" : "dotted"},
    "fast_unstable_pregraze.txt" : {
        "colour" : blue1,
        "linestyle" : "dotted",
        "graze" : True},
    "fast_unstable_postfold.txt" : {
        "colour" : blue1,
        "linestyle" : "dotted"},
    "fast_unstable_postgraze.txt" : {
        "colour" : blue1fade,
        "linestyle" : "dotted"},
    "fast_stable_pregraze.txt" : {
        "colour" : blue1,
        "linestyle" : "solid",
        "graze" : True},
    "fast_stable_postfold.txt" : {
        "colour" : blue1,
        "linestyle" : "solid"},
    "fast_stable_postgraze.txt" : {
        "colour" : blue1fade,
        "linestyle" : "solid"},
    "fast_stable_unstable.txt" : {
        "colour" : blue1fade,
        "linestyle" : "dotted"}
    }

grazes = []

for name in branches:
    infile = open(name)
    plot_t1 = []
    plot_R = []
    for line in infile:
        part = line.strip("\n").split(",")
        plot_t1.append(float(part[1]))
        plot_R.append(float(part[-1]))
    infile.close()
    if "label" in branches[name]:
        plt.plot(plot_R, plot_t1, c = branches[name]["colour"],
                 linestyle = branches[name]["linestyle"],
                 label = branches[name]["label"])
    else:
        plt.plot(plot_R, plot_t1, c = branches[name]["colour"],
                 linestyle = branches[name]["linestyle"])
    if "graze" in branches[name]:
        if branches[name]["graze"]:
            grazes.append([plot_R[0], plot_t1[0], branches[name]["colour"]])
        else:
            grazes.append([plot_R[-1], plot_t1[-1], branches[name]["colour"]])

for graze in grazes:
    plt.scatter(graze[0], graze[1], color = graze[2], zorder = 100)
    
plt.xlabel("Synaptic response rate $\\beta$")
#plt.xlabel("Wave speed $c$")
#plt.ylabel("Wave speed $c$")
plt.ylabel("Inter-spike time $t_1$")
#plt.title("Two-spike wave solutions, D = 1")
plt.title("Wave solutions for given $\\beta$; $R$ = 2.7")
plt.xlim(0,10)
#plt.ylim(0,4)
plt.ylim(0, 5.5)
#plt.legend()
if print_mode:
    plt.savefig("PAC R=2.7 vs t.pdf")
    plt.savefig("PAC R=2.7 vs t.png")
else:
    plt.show()





