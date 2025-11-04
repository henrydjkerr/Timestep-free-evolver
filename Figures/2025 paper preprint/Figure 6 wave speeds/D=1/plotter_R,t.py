import numpy as np
import matplotlib.pyplot as plt

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

blue1fade = (1 - fade) * blue1 + fade * white
blue2fade = (1 - fade) * blue2 + fade * white
goldfade  = (1 - fade) * gold + fade * white

branches = {
    "slow_stable_pregraze.txt" : { #---
        "colour" : blue2,
        "linestyle" : "solid",
        "graze" : False},
    "slow_stable_postgraze.txt" : {
        "colour" : blue2fade,
        "linestyle" : "solid"},
    "slow_unstable_pregraze.txt" : {
        "colour" : blue2,
        "linestyle" : "dotted",
        "graze" : False},
    "slow_unstable_postgraze.txt" : {
        "colour" : blue2fade,
        "linestyle" : "dotted"},
    "fast_unstable_pregraze.txt" : { #---
        "colour" : blue1,
        "linestyle" : "dotted",
        "graze" : True},
    "fast_unstable_postgraze.txt" : {
        "colour" : blue1fade,
        "linestyle" : "dotted"},
    "fast_unstable_other_pregraze.txt" : {
        "colour" : blue1,
        "linestyle" : "dotted",
        "graze" : True},
    "fast_unstable_other_postgraze.txt" : {
        "colour" : blue1fade,
        "linestyle" : "dotted"},
    "fast_stable_pregraze.txt" : {
        "colour" : blue1,
        "linestyle" : "solid"}
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
    
plt.xlabel("Ion channel response rate $R$")
#plt.xlabel("Wave speed $c$")
#plt.ylabel("Wave speed $c$")
plt.ylabel("Inter-spike time $\\tau_2$")
#plt.title("Two-spike wave solutions, D = 1")
plt.title("Wave solutions for given $R$; $D$ = 1")
plt.xlim(0,4)
#plt.ylim(0,4)
plt.ylim(0, 10)
#plt.legend()
if print_mode:
    plt.savefig("PAC D=1.0 high res vs t.pdf")
    plt.savefig("PAC D=1.0 high res vs t.png")
else:
    plt.show()





