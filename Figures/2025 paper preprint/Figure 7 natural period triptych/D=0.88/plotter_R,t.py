import numpy as np
import matplotlib.pyplot as plt
from math import pi

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
    "1st_branch_stable.txt" : { #--- 1
        "colour" : blue1,
        "linestyle" : "solid"},
    "1st_branch_unstable_pregraze.txt" : {
        "colour" : blue1,
        "linestyle" : "dotted",
        "graze" : False},
    "1st_branch_unstable_postgraze.txt" : {
        "colour" : blue1fade,
        "linestyle" : "dotted"},
    "2nd_branch_pregraze.txt" : { #--- 2
        "colour" : blue1,
        "linestyle" : "dotted",
        "graze" : False},
    "2nd_branch_postgraze.txt" : {
        "colour" : blue1fade,
        "linestyle" : "dotted"},
    "3rd_branch_prepregraze.txt" : { #--- 3
        "colour" : blue1,
        "linestyle" : "solid"},
    "3rd_branch_pregraze.txt" : {
        "colour" : blue1,
        "linestyle" : "solid",
        "graze" : False},
    "3rd_branch_postgraze.txt" : {
        "colour" : blue1fade,
        "linestyle" : "solid"},
    "4th_branch_pregraze.txt" : { #--- 4
        "colour" : blue1,
        "linestyle" : "dotted",
        "graze" : False},
    "4th_branch_postgraze.txt" : {
        "colour" : blue1fade,
        "linestyle" : "dotted"},
    "5th_branch_prepregraze.txt" : { #--- 5
        "colour" : blue1,
        "linestyle" : "solid"},
    "5th_branch_pregraze.txt" : {
        "colour" : blue1,
        "linestyle" : "solid",
        "graze" : False},
    "5th_branch_postgraze.txt" : {
        "colour" : blue1fade,
        "linestyle" : "solid"},
    "6th_branch_pregraze.txt" : { #--- 6
        "colour" : blue1,
        "linestyle" : "dotted",
        "graze" : True},
    "6th_branch_postgraze.txt" : {
        "colour" : blue1fade,
        "linestyle" : "dotted"},
    "r_1st_branch_stable.txt" : { #---
        "colour" : blue1fade,
        "linestyle" : "solid"},
    "r_1st_branch_unstable.txt" : {
        "colour" : blue1fade,
        "linestyle" : "dotted"},
    "r_2nd_branch_stable.txt" : { #---
        "colour" : blue1fade,
        "linestyle" : "solid"},
    "r_2nd_branch_unstable.txt" : {
        "colour" : blue1fade,
        "linestyle" : "dotted"},
    "r_3rd_branch_stable.txt" : { #---
        "colour" : blue1fade,
        "linestyle" : "solid"},
    "r_3rd_branch_unstable.txt" : {
        "colour" : blue1fade,
        "linestyle" : "dotted"},
    "r_4th_branch_stable.txt" : { #---
        "colour" : blue1fade,
        "linestyle" : "solid"},
    "r_4th_branch_unstable.txt" : {
        "colour" : blue1fade,
        "linestyle" : "dotted"},
    "r_5th_branch_stable.txt" : { #---
        "colour" : blue1fade,
        "linestyle" : "solid"},
    "r_5th_branch_unstable.txt" : {
        "colour" : blue1fade,
        "linestyle" : "dotted"},
    "r_6th_branch_stable.txt" : { #---
        "colour" : blue1fade,
        "linestyle" : "solid"},
    "r_6th_branch_unstable.txt" : {
        "colour" : blue1fade,
        "linestyle" : "dotted"},
    }

#Trying out some resonance comparison
#Natural frequency
points = 100
D = 0.88
base_R = np.linspace(0.08, 15, points)
period = np.zeros(points)
for n in range(points):
    R = base_R[n]
    q2 = (D - 1)**2 - 4*R
    abs_q = (-q2)**0.5
    period[n] = 2*pi / abs_q
for n in range(1, 14):
    plt.plot(base_R, n * period, color="#000000", linestyle="dashed",
             label="Natural period", lw=0.8)

#Forced frequency
##in_R = []
##in_f = []
##infile = open("resonant_frequency.txt")
##for line in infile:
##    entries = line.strip("\n").split(",")
##    in_R.append(float(entries[0]))
##    in_f.append(2*pi / float(entries[1]))
##infile.close()
##plt.plot(in_R, in_f, color="#999999", linestyle="dashed",
##         label="Forced period")


#Plotting branches
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
##    if "label" in branches[name]:
##        plt.plot(plot_R, plot_t1, c = branches[name]["colour"],
##                 linestyle = branches[name]["linestyle"],
##                 label = branches[name]["label"])
##    else:
    plt.plot(plot_R, plot_t1, c = branches[name]["colour"],
             linestyle = branches[name]["linestyle"], lw=2)
    if "graze" in branches[name]:
        if branches[name]["graze"]:
            grazes.append([plot_R[0], plot_t1[0], branches[name]["colour"]])
        else:
            grazes.append([plot_R[-1], plot_t1[-1], branches[name]["colour"]])

for graze in grazes:
    plt.scatter(graze[0], graze[1], color = graze[2], zorder = 100)


    
plt.xlabel("$R$ (Ion channel response rate)")
#plt.xlabel("Wave speed $c$")
#plt.ylabel("Wave speed $c$")
plt.ylabel("$\\tau_2$ (Inter-spike time)")
#plt.title("Two-spike wave solutions, D = 1")
plt.title("Wave solutions compared to natural period")
#plt.xlim(0,14)
#plt.ylim(0, 17)
plt.xlim(4,6)
plt.ylim(2,6)
##plt.legend(loc = "upper left")
if print_mode:
    plt.savefig("2spike_natural_frequency_D={}.pdf".format(str(D)))
    plt.savefig("2spike_natural_frequency_D={}.png".format(D))
else:
    #plt.savefig("inset.pdf")
    plt.show()





