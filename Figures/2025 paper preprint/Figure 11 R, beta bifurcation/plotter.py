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

branches = {
    "1spike_graze.txt" : {
        "colour" : gold,
        "linestyle" : "solid",
        "label" : "Graze, one-spike solution"},
    "1spike_graze_pt2.txt" : {
        "colour" : gold,
        "linestyle" : "solid"},
    "slow_graze.txt" : {
        "colour" : blue2,
        "linestyle" : "solid",
        "label" : "Graze, slow branch"},
    "slow_graze_pt2.txt" : {
        "colour" : blue2,
        "linestyle" : "solid"},
    "stable_graze.txt" : {
        "colour" : blue1,
        "linestyle" : "solid",
        "label" : "Graze, fast stable branch"},
    "stable_graze_late.txt" : {
        "colour" : blue1,
        "linestyle" : "solid"},
    "unstable_graze.txt" : {
        "colour" : blue1,
        "linestyle" : "dotted",
        "label" : "Graze, fast unstable branch"},
    "unstable_graze_late.txt" : {
        "colour" : blue1,
        "linestyle" : "dotted"},
    "fast_fold.txt" : {
        "colour" : blue1,
        "linestyle" : "dashed",
        "label" : "Folds"},#"label" : "Fold that splits the fast branch"},
    "slow_fold.txt" : {
        "colour" : 0.33 * blue2 + 0.67 * white,
        "linestyle" : "dashed",
        },#"label" : "Fold, slow branch"},
    "stable_fold.txt" : {
        "colour" : 0.33 * blue1 + 0.67 * white,
        "linestyle" : "dashed",
        },#"label" : "Fold, fast stable branch"},
    "unstable_fold.txt" : {
        "colour" : 0.33 * blue1 + 0.67 * white,
        "linestyle" : "dashed",
        },#"label" : "Fold, fast unstable branch"}
    }

plt.axhline(2.5, color = "#ffaaee", lw=0.75, linestyle = "dashed")
plt.axhline(2.6, color = "#ffaaee", lw=0.75, linestyle = "dashed")
plt.axhline(2.7, color = "#ffaaee", lw=0.75, linestyle = "dashed")

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


    
plt.xlabel("$\\beta$ (Synaptic response rate)")
plt.ylabel("$R$ (Ion channel response rate)")
#plt.title("Folds and grazes for two-spike wave solutions, $D$ = 1")
plt.title("Folds and grazes for wave solutions, $D$ = 1")
plt.xlim(0,30)
plt.ylim(0,12.5)
plt.legend(loc="upper right")
if print_mode:
    plt.savefig("Rbeta_diagram.pdf")
    plt.savefig("Rbeta_diagram.png")
else:
    plt.show()





