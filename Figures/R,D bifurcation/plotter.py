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
    plt.figure(figsize=(6, 6), dpi=400)
else:
    plt.figure(figsize=(6,6))

branches = {
    "1spike_graze.txt" : {
        "colour" : gold,
        "linestyle" : "solid",
        "label" : "Graze, one-spike solution"},
    "slow_graze.txt" : {
        "colour" : blue2,
        "linestyle" : "solid",
        "label" : "Graze, slow branch"},
    "stable_graze.txt" : {
        "colour" : blue1,
        "linestyle" : "solid",
        "label" : "Graze, fast stable branch"},
    "unstable_graze.txt" : {
        "colour" : blue1,
        "linestyle" : "dotted",
        "label" : "Graze, fast unstable branch"},
    "stable_graze_pt2.txt" : {
        "colour" : blue1,
        "linestyle" : "solid"},
    "unstable_graze_pt2.txt" : {
        "colour" : blue1,
        "linestyle" : "dotted"},
    "fold.txt" : {
        "colour" : "#000000",
        "linestyle" : "dashed",
        "label" : "Fold, fast branch"}
    }

for name in branches:
    infile = open(name)
    plot_R = []
    plot_D = []
    for line in infile:
        part = line.strip("\n").split(",")
        plot_R.append(float(part[-2]))
        plot_D.append(float(part[-1]))
    infile.close()
    if "label" in branches[name]:
        plt.plot(plot_D, plot_R, c = branches[name]["colour"],
                 linestyle = branches[name]["linestyle"],
                 label = branches[name]["label"])
    else:
        plt.plot(plot_D, plot_R, c = branches[name]["colour"],
                 linestyle = branches[name]["linestyle"])


points = 50
base_D = np.linspace(0, 6, points)
base_R = np.zeros(points)
for n in range(points):
    base_R[n] = 0.25 * (base_D[n] - 1)**2
plt.plot(base_D, base_R, color="#999999", linestyle="dashed",
         label="Trigonometric limit")

plt.xlabel("$D$")
plt.ylabel("$R$")
plt.title("Folds and grazes for two-spike wave solutions, $\\beta$ = 6")
plt.xlim(0,4)
plt.ylim(0,8)
plt.legend(loc="lower right")
if print_mode:
    plt.savefig("RD_diagram.pdf")
    plt.savefig("RD_diagram.png")
else:
    plt.show()





