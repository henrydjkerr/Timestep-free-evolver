import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect

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
purple = np.array((0.9, 0.5, 0.75))
green = np.array((0.5, 0.7, 0.1))

def fade(colour):
    return 0.33 * colour + 0.67 * white

def getcol(n):
    if n == 1:
        return gold
    elif n == 2:
        return blue2
    elif n == 3:
        return purple
    elif n == 4:
        return green
    else:
        return blue3
    
grazes = []
precision = 0.05
max_spikes = 10

#Load in graze point data
graze_list_c = []
graze_list_R = []
infile = open("grazes.txt")
for line in infile:
    part = line.strip("\n").split(",")
    graze_list_c.append(float(part[0]))
    graze_list_R.append(float(part[-1]))

#Load in Hopf bifurcation data
hopf_list = []
infile = open("hopf.txt")
for line in infile:
    part = line.strip("\n")
    hopf_list.append(int(part) - 1)

#Load in solution curve data
spike_list = []
for spike in range(1, max_spikes+1):
    infile = open(str(spike) + "spike.txt")
    spike_list.append([[],[]])
    for line in infile:
        part = line.strip("\n").split(",")
        spike_list[-1][0].append(float(part[0]))
        spike_list[-1][1].append(round(float(part[-1]) / precision) * precision)
    infile.close()

#Plotting curves
for spike in range(0, max_spikes):
    #Format data, to split between pre/post-graze
    graze_c = graze_list_c[spike]
    graze_R = graze_list_R[spike]
    graze_index = bisect(spike_list[spike][1], graze_R)
    pre_graze_c = spike_list[spike][0][:graze_index]
    pre_graze_R = spike_list[spike][1][:graze_index]
    post_graze_c = spike_list[spike][0][graze_index:]
    post_graze_R = spike_list[spike][1][graze_index:]
    #Make sure they join in the middle
    pre_graze_c.append(graze_c)
    pre_graze_R.append(graze_R)
    post_graze_c.insert(0, graze_c)
    post_graze_R.insert(0, graze_R)
    #Then split off the pre-Hopf part too
    hopf_flag = 0
    if spike < len(hopf_list):
        hopf_index = hopf_list[spike]
        if hopf_index > 0:
            if hopf_index < graze_index:
                hopf_flag = 1
                unstable_c = pre_graze_c[:hopf_index + 1]
                unstable_R = pre_graze_R[:hopf_index + 1]
                stable_c = pre_graze_c[hopf_index :]
                stable_R = pre_graze_R[hopf_index :]
            else:
                hopf_flag = 2
                new_index = hopf_index - graze_index + 1
                unstable_c = post_graze_c[:new_index + 1]
                unstable_R = post_graze_R[:new_index + 1]
                stable_c = post_graze_c[new_index:]
                stable_R = post_graze_R[new_index:]              
    #Plot curves
    thiscol = getcol(spike + 1)
    if hopf_flag == 0:
        plt.plot(pre_graze_R, pre_graze_c, color = thiscol)
        plt.plot(post_graze_R, post_graze_c, color = fade(thiscol))
        plt.scatter(graze_R, graze_c, color = thiscol, zorder = 100)
    elif hopf_flag == 1:
        plt.plot(unstable_R, unstable_c, color = thiscol, linestyle = "dotted")
        plt.plot(stable_R, stable_c, color = thiscol)
        plt.plot(post_graze_R, post_graze_c, color = fade(thiscol))
        plt.scatter(graze_R, graze_c, color = thiscol, zorder = 100)
    elif hopf_flag == 2:
        plt.plot(pre_graze_R, pre_graze_c, color = thiscol, linestyle = "dotted")
        plt.plot(unstable_R, unstable_c, color = fade(thiscol), linestyle = "dotted")
        plt.plot(stable_R, stable_c, color = fade(thiscol))
        plt.scatter(graze_R, graze_c, color = thiscol, zorder = 100)
        

    
#plt.plot(spike_list[-1][1], spike_list[-1][0])

plt.xlabel("$R$ (Ion channel response rate)")
#plt.xlabel("Wave speed $c$")
plt.ylabel("$c$ (Wave speed)")
#plt.ylabel("Inter-spike time $t_1$")
#plt.title("Two-spike wave solutions, D = 1")
plt.title("Wave solutions for given $R$; $D$ = 1")
plt.xlim(0,6)
plt.ylim(0,4)
plt.yticks([0, 1, 2, 3, 4])
#plt.legend()
if print_mode:
    plt.savefig("speed.pdf")
    plt.savefig("speed.png")
else:
    plt.show()





