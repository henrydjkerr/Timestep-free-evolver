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

fade = 0.67
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
max_spikes = 16

#Load in graze point data
graze_list_c = []
graze_list_R = []
infile = open("grazes.txt")
for line in infile:
    part = line.strip("\n").split(",")
    graze_list_c.append(float(part[0]))
    graze_list_R.append(float(part[-1]))

#Load in solution curve data
spike_list = []
for spike in range(1, max_spikes+1):
    infile = open(str(spike) + "spike.txt")
    spike_list.append([[],[]])
    n = 0
    for line in infile:
        part = line.strip("\n").split(",")
        if spike == 1:
            spike_list[-1][0].append(float(part[0]))
        else:
            try:
                #Express their speed relative to the 1-spike wave speed
                spike_list[-1][0].append(float(part[0]) / spike_list[0][0][n])
            except IndexError:
                #Keeps everything no longer than the 1-spike array
                break
        spike_list[-1][1].append(round(float(part[-1]) / precision) * precision)
        n += 1
    infile.close()

#Plotting curves
#1-spike baseline
thiscol = getcol(1)
plt.hlines(1, 0, graze_list_R[0], color = thiscol)
plt.hlines(1, graze_list_R[0], 6, color = fade(thiscol))
plt.scatter(graze_list_R[0], 1, color = thiscol, zorder = 100)
for spike in range(1, max_spikes):
    #Format data, to split between pre/post-graze
    graze_R = graze_list_R[spike]
    graze_index = bisect(spike_list[spike][1], graze_R)
    #Interpolate the relative c value of the graze
    c_before = spike_list[spike][0][graze_index]
    R_before = spike_list[spike][1][graze_index]
    c_after = spike_list[spike][0][graze_index + 1]
    R_after = spike_list[spike][1][graze_index + 1]
    graze_c = c_before + (c_after - c_before) * (graze_R - R_before) / (R_after - R_before)    
    pre_graze_c = spike_list[spike][0][:graze_index]
    pre_graze_R = spike_list[spike][1][:graze_index]
    post_graze_c = spike_list[spike][0][graze_index:]
    post_graze_R = spike_list[spike][1][graze_index:]
    #Make sure they join in the middle
    pre_graze_c.append(graze_c)
    pre_graze_R.append(graze_R)
    post_graze_c.insert(0, graze_c)
    post_graze_R.insert(0, graze_R)
    #Plot curves
    thiscol = getcol(spike + 1)
    plt.plot(pre_graze_R, pre_graze_c, color = thiscol)
    plt.plot(post_graze_R, post_graze_c, color = fade(thiscol))
    plt.scatter(graze_R, graze_c, color = thiscol, zorder = 100)


plt.xlabel("$R$ (Ion channel response rate)")
plt.ylabel("Difference in $c$")
plt.title("Wave speeds relative to 1-spike wave, $D$ = 1")
plt.xlim(0,6)
#plt.ylim(0,4)
#plt.yticks([0, -1, -2, -3])
if print_mode:
    plt.savefig("relative speed.pdf")
    plt.savefig("relative speed.png")
else:
    plt.show()





