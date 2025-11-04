import numpy
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits import mplot3d
import data_reader

#dotcolour = "#9569be"
dotcolour = "#000000"
linecolour = "#dd3333"
bifcolour = "#8811aa"

def make_graph(filename):
    source = data_reader.get(filename)
    max_t = source.data["time"][-1]

    plt.figure(figsize=(2.5,4), dpi=400)
    ax = plt.axes()
##    ax.scatter(source.data["coord"][:,0], source.data["time"],
##               color=dotcolour, s=0.1, marker=".")
    ax.set_xlabel("Neuron position")
    ax.set_ylabel("Time")

    beta_start = float(source.lookup["R_original"])
    beta_end = float(source.lookup["R"])
    beta_bif = float(source.lookup["bifurcation"])
    dR = float(source.lookup["dR"])
    ax.axhline((beta_end - beta_start) / dR, color = linecolour)
    ax.axhline((beta_bif - beta_start) / dR, color = bifcolour,
               linestyle = "dashed")

    x_range = int(source.lookup["neuron_count_x"]) * float(source.lookup["dx"])
    ax.set_xlim(-x_range/2, x_range/2)
    ax.set_ylim(0, max_t)
    #ax.set_ylim(0, 80)

    #plt.title("10000 neurons, 20000 spikes, 2D")
    plt.title("Neuron firing times")
    plt.margins(x=0, y=0.01)
##    plt.savefig(filename + "_print.png")
    plt.savefig(filename + "_print.pdf")

make_graph("output-20250615193533-4000N-50000sp")

