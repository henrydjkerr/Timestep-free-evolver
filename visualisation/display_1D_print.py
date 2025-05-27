import numpy
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits import mplot3d
import data_reader

#colour = "#9569be"
colour = (0.15, 0.55, 0.7)

def make_graph(filename):
    source = data_reader.get(filename)
    max_t = source.data["time"][-1]

    plt.figure(figsize=(4,4), dpi=400)
    ax = plt.axes()
    ax.scatter(source.data["coord"][:,0], source.data["time"],
               color=colour, s=0.2, marker=".")
    ax.set_xlabel("Neuron position")
    ax.set_ylabel("Time")

    x_range = int(source.lookup["neuron_count_x"]) * float(source.lookup["dx"])
    ax.set_xlim(-x_range/2, x_range/2)
    ax.set_ylim(0, max_t)

    #plt.title("10000 neurons, 20000 spikes, 2D")
    plt.title("Neuron firing times")
    plt.margins(x=0, y=0.01)
    plt.savefig(filename + "_1D_print.png")
    plt.savefig(filename + "_1D_print.pdf")
    #plt.show()

if __name__ == "__main__":
    make_graph("2D_demo_newtype")

