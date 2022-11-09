import numpy
import matplotlib.pyplot as plt
import matplotlib
import data_reader

cmap = matplotlib.colormaps["viridis"]

def make_graph(filename):
    source = data_reader.get(filename)
    max_t = source.data["time"][-1]

    colours = numpy.array([cmap(t/max_t) for t in source.data["time"]])

    plt.figure()
    ax = plt.axes()
    ax.scatter(source.data["coord"][:,0], source.data["coord"][:,1],
                color=colours, marker = ".", edgecolors="none")
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    x_range = int(source.lookup["neuron_count_x"]) * float(source.lookup["dx"])
    y_range = int(source.lookup["neuron_count_y"]) * float(source.lookup["dy"])
    ax.set_xlim(-x_range/2, x_range/2)
    ax.set_ylim(-y_range/2, y_range/2)
    plt.show()

if __name__ == "__main__":
    make_graph("2D_demo_newtype")
