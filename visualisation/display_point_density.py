import numpy
import matplotlib.pyplot as plt
import matplotlib
import data_reader

def make_graph(filename):
    source = data_reader.get(filename)

    #Density heuristic
    #Haven't actually come up with one
    weight = 0.05

    plt.figure()
    ax = plt.axes()
    plt.scatter(source.data["coord"][:,0], source.data["coord"][:,1],
            marker = ".", color=(1, 0, 0, weight), edgecolors="none")

    x_range = int(source.lookup["neuron_count_x"]) * float(source.lookup["dx"])
    y_range = int(source.lookup["neuron_count_y"]) * float(source.lookup["dy"])
    ax.set_xlim(-x_range/2, x_range/2)
    ax.set_ylim(-y_range/2, y_range/2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.show()



if __name__ == "__main__":
    make_graph("2D_demo_newtype")
