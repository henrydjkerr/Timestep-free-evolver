import numpy
import matplotlib.pyplot as plt
import matplotlib
import data_reader

colmap = matplotlib.colormaps["viridis"]

def make_graph(filename):
    source = data_reader.get(filename)

    #Make grid approximation
    x_division = int(source.lookup["neuron_count_x"])
    y_division = int(source.lookup["neuron_count_y"])
    x_range = x_division * float(source.lookup["dx"])
    y_range = y_division * float(source.lookup["dy"])
    x_min = -x_range/2
    y_min = -y_range/2
    grid = numpy.zeros((y_division, x_division), dtype=float)
    max_value = 1
    for k in range(len(source.data)):
        x_coord = int(((source.data["coord"][:,0][k] \
                        - x_min) / x_range) * x_division)
        y_coord = int(((source.data["coord"][:,1][k] \
                        - y_min) / y_range) * y_division)
        grid[y_coord][x_coord] += 1
        if grid[y_coord][x_coord] > max_value:
            max_value = grid[y_coord][x_coord]

    for j in range(y_division):
        for k in range(x_division):
            grid[j][k] /= max_value

    plt.figure()
    plt.imshow(grid, cmap=colmap, origin="lower",
               extent=(x_min, x_range + x_min, y_min, y_range + y_min))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

if __name__ == "__main__":
    make_graph("2D_demo_newtype")
