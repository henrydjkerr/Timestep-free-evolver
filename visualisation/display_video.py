import numpy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import data_reader

#With thanks to https://matplotlib.org/stable/gallery/animation/rain.html
# as a model example

#------------------------------------------------------------------------------

def get_opacity(frame, t):
    """
    Determines how faded a firing event's point should be.

    If the firing event has not yet occurred, it is invisible.
    If it has just occurred it is at maximum visibility.
    It then fades linearly back to invisibility over a set time period.
    """
    frame_t = frame * frame_length
    event_t = t * total_length / max_t
    if event_t > frame_t:
        return 0
    else:
        return max([0, 1 - (frame_t - event_t) / point_duration])

def update(i):
    """Alters the colours for each point per frame."""
    for k in range(len(source.data)):
        colours[k, 3] = get_opacity(i, source.data["time"][k])
    plot.set_color(colours)

#------------------------------------------------------------------------------

def make_graph(filename):
    global source
    source = data_reader.get(filename)
    global max_t
    max_t = source.data["time"][-1]
    global frame_length
    global total_length
    global point_duration
    frame_length = 50
    total_length = len(source.data) * 2
    point_duration = 1 * 1000 #How long before a given point fades completely
    frame_count = total_length // frame_length

    global colours
    colours = numpy.array([[1.0, 0.0, 0.0, 0.0] for d in source.data])

    global fig
    fig = plt.figure()
    global plot
    plot = plt.scatter(source.data["coord"][:,0], source.data["coord"][:,1],
                       c=colours, marker=".")
    plt.margins(x=0.001, y=0.001)
    animation = FuncAnimation(fig, update, frames=frame_count,
                              interval=frame_length)
    animation.save(filename + ".gif")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

if __name__ == "__main__":
    make_graph("2D_demo_newtype")
