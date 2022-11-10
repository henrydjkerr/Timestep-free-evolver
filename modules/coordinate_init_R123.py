"""
Sets up coordinates for Real-space 1/2/3D.
"""

from numba import cuda

from modules import Control
lookup = Control.lookup

neurons_number = lookup["neurons_number"]
dimension = lookup["dimension"]

neuron_count_x = lookup["neuron_count_x"]
neuron_count_y = lookup["neuron_count_y"]
neuron_count_z = lookup["neuron_count_z"]
dx = lookup["dx"]
dy = lookup["dy"]
dz = lookup["dz"]
even_offset_yx = lookup["even_offset_yx"]
even_offset_zx = lookup["even_offset_zx"]
even_offset_zy = lookup["even_offset_zy"]


@cuda.jit()
def coordinate_init(coordinates_d):
    """
    Initialises supplied coordinate array according to values in parameters.txt.
    Automatically adjusts to 1D, 2D or 3D setups.
    """
    n = cuda.grid(1)
    if n < neurons_number:
        if dimension == 1:
            coordinates_d[n, 0] = (n + 0.5 - neuron_count_x / 2) * dx
        elif dimension == 2:
            x = n % neuron_count_x + (0.5 - neuron_count_x / 2)
            y = n // neuron_count_x + (0.5 - neuron_count_y / 2)
            coordinates_d[n, 0] = x * dx + (1 - y % 2) * even_offset_yx
            coordinates_d[n, 1] = y * dy
        elif dimension == 3:
            x = n % neuron_count_x + (0.5 - neuron_count_x / 2)
            y = (n // neuron_count_x) % neuron_count_y \
                + (0.5 - neuron_count_y / 2)
            z = n // (neuron_count_x * neuron_count_y) \
                + (0.5 - neuron_count_z / 2)
            coordinates_d[n, 0] = x * dx + (1 - y % 2) * even_offset_yx \
                                         + (1 - z % 2) * even_offset_zx
            coordinates_d[n, 1] = y * dy + (1 - z % 2) * even_offset_zy
            coordinates_d[n, 2] = z * dx
