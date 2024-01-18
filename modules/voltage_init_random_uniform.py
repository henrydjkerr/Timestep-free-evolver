"""
Initialises the voltage array on device with random independent values
between v_r and v_th.
Reliant on setting up the host-side voltage array with random values in
the first place, so it's not doing the heavy lifting.
"""


from numba import cuda
#from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

from modules.general import Control
lookup = Control.lookup

neurons_number = lookup["neurons_number"]
v_r = lookup["v_r"]
v_th = lookup["v_th"]

#rng_array = create_xoroshiro128p_states(neurons_number, seed = 1)

@cuda.jit()
def array_init(voltage_d, coordinates_d):
    """Set voltage array to randomly distribute between v_r and v_th."""
    n = cuda.grid(1)
    if n < neurons_number:
        #unif = xoroshiro128p_uniform_float32(rng_d, n)
        voltage_d[n] = v_r + 0.5 * (v_th - v_r) * voltage_d[n]
