"""
Constantly recalculates the input current to balance changes in the parameters
R and D.  Assumes a constant input across the network.
"""

from numba import cuda

from modules.general import Control
lookup = Control.lookup

neurons_number = lookup["neurons_number"]
input_shutoff = lookup["input_shutoff"]
input_base_after = lookup["input_base_after"]

blocks = lookup["blocks"]
threads = lookup["threads"]

raw_I = lookup["input_base_before"]
R = lookup["R"]
D = lookup["D"]
base_I = raw_I * D / (R + D)


#-------------------------------------------------------------------------------

input_active = True

def input_control(new_time):
    return True

def input_update(arrays, new_time, time_change):
    input_strength_d = arrays["input_strength"]
    new_R = lookup["R"]
    new_D = lookup["D"]
    new_I = base_I * (new_R + new_D) / new_D
    #print(new_I)
    input_update_device[blocks, threads](input_strength_d, new_I)

@cuda.jit()
def input_update_device(input_strength_d, new_I):
    n = cuda.grid(1)
    if n < neurons_number:
        input_strength_d[n] = new_I
    
