"""
'Controls' the input current when it's set to be constant for all time.
In other words, it's a dummy function.
"""

from numba import cuda

from modules import Control
lookup = Control.lookup

neurons_number = lookup["neurons_number"]

#-------------------------------------------------------------------------------

def input_control(new_time):
    return False

@cuda.jit()
def input_update(input_strength_d, new_time, time_change):
    pass
