"""
'Shuts off' the input current *after* a specified time.
To be exact, it shuts it off at the time of the first firing event
after the specified time, and it can set it to a non-zero constant
value instead of zero.
"""

from numba import cuda

from modules import Control
lookup = Control.lookup

neurons_number = lookup["neurons_number"]
input_shutoff = lookup["input_shutoff"]
input_base_after = lookup["input_base_after"]


#-------------------------------------------------------------------------------

input_active = True

def input_control(new_time):
    global input_active
    if (new_time > input_shutoff) and input_active:
        return True
    else:
        return False

##def input_control(new_time):
##    return False

@cuda.jit()
def input_update(input_strength_d, new_time, time_change):
    n = cuda.grid(1)
    if n < neurons_number:
        input_strength_d[n] = input_base_after
    
