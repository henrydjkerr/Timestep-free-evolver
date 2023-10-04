"""
'Controls' the input current when it's set to be constant for all time.
In other words, it's a dummy function.
"""

from numba import cuda

from modules.general import Control
lookup = Control.lookup

neurons_number = lookup["neurons_number"]

#-------------------------------------------------------------------------------

def input_control(new_time):
    return False

def input_update(arrays, new_time, time_change):
    pass
