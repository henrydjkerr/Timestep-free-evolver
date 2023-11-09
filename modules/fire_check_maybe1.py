"""
Wrapper that selects between fire_check_is1 and fire_check_not1 on the
basis of the value of synapse_decay.
"""

from numba import cuda

from modules.general import Control
lookup = Control.lookup

synapse_decay = lookup["synapse_decay"]

if abs(synapse_decay - 1) < 0.001:   #Probably shouldn't hardcode this value
    from modules.fire_check_is1 import fire_check
else:
    from modules.fire_check_not1 import fire_check
    
