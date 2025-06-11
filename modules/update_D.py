"""
Linearly change the value of D as time goes on, up to some limit.
"""

from modules.general.ParamPlus import lookup

dD = lookup["dD"]
D_target = lookup["D_target"]
#Hack to make sure the original value of D is preserved in the output file
lookup["D_original"] = lookup["D"]

def update(dt):
    lookup["D"] += dt * dD
    if (dD < 0 and lookup["D"] < D_target) or (dD > 0 and lookup["D"] > D_target):
        lookup["D"] = D_target
    

