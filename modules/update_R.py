"""
Linearly change the value of R as time goes on, up to some limit.
"""

from modules.general.ParamPlus import lookup

dR = lookup["dR"]
R_target = lookup["R_target"]
#Hack to make sure the original value of R is preserved in the output file
lookup["R_original"] = lookup["R"]

def update(dt):
    lookup["R"] += dt * dR
    if (dR < 0 and lookup["R"] < R_target) or (dR > 0 and lookup["R"] > R_target):
        lookup["R"] = R_target
    

