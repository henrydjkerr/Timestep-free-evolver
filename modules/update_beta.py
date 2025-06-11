"""
Linearly change the value of beta (synapse_decay) as time goes on, up to some limit.
"""

from modules.general.ParamPlus import lookup

dbeta = lookup["dbeta"]
beta_target = lookup["beta_target"]
#Hack to make sure the original value of synapse_decay is preserved in the output file
lookup["synapse_decay_original"] = lookup["synapse_decay"]

def update(dt):
    lookup["synapse_decay"] += dt * dbeta
    if (dbeta < 0 and lookup["synapse_decay"] < beta_target):
        lookup["synapse_decay"] = beta_target
    if (dbeta > 0 and lookup["synapse_decay"] > beta_target):
        lookup["synapse_decay"] = beta_target
    

