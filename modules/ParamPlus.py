"""
Pulls through the dictionary of parameters from Param, but generates
some derived parameters along the way so that everything has easy
access to them.

Also gives a chance to do input validation.
"""

if __name__.split(".")[0] == "modules":
    from modules.Param import lookup
else:
    from Param import lookup


##if __name__ == "__main__":
##    from Param import lookup
##else:
##    from modules.Param import lookup

#Type checking
assert type(lookup["neuron_count_x"]) == int
assert type(lookup["neuron_count_y"]) == int
assert type(lookup["neuron_count_z"]) == int
assert type(lookup["dx"]) in (int, float)
assert type(lookup["dy"]) in (int, float)
assert type(lookup["dz"]) in (int, float)
assert lookup["dx"] > 0
assert lookup["dy"] > 0
assert lookup["dz"] > 0
assert type(lookup["even_offset_yx"]) in (int, float)
assert type(lookup["even_offset_zx"]) in (int, float)
assert type(lookup["even_offset_zy"]) in (int, float)

assert type(lookup["threads"]) == int
if lookup["threads"] < 1: lookup[threads] = 1
assert type(lookup["spikes_sought"]) == int
if lookup["spikes_sought"] < 1: lookup["spikes_sought"] = 1

assert type(lookup["v_r"]) in (int, float)
assert type(lookup["v_th"]) in (int, float)
assert type(lookup["synapse_decay"]) in (int, float)
assert lookup["synapse_decay"] > 0
##assert type(lookup["sigma"]) in (int, float)
##assert lookup["sigma"] > 0
##assert type(lookup["c_sigma_1"]) in (int, float)
##assert lookup["c_sigma_1"] > 0
##assert type(lookup["c_sigma_2"]) in (int, float)
##assert lookup["c_sigma_2"] > 0
assert type(lookup["leniency_threshold"]) in (int, float)
assert lookup["leniency_threshold"] >= 0
assert type(lookup["error_bound"]) in (int, float)
assert lookup["error_bound"] > 0

#I might have to remove this stuff or put it elsewhere since I'm trying to
# make things more generic.


#-------------------------------------------------------------------------------

#Ensure each dimension is positive
for key in ("neuron_count_x", "neuron_count_y", "neuron_count_z"):
    if lookup[key] < 1:
        lookup[key] = 1

#Calculating the appropriate dimension and number of neurons used
dimension = 0
neurons_number = 1
for key in ("neuron_count_x", "neuron_count_y", "neuron_count_z"):
    if lookup[key] > 1:
        dimension += 1
        neurons_number *= lookup[key]
    else:
        break
if neurons_number == 1:
    dimension = 1

lookup["neurons_number"] = neurons_number
lookup["dimension"] = dimension

#Calculating the number of blocks needed from the number of threads per block
threads = lookup["threads"]
lookup["blocks"] = int((neurons_number + threads - 1) / threads)


#Checking each offset is non-negative
for key in ("even_offset_yx", "even_offset_zx", "even_offset_zy"):
    if lookup[key] < 0:
        lookup[key] *= -1


if __name__ == "__main__":
    for key in lookup:
        print(key, "=", lookup[key])
