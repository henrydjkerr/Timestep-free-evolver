"""
A common access point for all the custom runtime-selected modules.
Also used as the common access point for parameters by importing ParamPlus.
"""

import importlib

from modules.general import locator
from modules.general.ParamPlus import lookup

#------------------------------------------------------------------------------

filename = locator.location["modules"]
raw_data = locator.file_reader(filename)

names = {}
for line in raw_data:
    key = line[0]
    try:
        value = line[1].strip()
    except ValueError:
        pass
    names[key] = "modules." + value


#------------------------------------------------------------------------------

#Might want to rethink this way of binding        
d = importlib.import_module(names["distance"])
c = importlib.import_module(names["connection_weight"])
v = importlib.import_module(names["v_calcs"])
i = importlib.import_module(names["i_update"])

vi = importlib.import_module(names["v_init"])
ii = importlib.import_module(names["i_init"])
xi = importlib.import_module(names["coord_init"])

check = importlib.import_module(names["fire_check"])
solve = importlib.import_module(names["root_finder"])
clean = importlib.import_module(names["cleanup"])

#Really this should be some sort of loop
references = {"voltage": "v_init",
              #"synapse": "s_init",     #Don't actually have this set up yet
              "coordinates": "coord_init",
              "input_strength": "i_init"}

blocks = lookup["blocks"]
threads = lookup["threads"]

def fill_arrays(arrays):
    temp = importlib.import_module(names[references["coordinates"]])
    temp.coord_init[blocks, threads](arrays["coordinates"])
    for key in references:
        if key != "coordinates":
            if key in arrays:
                temp = importlib.import_module(names[references[key]])
                temp.array_init[blocks, threads](arrays[key],
                                                 arrays["coordinates"])
                #Requires making the init calls all generic
                
                


