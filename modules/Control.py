"""
A common access point for all the custom runtime-selected modules.
Also used as the common access point for parameters by importing ParamPlus.
"""

import importlib
from modules.ParamPlus import lookup

version = lookup["version"]

#------------------------------------------------------------------------------

filename = "import-profiles.txt"

controlfile = open(filename)
names = {}
readflag = False

for line in controlfile:
    #print(line)
    line = line.replace(",", "").replace("\n", "")
    if (len(line) > 0) and (line[0] == ">"):
        if line[1:] == version:
            readflag = True
        else:
            readflag = False
        #print(line, "vs", version)
        #print("readflag", readflag)
    elif (readflag == True) and (":" in line):
        sections = line.split(":")
        key = sections[0]
        try:
            value = sections[1].strip()
        except ValueError:
            pass

        names[key] = "modules." + value

#print(names)

#------------------------------------------------------------------------------
        
d = importlib.import_module(names["distance"])
c = importlib.import_module(names["connection_weight"])
v = importlib.import_module(names["v_calcs"])
i = importlib.import_module(names["i_update"])

vi = importlib.import_module(names["v_init"])
ii = importlib.import_module(names["i_init"])
xi = importlib.import_module(names["coord_init"])

check = importlib.import_module(names["fire_check"])
solve = importlib.import_module(names["root_finder"])

