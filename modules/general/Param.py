"""
Machinery that reads in parameters from parameters.txt

It only considers lines with a single : to be value statements, and will
attempt to save the values as the strictest possible of int, float or string.

It should be quite durable to writing whatever else you want in the file.
Since it just produces a dictionary there's not much help if a value is
missing.
"""

from modules.general import locator

filename = locator.location["parameters"]
raw_data = locator.file_reader(filename)

lookup = {}
for line in raw_data:
    key = line[0]
    try:
        #Dirty way to figure out what data type the value is supposed to be
        value = line[1].strip()
        value = float(line[1])
        value = int(line[1])
    except ValueError:
        pass
    lookup[key] = value


##if __name__ == "__main__":
##    for key in lookup:
##        print(key, "=", lookup[key])
        

