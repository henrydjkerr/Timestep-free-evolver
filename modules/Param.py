"""
Machinery that reads in parameters from parameters.txt

It only considers lines with a single : to be value statements, and will
attempt to save the values as the strictest possible of int, float or string.

It should be quite durable to writing whatever else you want in the file.
Since it just produces a dictionary there's not much help if a value is
missing.
"""

filename = "parameters.txt"

paramfile = open(filename)
lookup = {}

for line in paramfile:
    if ":" in line:
        sections = line.strip("\n").split(":")
        key = sections[0]
        try:
            value = sections[1].strip()
            value = float(sections[1])
            value = int(sections[1])
        except ValueError:
            pass

        lookup[key] = value


if __name__ == "__main__":
    for key in lookup:
        print(key, "=", lookup[key])
        

