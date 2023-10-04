"""
Reads and stores where individual configuration files are to be found.
Also hosts the general function for file reading.
"""

def file_reader(filename):
    file = open(filename)
    file_data = []

    for line in file:
        if ":" in line:
            line = line.replace(",", "").replace("\n", "")
            sections = line.split(":")
            file_data.append(sections)
    file.close()
    return file_data

filename = "settings_selection.txt"
raw_data = file_reader(filename)

location = {}
for line in raw_data:
    try:
        location[line[0]] = line[1]
    except IndexError:
        print("Incorrect formatting in {}".format(filename))
        raise



