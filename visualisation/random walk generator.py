import random

x_min = -10
x_max = 10
y_min = -10
y_max = 10

x_range = x_max - x_min
y_range = y_max - y_min

def new_coord(old_coord, delta):
    new_x = ((old_coord[0] + delta[0] - x_min) % x_range) + x_min
    new_y = ((old_coord[1] + delta[1] - y_min) % y_range) + y_min
    return [new_x, new_y]

number_of_points = 1000
record_of_points = []
t = 0
t_sigma = 0.1

coord = [x_range/2 + x_min, y_range/2 + y_min]

while len(record_of_points) < number_of_points:
    record_of_points.append([t, coord[0], coord[1]])
    
    delta_t = abs(random.gauss(0, t_sigma))
    #delta_t = 0.1
    t += delta_t
    
    delta = [random.gauss(0, delta_t**0.5),
             random.gauss(0, delta_t**0.5)]
    coord = new_coord(coord, delta)

outfile = open("2D_demo.csv", "w")
for entry in record_of_points:
    line = "{},{},{},{}\n".format("0", str(entry[0]), str(entry[1]), str(entry[2]))
    outfile.write(line)
outfile.close()

    

    



    
