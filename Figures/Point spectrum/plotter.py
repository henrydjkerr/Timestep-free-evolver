import numpy as np
import matplotlib.pyplot as plt

print_mode = True
if print_mode:
    plt.figure(figsize=(4,4), dpi=400)
else:
    plt.figure(figsize=(4,4))

points = [
    (0, 0),
    (1.02, 0),
    (-0.49, 4.08),
    (-1.90, 5.47),
    (-3.36, 0.80),
    (-4.51, 5.00),
    (-2.32, 7.75)
    ]

points_x = [x[0] for x in points]
points_y = [y[1] for y in points]
minus_y = [-y for y in points_y]

plt.scatter(points_x, points_y, color="#000000", zorder = 1)
plt.scatter(points_x[2:], minus_y[2:], color="000000", zorder = 1)


plt.axhline(0, color="#999999", linestyle="dotted", zorder = 0)
plt.axvline(0, color="#999999", linestyle="dotted", zorder = 0)
plt.xlim(-5, 5)
plt.yticks([-8, -6, -4, -2, 0, 2, 4, 6, 8])
plt.xticks([-4, -2, 0, 2, 4])

plt.title("Point spectrum for wave stability")
plt.xlabel("Re($\\lambda$)")
plt.ylabel("Im($\\lambda$)")

if print_mode:
    plt.savefig("point spectrum.pdf")
    plt.savefig("point spectrum.png")
else:
    plt.show()

