#from display_1D_print import make_graph
from display_3D_xyt import make_graph
#from display_video_frames import make_graph
#from display_colourmap import make_graph

import time
stopwatch = time.time()

#make_graph("output-20250612145739-10000N-20000sp")
make_graph("output-20250612144739-10000N-10000sp")

print(time.time() - stopwatch)
