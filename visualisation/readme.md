This folder contains various tools to visualise outputs from the main program.  Copy an output .csv file from the output folder into this folder, paste its name into `thing.py` and select your visualisation method through module imports.

Your options for visualisation are:
 - `display_1D.py` plots individual firing events as dots, with the x-coordinate of the neuron's position on the horizontal axis and the firing time on the vertical axis.  Further spatial dimensions are discarded.
 - `display_1D_print.py` is as `display_1D.py`, but saves the output directly to file for convenience.
 - `display_3D_xyt` displays a 3D plot of firing events, using the x- and y-coordinates of the neuron's position and the firing time of the neuron.  Additionally, the points are coloured according to the firing time using the viridis colour scale.
 - `display_colourmap.py`
 - `display_grid_density.py`
 - `display_point_density.py`
 - `display_video.py`
 - `display_video_frames.py`

If you run any of the above options directly, they will make use of `2D_demo_newtype.csv` as a dummy input file.  This represents a simple random walk, and is not related to the actual outputs of the main program.
