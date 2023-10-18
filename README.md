# 3D_MirrorMicroscope
Code to convert 2D image of object and reflection into 3D trajectory

# 3D_Detect.py 
Reads video frame by frame and detects object and reflection x,y location. Uses distance between object and reflection to calculate z location. Input=Video file (*.mov or *.mp4 format), Output=3D_Detect.csv

# 3D_Track.py
Reads detect file (x,y,z for each frame) and assigns ID to objects. Uses "Cam" model to track with persistance to maintain ID during object dropout. Input=3D.Detect.csv, Output=3D_Track.csv

# 3D_Plot.py
Reads track file (x,y,z,ID for each frame) and generates 3D plots, one per ID. The maximum number of points per plot can be specified, breaking an ID into several plots, to prevent clutter. Input=3D_Track.csv, Output=3D plots in a subdirectory.

# TrackingObjects.pptx
Powerpoint deck explaining how "Cam" tracking method works.

