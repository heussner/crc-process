# description
Helen is an image analysis tool for detecting circulating hybrid cells (CHCs) in immunofluorescence images of peripheral blood.

# installation:
1. make a virutual environment
2. install dependences
3. follow instructions here to install MCMICRO

# usage:
Each sample/batch should be stored in a folder along with a markers.csv file that describes the image metadata. The markers.csv file should be organized as so:

1. make-inputs.py: takes Zeiss-produced .czi image and creates a dated directory in which results will be stored.
2. run-mcmicro.py: uses MCMICRO to stitch, correct background illumination (BaSIC), and register the image.
3. run-mesmer.py: performs nuclear/whole-cell segmentation.
4. run-table.py: creates feature tables
5. run-threshold.py: measures positive pixel ratio (PPR) for each cell
6. run-callout.py: computationally gates markers
7. run-crop.py: crops individual cells

