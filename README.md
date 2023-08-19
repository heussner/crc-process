# crc-process
HELEN is an image analysis tool for detecting circulating hybrid cells (CHCs) in immunofluorescence images of peripheral blood. These are the steps that HELEN takes to accomplish this task at human level:

1. make-inputs.py: takes Zeiss-produced .czi image and creates a dated directory in which results will be stored.
2. run-mcmicro.py: uses MCMICRO to stitch, correct background illumination (BaSIC), and register the image.
3. run-mesmer.py: performs nuclear/whole-cell segmentation.
4. run-table.py: creates feature tables
5. run-threshold.py: measures positive pixel ratio for each cell
6. run-callout.py: computationally gates markers
7. run-crop.py: crops individual cells

