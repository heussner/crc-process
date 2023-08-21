# description
Helen is an image analysis tool for detecting circulating hybrid cells (CHCs) in immunofluorescence images of peripheral blood.

# installation:
1. make a virutual environment
2. install dependences
3. follow instructions here to install MCMICRO

# usage:
Each sample/batch should be stored in a folder along with a markers.csv file that describes the image metadata. 
'''
/
|- PROJECT-DATA
|  |- in-data
|  |  |- file1.czi
|  |  |- file2.czi
|  |  |- file3.czi
|  |  |- markers.csv
'''
The markers.csv file should be organized as so:
'''
|- cycle            # indicates imaging round
|- channel          # zero-indexed channel integer for that marker. note that channels need not be ordered
|- marker_name      # corresonds to the protein or fluorophore.
|- seg_type         # indicates whether the marker is a nuclear/membrane marker so segmentation knows which channels
                    # only a single row should include the "nuclear" label value; however, multiple "membrane" markers can be used
                    # if you label multiple channels as "membrane" then the `projection` column must be filled in the first row.
|- projection       # tells Mesmer how to combine the multiple "membrane" marker channels
                    # supported values are "mean" and "max" corresponding to mean or maximum projection of the "membrane" labeled channels
'''
Next, run scripts in this order to make use of the full pipeline. for troubleshooting, log files are stored in the sample directory in .log folder.
1. make-inputs.py: takes Zeiss-produced .czi image and creates a dated directory in which results will be stored.
2. run-mcmicro.py: uses MCMICRO to stitch, correct background illumination (BaSIC), and register (Ashlar) the image.
3. run-mesmer.py: performs nuclear/whole-cell segmentation.
4. run-table.py: creates feature tables
5. run-threshold.py: measures positive pixel ratio (PPR) for each cell
6. run-callout.py: computationally gates markers
7. run-crop.py: crops individual cells

# acknowledgements
We thank Luke Strgar for his contributions the project. 

