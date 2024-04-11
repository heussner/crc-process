# Colorectal cancer PBMC processing utilities
Used to detect circulating hybrid cells in https://onlinelibrary.wiley.com/doi/10.1002/cyto.a.24826?af=R
## Installation

#### 1. Create a virtual environment + install dependencies:

```
conda create --name crc-process python=3.8.12 pip=21.2.4 setuptools=58.0.4
conda activate crc-process
pip install -r requirements.txt
conda install -c ome bioformats2raw=0.3.0 raw2ometiff=0.3.0 
conda install cudatoolkit=11.3.1 cudnn=8.2.1
```

#### 2. Install MCMICRO
Install the Nextflow-based MCMICRO tool following instructions here: https://mcmicro.org/instructions/nextflow/. We copied the relevant commands for convenience below. NOTE: If you are working on Exacloud you do not need to install Docker as this code uses Singularity for containers, which is available by default on the compute nodes.

**IMPORTANT**: you must also set the `NXF_HOME` environment variable in your `.bashrc` file to a directory in which you have read & write permissions and is under RDS (OHSU's Research Data Storage service). By default `NXF_HOME` is assigned to user `$HOME`, which has storage limits imposed by ACC. For example, one might create (or already have) a directory `/home/groups/ChangLab/<MYUSER>`. In this case, add the line `export NXF_HOME="/home/groups/ChangLab/<MYUSER>/.nextflow"` to `.bashrc` -- this would be a good `NXF_HOME` path because it is owned by you and sits underneath a group with configured RDS. After assigning `NXF_HOME` in `~/.bashrc` make sure to `source ~/.bashrc`.

```
cd ~                                                # Change into your home directory
curl -s https://get.nextflow.io | bash              # Create the Nextflow executable
mkdir bin                                           # Creates a bin directory in the home folder (cwd)
mv nextflow ~/bin                                   # Moves nextflow executable to that directory
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc   # Make the directory accessible on $PATH
source ~/.bashrc                                    # Reload the shell configuration
cd /some/other/directory                            # Move out of your home directory
nextflow                                            # Verify nextflow install worked and executable is found on $PATH
```

This code was developed with `nextflow version 21.10.5.5658`. If you find you are having trouble with MCMICRO portions of the pipeline and your version of Nextflow is different (check with `nextflow -v`) consider adding `export NXF_VER="21.10.5"` to your `.bashrc` file then `source ~/.bashrc`.

Finally, (and **IMPORTANTLY**) run `setup.sh` to test you are able to process Nextflow example data with singularity. This will also download necessary source code and container images for future runs:

```
chmod +x setup.sh
bash setup.sh
```

After completion, `ls $NXF_HOME` should show the following files:

```
labsyspharm-basic-illumination-1.1.0.img
labsyspharm-ashlar-1.14.0.img
```

## Usage

#### 1. Make input directory, markers.csv file

We recommend creating a top level directory for the whole procedure and a subdirectory for the input data files and markers.csv file. For example:

    /
    |- PROJECT-DATA
    |  |- in-data
    |  |  |- file1.czi
    |  |  |- file2.czi
    |  |  |- file3.czi
    |  |  |- markers.csv

**Each .czi file must be a single scene**. The markers.csv file must contain the following columns:

    |- cycle            # indicates imaging round
    |- channel          # zero-indexed channel integer for that marker. note that channels need not be ordered
    |- marker_name      # corresonds to the protein or fluorophore.
    |- seg_type         # indicates whether the marker is a nuclear/membrane marker so segmentation knows which channels
                        # only a single row should include the "nuclear" label value; however, multiple "membrane" markers can be used
                        # if you label multiple channels as "membrane" then the `projection` column must be filled in the first row.
    |- projection       # tells Mesmer how to combine the multiple "membrane" marker channels
                        # supported values are "mean" and "max" corresponding to mean or maximum projection of the "membrane" labeled channels
    
Here is an example `markers.csv` file with 5 "misordered" channels and three membrane markers with a desired maximum projection before segmentation: 

```
cycle,channel,marker_name,seg_type,projection
1,4,DNA,nuclear,max
1,1,AF647,,
1,0,AF751,membrane,
1,2,AF555,membrane,
1,3,FITC,membrane,
```

#### 2. Explore then run pipeline

Before running any code please review the main source files and their command line usage, which includes documentation. At a high level, here are descriptions of the function of each script and the order you probably want to use them in when processing PBMC samples for analysis / modelling with a VAE. If you are working with tissue slides you probably don't need to do anything with `run-crop.py` -- the MCMICRO illumination correction and stitching sequence along with Mesmer segmentation should be sufficient. 

1. `make-inputs.py`: create new directory structure, one for each input samples. All processed files, intermediate files, log files will live under the subdirectory named after the sample in question. The `-o` flag indicates the parent folder for all sample-specific subdirectories. In our experience using the `PROJECT-DATA` folder as in the above example makes sense.

2. `make-viz.py` (optional): create Avivator compatible images from raw samples (with `--raw` flag).

3. `run-mcmicro.py`: perform illumination correction and stitching. When running multiple samples simultaneously this is liable to fail due to multiple Nextflow processes trying to obtain a lock on the `labsyspharm/mcmicro` github repository. The present solution is to add a short delay between parallel job invocations; however, this is not robust to SLURM queueing delay. See Known Issues section for details.

4. `make-viz.py` (optional): create Avivator compatible images from corrected data.

5.`run-mesmer.py`: segment the corrected image data. The `--compartment` flag can be used to specify whole-cell or nuclear segmentation.

6. `utils.py -- methods `split_channels`, `write_seg_bounds` (optional): to visualize segmentations you first need to split the image into separate data channels and save segmentation cell boundaries as another single channel image. These separate data channels can later be combined as an Avivator compatible image.

7. `make-viz.py` (optional): create Avivator compatible images including segmentation boundaries (using `--segmentation`).

8. `run-crop.py` (PBMC only): construct single cell image dataset by cropping out cell instances.

9. `run-table.py`: extract single cell feature (mean intensity, centroid, area, etc)

10. `run-threshold.py` (optional): threshold the PanCK/CD45 channels (for CHC detection)

11. `run-callout.py` (optional) : identify likely-CHCs (for CHC detection)

12. `run-extract.py` (optional): extract CHCs from label mask

## High-level Picture

Most of the heavy-lifting steps in the pipeline will launch many jobs (one for each data sample) in parallel from the respective python scripts. This includes `run-mcmicro.py`, `run-mesmer.py`, `run-crop.py`, and `make-viz.py`. Take a look at any one of these scripts and you will see they share a common structure that can be summarized in pseudocode as:

```
from subprocess import Popen ## Library method for launching processes from a python script

args = parse_args() ## Collect job/script specific command line arguments

subdirs = get_subdirs(args) ## Select data subdirectories for processing

procs = [] ## Place to save individual processes after they get started

## for loop at the level of data samples
for s in tqdm(subdirs):
    
    .......  ## Do script specific setup work
    bash_string = .... ## Command to execute in subprocess
    out_file = open("/path/to/log/file", "w") ## where to write subprocess stdout and stderr
    
    try:
        p = Popen(
            bash_string.split(),
            stdout=out_file,
            stderr=out_file,
            universal_newlines=True,
        )
        procs.append(p)
    except ValueError as e:
        print("Error: {}".format(e))
        print("Failed on sample {}".format(s))
        print("#" * 80)

for i, p in enumerate(tqdm(procs)): ## Wait for all subprocesses to finish
    p.wait()
    if p.returncode != 0:
        ..... ## Handle errors
        
...... ## Cleanup
```

If you cancel one of these python scripts before completion it is likely the launched processes will continue to run -- to cancel the launched SLURM processes using `scancel`. More info on managing SLURM can be found on the ACC docs page -- https://wiki.ohsu.edu/display/ACC/ACC+-+Advanced+Computing+Center. Usage of utils.py can occur from an `salloc` session because this does not launch SLURM managed processes -- you will slow down the head nodes if you do not launch an salloc session (e.g. `salloc -c 8 --mem 128 --time=3:00:00`) or explicitly run the python job with `srun`. It is generally best practice to invoke all scripts with srun; however, the scripts that launch separate SLURM jobs are light-weight.

## Debugging 

When jobs fail information about the failed jobs will be printed to log files under the `.logs` directory. More specific information is available in the `logs` subdirectory of each sample data folder (e.g. the subdirectory produced by `make-inputs.py`). The `logs` subdirectory of a sample data folder will include log files for specific jobs invoked for that data sample. For example, you might see a file titled `mcmicro-M_D_Y_H_m.out` which would correspond to execution of mcmicro at the specified datetime stamp. A related, and important log file for mcmicro execution, is `nextflow.log` which includes output directly from the Nextflow engine (on which mcicro is built). 

## Aside -- Visualization w/ Avivator

The pipeline steps above include multiple opportunities to visualize data before processing, after correction, and with overlaid segmentation boundaries. These steps rely on usage of Avivator. The documentation describing install and usage can be found here: https://github.com/hms-dbmi/viv/tree/master/sites/docs/tutorial. This link also includes useful information: http://viv.gehlenborglab.org/. In brief, you will need to install node and NPM, some node packages, and Avivator itself via pip. In essence, data will be tunnelled via ssh and then viewed in the browser using the Avivator webgl backend. Before this can happen, data must be converted to into a pyramidal format before reading with Avivator. The conversion process occurs in `tools/make-ometiff.sh`, which relies on bioformats2raw (https://github.com/glencoesoftware/bioformats2raw) and raw2ometiff (https://github.com/glencoesoftware/raw2ometiff). 

Assuming correct intallation of the required software, after running `make-viz.py` at any point in the pipeline you can use `tools/http-serv.sh` to bootup a server. Since you would likely apply the same visualization procedure to other workflows, it may make sense to copy this script to your home directory. The `http-serv.sh` script takes a single argument, which is interpreted as a path to the folder that should be hosted. Since this script is running a server it should be kept running and therefore invoked with `sbatch http-serv.sh ./path/to/folder/to/host`. Shortly after invocation a `slurm-xxxx.out` file will be created in `cwd`, which the script writes stdout and stderr to. `cat`'ing this file will print ssh information to actually tunnel the data. For example, output may look something like below. The important bit is the ssh command, which must be executed in another shell to open the tunnel. Finally, copy the avivator.gehlenborglab.org url into your browser, append the path to the data (relative to the hosted path) and press go. 

Avivator is under active, rapid development and has been known to break following browser version updates. In our experience, Safari and Firefox have been more stable than Chrome. We recommend finding a browser / version that works and then turn off auto-updates. 

```
Node: exanode-3-6
Port: 8245

ssh strgar@exahead1.ohsu.edu -L 8245:exanode-3-6:8245

http://127.0.0.1:8245

https://avivator.gehlenborglab.org/?image_url=http://127.0.0.1:8245/
Starting up http-server, serving ./path/to/folder/to/host

http-server version: 14.0.0

http-server settings:
CORS: *
Cache: 3600 seconds
Connection Timeout: 120 seconds
Directory Listings: visible
AutoIndex: visible
Serve GZIP Files: false
Serve Brotli Files: false
Default File Extension: none

Available on:
  http://127.0.0.1:8245
  http://172.20.15.173:8245
```
## Acknowledgements
We thank Luke Strgar for contributing to the image preprocessing and quality control code, Koei Chin for his advice on IF imaging,  the Advanced Light Microscopy Core at OHSU, and Nicole Giske, Ranish Patel, John Swain, Abby Gillingham, Ethan Lu, and Ashvin Nair for help in annotating CHCs. This work was supported by the National Institutes of Health (R01 CA253860). YHC acknowledges funding from the National Institute of Health (U2CCA233280) and Kuni Foundation Imagination Grants, ANA acknowledges funding from the National Cancer Institute (F31CA271676). The resources of the Exacloud high-performance computing environment developed jointly by OHSU and Intel and the technical support of the OHSU Advanced Computing Center is gratefully acknowledged. 
