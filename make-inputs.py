import argparse
import os
import shutil
import datetime as dt
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Build input directory structure for MCMICRO')
parser.add_argument('-i','--input',help='Input directory of raw CZIs and (optionally) a corresponding markers csv files. \ If markers file provided it is assumed to apply to all CZIs. If it is not provided, a null markers file is created.',required=True)
parser.add_argument('-o', '--output', help='Top level output directory', required=True)
parser.add_argument('--no-copy', help='Do not copy raw CZI files to output directory', action='store_true')
args = parser.parse_args()

args.input = os.path.abspath(args.input)
args.output = os.path.abspath(args.output)

markers = False
args.output = os.path.join(args.output, dt.datetime.now().strftime('%Y_%m_%d'))

dirs = ["illumination", "raw", "registration", "segmentation", "viz", "logs", "tables"]
files = os.listdir(args.input)
###################
## TODO: add support for multiple marker files corresponding to different CZI files
###################
if "markers.csv" in files:
    print("Found markers file")
    markers = True
    files.remove("markers.csv")
print(f"Found {len(files)} data files")
for f in tqdm(files):
    if '.czi' in f:
        subdir = os.path.join(args.output, f.replace(".czi", ""))
    elif '.ome.tif' in f:
        subdir = os.path.join(args.output, f.replace(".ome.tif", ""))
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    for d in dirs:
        if not os.path.exists(os.path.join(subdir, d)):
            os.makedirs(os.path.join(subdir, d))

    if '.ome.tif' in f:
        shutil.copy(os.path.join(args.input, f), os.path.join(subdir, 'registration', f))
    if not args.no_copy:
        shutil.copy(os.path.join(args.input, f), os.path.join(subdir, "raw"))
    if markers:
        shutil.copy(os.path.join(args.input, "markers.csv"), os.path.join(subdir, "markers.csv"))
    else:
        with open(os.path.join(subdir, "markers.csv"), "w") as f:
            f.write("cycle,marker_name")
print("Done")
