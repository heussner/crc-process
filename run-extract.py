import os
import argparse
from tqdm import tqdm
from subprocess import Popen
import datetime as dt
from utils import make_log_dir, get_subdirs
import time
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='Extract annotated cells')
parser.add_argument('-i', '--input', help='Date stamped directory containing subdirectories of inputs', required=True)
parser.add_argument('--subdirs', help='Files containing subdirectories to process if not all', type=str, nargs="+")
parser.add_argument('--clear-logs', help='Clear previous log files', action='store_true')
args = parser.parse_args()

args.input = os.path.abspath(args.input)

ld = make_log_dir()

subdirs = get_subdirs(args)

fdir = os.path.dirname(os.path.realpath(__file__))
bash_path = os.path.join(fdir, "tools", "extract.sh")
    
print(f"Found {len(subdirs)} samples")

try:

    procs = []
    log_files = []
    time_ = dt.datetime.now().strftime("%m_%d_%Y_%H_%M")
    print(f"Starting {len(subdirs)} extraction processes...")
    for s in tqdm(subdirs):

        if not os.path.exists(os.path.join(args.input, s, "logs")):
            os.makedirs(os.path.join(args.input, s, "logs"))
        elif args.clear_logs:
            for f in os.listdir(os.path.join(args.input, s, "logs")):
                if f.startswith("table"):
                    os.remove(os.path.join(args.input, s, "logs", f))
        
        out_file = open(
            os.path.join(args.input, s, "logs",
            f"extract-{time_}.out"),
            "w"
        )
        log_files.append(out_file)

        image_file = os.path.join(args.input, s, "registration", s + ".ome.tif")
        seg_file = os.path.join(args.input, s, "segmentation", s + ".ome.tif_CELL__MESMER.tif")
        annotation_file = os.path.join(args.input, s, s+'ann.tif')
        table_file = os.path.join(args.input, s, 'tables',s+'_THRESHOLDED.pkl')
        metadata_file = os.path.join(args.input, s, 'tables',s+'_METADATA.pkl')
        out_dir = os.path.join(args.input, s, "tables")
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        python_script_args = f"-i {image_file} -s {seg_file} -a {annotation_file} -t {table_file} -m {metadata_file} -o {out_dir} -n {s}"
       
        bash_string = f"bash {bash_path} {python_script_args}"
        
        out_file.write(bash_string + "\n")
        out_file.flush()

        try:
            p = Popen(
                bash_string.split(),
                stdout=out_file,
                stderr=out_file,
                universal_newlines=True,
            )
            procs.append(p)
        except ValueError as e:
            print(f"Error: {e}")
            print(f"Failed on sample {s}")
            print("#" * 80)

    print("Waiting for processes to complete...")
    err_file = open(f"{ld}/run-extract_err_{time_}.log", "w")
    found_err = False
    for i, p in enumerate(tqdm(procs)):
        p.wait()
        if p.returncode != 0:
            found_err = True
            err_file.write(f"{subdirs[i]}\n")
            err_file.write(f"Process Args: {p.args}\n")
            err_file.write(f"Error Code: {p.returncode}\n")
            err_file.write("#" * 80 + "\n")
            err_file.flush()
    err_file.close()
    if found_err:
        print(f"FAILURE: One or more processes exited with non-zero error codes. See {err_file.name}")
    else:
        os.remove(err_file.name)

    for f in log_files:
        f.close()

except KeyboardInterrupt:
    print("WARNING: Running processes will continue to run")
    print("Exiting...")
    exit(0)

print("All processes complete")