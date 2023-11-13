from operator import xor
import os
import argparse
from tqdm import tqdm
import datetime as dt
from subprocess import Popen
import pandas as pd
from utils import make_log_dir, get_subdirs
import numpy as np
import time

## TODO: Add support for mpp

parser = argparse.ArgumentParser(description='Run Mesmer Segmentation')
parser.add_argument('-i', '--input', help='Date stamped directory containing subdirectories of inputs', required=True)
parser.add_argument('--compartment', help='Compartment to segment', choices=["whole-cell", "nuclear", "both"], required=True, type=str)
parser.add_argument('--subdirs', help='Files containing subdirectories to process', type=str, nargs="+")
parser.add_argument('--clear-logs', help='Clear previous log files', action='store_true')
parser.add_argument('--nuclear-only', help='Run segmentation with nuclear marker only', action='store_true')
args = parser.parse_args()

args.input = os.path.abspath(args.input)

assert not (args.compartment == "whole-cell" and args.nuclear_only), "If using nuclear marker only you should be running nuclei segmentation"

ld = make_log_dir()

subdirs = get_subdirs(args)

print(f"Found {len(subdirs)} samples")

fdir = os.path.dirname(os.path.realpath(__file__))
bash_path = os.path.join(fdir, "tools", "mesmer.sh")

try:
    procs = []
    log_files = []
    time = dt.datetime.now().strftime("%m_%d_%Y_%H_%M")
    print(f"Starting {len(subdirs)} Mesmer segmentation processes...")
    for s in tqdm(subdirs):
        
        time.sleep(960)

        if not os.path.exists(os.path.join(args.input, s, "logs")):
            os.makedirs(os.path.join(args.input, s, "logs"))
        
        elif args.clear_logs:
            for f in os.listdir(os.path.join(args.input, s, "logs")):
                if f.startswith("mesmer"):
                    os.remove(os.path.join(args.input, s, "logs", f))
        
        out_file = open(
            os.path.join(args.input, s, "logs",
            f"mesmer-{time}.out",
            "w"
        )
        log_files.append(out_file)

        in_file = os.path.join(args.input, s, "registration", s + ".ome.tif")
        out_dir = os.path.join(args.input, s, "segmentation")
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        markers_df = pd.read_csv(os.path.join(args.input, s, "markers.csv"))
        markers_df = markers_df.replace({np.nan: None})
        nuc_idx = markers_df.loc[markers_df["seg_type"] == "nuclear"]["channel"].item()
        mem_idx = markers_df.loc[markers_df["seg_type"] == "membrane"]["channel"].tolist()
        assert nuc_idx not in mem_idx
        proj_type = markers_df["projection"][0]
        assert proj_type in ["max", "mean", None]

        if args.nuclear_only:
            mem_idx = [nuc_idx]
        
        markers_string = " ".join([str(m) for m in mem_idx])
        python_script_args = f"-i {in_file} -o {out_dir} -n {nuc_idx} -m {markers_string} -c {args.compartment}"
        if not proj_type is None:
            python_script_args += " -p {}".format(proj_type)
        
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
    err_file = open(f"{ld}/run-mesmer_err_{time}.log", "w")
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
