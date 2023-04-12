from operator import xor
import os
import argparse
from tqdm import tqdm
import datetime as dt
from subprocess import Popen
import pandas as pd
from utils import make_log_dir, get_subdirs
import numpy as np

## TODO: Add support for mpp

parser = argparse.ArgumentParser(description='Run thresholding')
parser.add_argument('-i', '--input', help='Date stamped directory containing subdirectories of inputs', required=True)
parser.add_argument('-m', '--mem', help='Memory to request from the scheduler (in GB)', default=64, type=int)
parser.add_argument('-t', '--time', help='Time string to request from the scheduler', default='4:00:00', type=str)
parser.add_argument('-g', '--gpu-str', help='GPU resource string to request from scheduler (gpuname:ngpu)', default='rtx2080:1')
parser.add_argument('--subdirs', help='Files containing subdirectories to process', type=str, nargs="+")
parser.add_argument('--clear-logs', help='Clear previous log files', action='store_true')
parser.add_argument('--compartment', help='Compartment to segment', choices=["whole-cell", "nuclear", "both"], default="whole-cell", type=str)
parser.add_argument('--nuclear-only', help='Run segmentation with nuclear marker only', action='store_true')
parser.add_argument('--exacloud', help='run script using slurm on exacloud', action='store_true')
args = parser.parse_args()

args.input = os.path.abspath(args.input)

assert not (args.compartment == "whole-cell" and args.nuclear_only), "If using nuclear marker only you should be running nuclei segmentation"

ld = make_log_dir()

subdirs = get_subdirs(args)

print("Found {} samples".format(len(subdirs)))

fdir = os.path.dirname(os.path.realpath(__file__))
bash_path = os.path.join(fdir, "tools", "threshold.sh")

try:
    procs = []
    log_files = []
    print("Starting {} threshold processes...".format(len(subdirs)))
    for s in tqdm(subdirs):

        if not os.path.exists(os.path.join(args.input, s, "logs")):
            os.makedirs(os.path.join(args.input, s, "logs"))
        elif args.clear_logs:
            for f in os.listdir(os.path.join(args.input, s, "logs")):
                if f.startswith("threshold"):
                    os.remove(os.path.join(args.input, s, "logs", f))
        
        out_file = open(
            os.path.join(args.input, s, "logs",
            "threshold-{}.out".format(dt.datetime.now().strftime("%m_%d_%Y_%H_%M"))),
            "w"
        )
        log_files.append(out_file)

        image_file = os.path.join(args.input, s, "registration", s + ".ome.tif")
        seg_dir = os.path.join(args.input, s, "segmentation")
        table_file = os.path.join(args.input, s, "tables", s+".pkl")
        out_dir = os.path.join(args.input, s, "tables")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        markers_df = pd.read_csv(os.path.join(args.input, s, "markers.csv"))
        markers_df = markers_df.replace({np.nan: None})
        #thresholds all markers
        markers = markers_df["marker_name"].tolist()
        marker_indexes = markers_df["channel"].tolist()
        #arrange marker names in order of channels
        zipped = zip(marker_indexes, markers)
        zipped = sorted(zipped, key=lambda x: x[0])
        markers = [list(t) for t in zip(*zipped)][1]
        #markers_df.loc[markers_df["seg_type"] == "membrane"] for only membrane
        python_script_args = "-i {} -s {} -t {} -m {} -o {} -n{} -c {}".format(
            image_file, seg_dir, table_file, " ".join([str(i) for i in markers]), out_dir, s, args.compartment)
       
        bash_string = f'bash {bash_path} {python_script_args}'
        if args.exacloud: bash_string = "srun -p gpu --gres gpu:{} --time {} --mem {}gb {} {}".format(
            args.gpu_str, args.time, str(args.mem), bash_path, python_script_args)
        
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
            print("Error: {}".format(e))
            print("Failed on sample {}".format(s))
            print("#" * 80)

    print("Waiting for processes to complete...")
    err_file = open("{}/run-threshold_err_{}.log".format(ld, dt.datetime.now().strftime("%m_%d_%Y_%H_%M")), "w")
    found_err = False
    for i, p in enumerate(tqdm(procs)):
        p.wait()
        if p.returncode != 0:
            found_err = True
            err_file.write(f"{subdirs[i]}\n")
            err_file.write("Process Args: {}\n".format(p.args))
            err_file.write("Error Code: {}\n".format(p.returncode))
            err_file.write("#" * 80 + "\n")
            err_file.flush()
    err_file.close()
    if found_err:
        print("FAILURE: One or more processes exited with non-zero error codes. See {}".format(err_file.name))
    else:
        os.remove(err_file.name)

    for f in log_files:
        f.close()

except KeyboardInterrupt:
    print("WARNING: Running processes will continue to run")
    print("Exiting...")
    exit(0)

print("All processes complete")
