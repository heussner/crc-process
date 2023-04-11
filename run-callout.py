import os
import datetime as dt
from subprocess import Popen
from tqdm import tqdm
import argparse
import pandas as pd
from utils import make_log_dir, get_subdirs
import numpy as np
## TODO: Add compartment functionality
parser = argparse.ArgumentParser(description='Crop cells')
parser.add_argument('-i', '--input', help='Date stamped directory containing subdirectories of inputs', required=True)
parser.add_argument('-o', '--output', help='Path to top level directory to write sample-specific subdirectories of crops under')
parser.add_argument('-c', '--cpus', help='Number CPUs to request from scheduler', default=4, type=int)
parser.add_argument('-m', '--mem', help='Memory to request from the scheduler (in GB)', default=64, type=int)
parser.add_argument('-t', '--time', help='Time string to request from the scheduler', default='1:00:00', type=str)
#parser.add_argument('-c', '--compartment',help='Compartment to measure', choices=["whole-cell", "nuclear", "both"], #default="whole-cell", type=str)
parser.add_argument('--exacloud', help='specifies running a scheduled job on exacloud', action='store_true')
parser.add_argument('--clear-logs', help='Clear previous log files', action='store_true')
args = parser.parse_args()

args.input = os.path.abspath(args.input)

ld = make_log_dir()

subdirs = os.listdir(args.input)
print("Found {} samples".format(len(subdirs)))

fdir = os.path.dirname(os.path.realpath(__file__))
bash_path = os.path.join(fdir, "tools", "callout.sh")

try:
    procs = []
    samps = []
    log_files = []
    print("Starting {} callout processes...".format(len(subdirs)))
    for s in tqdm(subdirs):
        
        table_file = os.path.abspath(os.path.join(args.input, s, "tables", s + "_THRESHOLDED.pkl"))
        if args.output:
            out_dir = os.path.abspath(args.output)
        else:
            out_dir = os.path.abspath(os.path.join(os.path.join(args.input, s, "tables")))
        
        markers_df = pd.read_csv(os.path.join(args.input, s, "markers.csv"))
        markers_df = markers_df.replace({np.nan: None})
        markers = markers_df[markers_df["seg_type"] == "membrane"]["marker_name"].tolist() #only membrane markers. need better integration
        PPR_thresholds = [0.7, 0.7]#hardcoded, based on elbow plots in :
                                 
        python_script_args = "--table {} --output {} --markers {} --PPR_thresholds {} --save_prefix {} --compartment {}".format(
            table_file, out_dir, " ".join([m for m in markers]), " ".join([str(t) for t in PPR_thresholds]), s, "whole-cell"
        )
        bash_string = f'bash {bash_path} {python_script_args}'
        if args.exacloud: bash_string = "srun -p gpu --gres gpu:{} --time {} --mem {}gb {} {}".format(
            args.gpu_str, args.time, str(args.mem), bash_path, python_script_args)

        if not os.path.exists(os.path.join(args.input, s, "logs")):
            os.makedirs(os.path.join(args.input, s, "logs"))
        elif args.clear_logs:
            for f in os.listdir(os.path.join(args.input, s, "logs")):
                if f.startswith("run-callout"):
                    os.remove(os.path.join(args.input, s, "logs", f))

        out_file = open(
            os.path.join(args.input, s, "logs",
            "run-callout-{}.out".format(dt.datetime.now().strftime("%m_%d_%Y_%H_%M"))),
            "w"
        )
        log_files.append(out_file)
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
            samps.append(s)
        except ValueError as e:
            print("Error: {}".format(e))
            print("Failed on sample {}".format(s))
            print("#" * 80)

    print("Waiting for processes to complete...")
    err_file = open("{}/run-callout_err_{}.log".format(ld, dt.datetime.now().strftime("%m_%d_%Y_%H_%M")), "w")
    found_err = False
    for i, p in enumerate(tqdm(procs)):
        p.wait()
        if p.returncode != 0:
            found_err = True
            err_file.write(f"{samps[i]}\n")
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