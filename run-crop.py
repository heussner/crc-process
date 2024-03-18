import os
import datetime as dt
from subprocess import Popen
from tqdm import tqdm
import argparse
from utils import make_log_dir, get_subdirs

parser = argparse.ArgumentParser(description='Crop cells')
parser.add_argument('-i', '--input', help='Date stamped directory containing subdirectories of inputs', required=True)
parser.add_argument('-l', '--labels', help='If specific cells from run-callout.py need to be cropped', action='store_true')
parser.add_argument('-a', '--arrange', help='Arrange channels in specified order', action='store_true')
parser.add_argument('-o', '--output', help='Path to top level directory to write sample-specific subdirectories of crops under')
parser.add_argument('-n', '--normalize', help='Normalization method',choices=["robust", "percentile", "min_max", None], default=None)
parser.add_argument('--crop-length', help="Maximum cell major axis length to crop", default=64, type=int)
parser.add_argument('--subdirs', help='Files containing subdirectories to process', type=str, nargs="+")
parser.add_argument('--clear-logs', help='Clear previous log files', action='store_true')
args = parser.parse_args()

args.input = os.path.abspath(args.input)

ld = make_log_dir()

subdirs = get_subdirs(args)

nsubs = []
for s in subdirs:
    if 's1' in s:
        continue
    else:
        nsubs.append(s)
subdirs = nsubs
print("Found {} samples".format(len(subdirs)))

fdir = os.path.dirname(os.path.realpath(__file__))
bash_path = os.path.join(fdir, "tools", "crop.sh")

try:
    procs = []
    samps = []
    log_files = []
    time_ = dt.datetime.now().strftime("%m_%d_%Y_%H_%M")
    print(f"Starting {len(subdirs)} cell crop processes...")
    for s in tqdm(subdirs):
        mti_file = os.path.abspath(os.path.join(args.input, s, "registration", s + ".ome.tif"))
        seg_file = os.path.abspath(os.path.join(args.input, s, "segmentation", s + ".ome.tif" + "_CELL__MESMER.tif"))
        
        if args.labels:
            label_dir = os.path.abspath(os.path.join(args.input, s, "tables",s+'_CALLOUTS.pkl'))
        else:
            label_dir = None
        
        if args.arrange:
            markers_dir = os.path.abspath(os.path.join(args.input, s, "markers.csv"))
        else:
            markers_dir = None
        
        if args.output:
            save_dir = os.path.abspath(os.path.join(args.output, s))
            if os.path.isdir(args.output) == False:
                os.makedirs(agrs.output)
            if os.path.isdir(save_dir) == False:
                os.makedirs(save_dir)
        else:
            save_dir = os.path.abspath(os.path.join(os.path.join(args.input, s, "crops")))
        crop_length = args.crop_length
        
        
        python_script_args = f"--segmentation_path {seg_file} --mti_path {mti_file} --labels {label_dir} --arrange {markers_dir} --save_dir {save_dir} --save_prefix {s} --crop_length {crop_length} --normalize {args.normalize}"
        
        bash_string = f'bash {bash_path} {python_script_args}'

        if not os.path.exists(os.path.join(args.input, s, "logs")):
            os.makedirs(os.path.join(args.input, s, "logs"))
        elif args.clear_logs:
            for f in os.listdir(os.path.join(args.input, s, "logs")):
                if f.startswith("run-crop"):
                    os.remove(os.path.join(args.input, s, "logs", f))

        out_file = open(
            os.path.join(args.input, s, "logs", f"crop-{time_}.out"),
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
            print(f"Error: {e}")
            print(f"Failed on sample {s}")
            print("#" * 80)

    print("Waiting for processes to complete...")
    err_file = open(f"{ld}/run-crop_err_{time_}.log", "w")
    found_err = False
    for i, p in enumerate(tqdm(procs)):
        p.wait()
        if p.returncode != 0:
            found_err = True
            err_file.write(f"{samps[i]}\n")
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
