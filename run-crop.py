import os
import datetime as dt
from subprocess import Popen
from tqdm import tqdm
import argparse
from utils import make_log_dir, get_subdirs
## TODO: Add compartment functionality
parser = argparse.ArgumentParser(description='Crop cells')
parser.add_argument('-i', '--input', help='Date stamped directory containing subdirectories of inputs', required=True)
parser.add_argument('-l', '--labels', help='If specific cells from run-callout.py need to be cropped', action='store_true')
parser.add_argument('-o', '--output', help='Path to top level directory to write sample-specific subdirectories of crops under')
parser.add_argument('-n', '--normalize', help='Robust saturation + min/max normalization per scene', action='store_true')
parser.add_argument('-c', '--cpus', help='Number CPUs to request from scheduler', default=4, type=int)
parser.add_argument('-m', '--mem', help='Memory to request from the scheduler (in GB)', default=64, type=int)
parser.add_argument('-t', '--time', help='Time string to request from the scheduler', default='1:00:00', type=str)
parser.add_argument('--crop-length', help="Maximum cell major axis length to crop", default=64, type=int)
parser.add_argument('--skip', help="Skip scene 1", action='store_true')
parser.add_argument('--subdirs', help='Files containing subdirectories to process', type=str, nargs="+")
parser.add_argument('--exacloud', help='specifies running a scheduled job on exacloud', action='store_true')
parser.add_argument('--clear-logs', help='Clear previous log files', action='store_true')
args = parser.parse_args()

args.input = os.path.abspath(args.input)

ld = make_log_dir()

subdirs = os.listdir(args.input)
if args.skip:
    dirs = []
    for s in subdirs:
        if 's1' in s:
            continue
        else:
            dirs.append(s)
subdirs = dirs
print("Found {} samples".format(len(subdirs)))

fdir = os.path.dirname(os.path.realpath(__file__))
bash_path = os.path.join(fdir, "tools", "crop.sh")

try:
    procs = []
    samps = []
    log_files = []
    print("Starting {} cell crop processes...".format(len(subdirs)))
    for s in tqdm(subdirs):
        
        mti_file = os.path.abspath(os.path.join(args.input, s, "registration", s + ".ome.tif"))
        seg_file = os.path.abspath(os.path.join(args.input, s, "segmentation", s + ".ome.tif" + "_CELL__MESMER.tif"))## See top
        
        if args.labels:
            label_dir = os.path.abspath(os.path.join(args.input, s, "tables",s+'_CALLOUTS.pkl'))
        else:
            label_dir = None
        
        if args.output:
            save_dir = os.path.abspath(os.path.join(args.output, s))
            if os.path.isdir(args.output) == False:
                os.makedirs(agrs.output)
            if os.path.isdir(save_dir) == False:
                os.makedirs(save_dir)
        else:
            save_dir = os.path.abspath(os.path.join(os.path.join(args.input, s, "crops")))
        crop_length = args.crop_length                        
        
        if args.normalize:
             python_script_args = "--segmentation_path {} --mti_path {} --labels {} --save_dir {} --save_prefix {} --crop_length {} --robust_saturation --min_max".format(
            seg_file, mti_file, label_dir, save_dir, s, crop_length
        )
        else:
            python_script_args = "--segmentation_path {} --mti_path {} --labels {} --save_dir {} --save_prefix {} --crop_length {}".format(
                seg_file, mti_file, label_dir, save_dir, s, crop_length
            )
        
        bash_string = f'bash {bash_path} {python_script_args}'
        if args.exacloud: bash_string = "srun -p gpu --gres gpu:{} --time {} --mem {}gb {} {}".format(
            args.gpu_str, args.time, str(args.mem), bash_path, python_script_args)

        if not os.path.exists(os.path.join(args.input, s, "logs")):
            os.makedirs(os.path.join(args.input, s, "logs"))
        elif args.clear_logs:
            for f in os.listdir(os.path.join(args.input, s, "logs")):
                if f.startswith("run-crop"):
                    os.remove(os.path.join(args.input, s, "logs", f))

        out_file = open(
            os.path.join(args.input, s, "logs",
            "run-crop-{}.out".format(dt.datetime.now().strftime("%m_%d_%Y_%H_%M"))),
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
    err_file = open("{}/run-crop_err_{}.log".format(ld, dt.datetime.now().strftime("%m_%d_%Y_%H_%M")), "w")
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