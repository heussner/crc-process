import argparse
import os
from tqdm import tqdm
from subprocess import Popen
import shutil
import datetime as dt
import sys
from utils import make_log_dir, get_subdirs

parser = argparse.ArgumentParser(description='Make pyramidal OME TIFFs from CZI files for viewing with Viv')
parser.add_argument('-i', '--input', help='Date stamped directory containing subdirectories of inputs', required=True)
parser.add_argument('--raw', help='Make from raw rather than corrected data', action='store_true')
parser.add_argument('--segmentation', help='Include segmentation in the output', action='store_true')
parser.add_argument('--large-cell-file', help='Include segmentation from large cells file only', action='store_true')
parser.add_argument('--cell-size-threshold', help='Specifies large cell segmentation file to use', type=float, required='--large-cell-file' in sys.argv)
parser.add_argument('--subdirs', help='Files containing subdirectories to process', type=str, nargs="+")
parser.add_argument('--clear-logs', help='Clear previous log files', action='store_true')
parser.add_argument('--write-paths', help='Make a file containing paths to all VIV compatible files', action='store_true')
parser.add_argument('--write-dir', help='Directory for writing paths file if --write-paths is set', required='--write-paths' in sys.argv)
parser.add_argument('--exacloud', help='specifices running with slurm on exacloud', action='store_true')
args = parser.parse_args()

args.input = os.path.abspath(args.input)
if args.write_paths:
    args.write_dir = os.path.abspath(args.write_dir)

ld = make_log_dir()

subdirs = get_subdirs(args)

fdir = os.path.dirname(os.path.realpath(__file__))
bash_path = os.path.join(fdir, "tools", "make-ometiff.sh")

print("Found {} samples".format(len(subdirs)))

try:
    procs = []
    tmp_dirs = []
    viz_files = []
    log_files = []
    cleanup = []
    if args.write_paths:
        if args.raw:
            viz_paths_file = os.path.join(
                args.write_dir, "viz_paths_raw_{}.txt".format(dt.datetime.now().strftime("%m_%d_%Y_%H_%M")))
        elif args.segmentation:
            viz_paths_file = os.path.join(
                args.write_dir, "viz_paths_seg_{}.txt".format(dt.datetime.now().strftime("%m_%d_%Y_%H_%M")))
        else:
            viz_paths_file = os.path.join(
                args.write_dir, "viz_paths_{}.txt".format(dt.datetime.now().strftime("%m_%d_%Y_%H_%M")))

    print("Starting {} processes to make OME TIFFs...".format(len(subdirs)))
    print("INFO: This may take some time if --segmentation flag is set or previous tmp files exist")
    if args.raw and args.segmentation:
        raise ValueError(
            "Segmentation cannot be included with raw data. \
                Assumes image files are split by channel and located in viz/split.")

    for s in tqdm(subdirs):

        viz_dir = os.path.join(args.input, s, "viz")
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        if args.raw:
            fpath = os.path.join(args.input, s, "raw", s + ".czi")
            viz_file = os.path.join(args.input, s, "viz", s + "__ORIG.ome.tiff")
            log_str = "make-viz-raw"
            tmp_dir = os.path.join(args.input, s, "viz", s + "__RAW__TMP")
        elif args.segmentation:
            split_dir = os.path.join(args.input, s, "viz", "split")
            pattern_file = os.path.join(split_dir, "input.pattern")
            files = [f for f in os.listdir(split_dir) if s in f]
            fname = files[0][:-5] 
            nchannels = len(files)
            fpath = os.path.join(split_dir, s + f"_c<0-{nchannels}>.tif")
            if args.large_cell_file:
                seg_file = os.path.join(split_dir, "segmentation_bounds_gt{}um.tif".format(round(args.cell_size_threshold)))
                viz_file = os.path.join(args.input, s, "viz", s + f"__SEG_LARGECELL_{round(args.cell_size_threshold)}.ome.tiff")
                log_str = "make-viz-seg-largecell"
            else:
                seg_file = os.path.join(split_dir, "segmentation_bounds.tif")
                viz_file = os.path.join(args.input, s, "viz", s + "__SEG.ome.tiff")
                log_str = "make-viz-seg"
            seg_channel = os.path.join(split_dir, fname + f"{nchannels}.tif")
            cleanup.append(seg_channel)
            shutil.copy(seg_file, seg_channel)
            with open(pattern_file, "w") as f:
                f.write(fpath)
            fpath = pattern_file
            tmp_dir = os.path.join(args.input, s, "viz", s + "__SEG__TMP")
        else:
            fpath = os.path.join(args.input, s, "registration", s + ".ome.tif")
            viz_file = os.path.join(args.input, s, "viz", s + ".ome.tiff")
            log_str = "make-viz"
            tmp_dir = os.path.join(args.input, s, "viz", s + "__TMP")

        viz_files.append(viz_file)
        
        tmp_dirs.append(tmp_dir)
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        bash_string = f'bash {bash_path} {fpath} {tmp_dir} {viz_file}'
        if args.exacloud: bash_string = "srun --time {} -c {} --mem {}gb {} {} {} {}".format(args.time, str(args.cpus), str(args.mem), bash_path, fpath, tmp_dir, viz_file)

        if not os.path.exists(os.path.join(args.input, s, "logs")):
            os.makedirs(os.path.join(args.input, s, "logs"))
        elif args.clear_logs:
            for f in os.listdir(os.path.join(args.input, s, "logs")):
                if f.startswith(log_str):
                    os.remove(os.path.join(args.input, s, "logs", f))

        out_file = open(
            os.path.join(args.input, s, "logs",
            "{}-{}.out".format(log_str, dt.datetime.now().strftime("%m_%d_%Y_%H_%M"))),
            "w"
        )
        out_file.write(bash_string + "\n")
        out_file.flush()
        log_files.append(out_file)

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
    err_file = open("{}/make-viz_err_{}.log".format(ld, dt.datetime.now().strftime("%m_%d_%Y_%H_%M")), "w")
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
        elif args.write_paths:
            with open(viz_paths_file, "a") as f:
                f.write(viz_files[i] + "\n")
    err_file.close()
    if found_err:
        print("FAILURE: One or more processes exited with non-zero error codes. See {}".format(err_file.name))
    else:
        os.remove(err_file.name)
    
    print("Cleaning up tmp data directories...")
    for d in tqdm(tmp_dirs):
        shutil.rmtree(d)

    for f in log_files:
        f.close()

    for f in cleanup:
        os.remove(f)

except KeyboardInterrupt:
    print("WARNING: Running processes will continue to run")
    print("Exiting...")
    exit(0)

print("All processes complete")