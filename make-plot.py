import os
import datetime as dt
from subprocess import Popen
from tqdm import tqdm
import argparse
from utils import make_log_dir, get_subdirs

parser = argparse.ArgumentParser(description='Plot embedding results')
parser.add_argument('-i', '--input', help='Date stamped directory containing subdirectories of inputs', required=True)
parser.add_argument('-r', '--results', help='Results directory containing umap.csv, tsne.csv embedding files', required=True)
parser.add_argument('-o', '--output', help='Path to top level directory to write sample-specific embedding plots under', required=True)
parser.add_argument('--subdirs', help='Files containing subdirectories to process', type=str, nargs="+")
parser.add_argument('--clear-logs', help='Clear previous log files', action='store_true')
parser.add_argument('--umap', help='Plot UMAP embedding', action='store_true')
parser.add_argument('--tsne', help='Plot t-SNE embedding', action='store_true')
parser.add_argument('--grid-plot-dim', help='Dimension of grid plot', default=15000, type=int)
parser.add_argument('--comb-scenes', help='Combine different scenes from same patient', action='store_true')
parser.add_argument('--channels', help='Channels to plot', nargs='+', type=int, required=True)
args = parser.parse_args()

args.input = os.path.abspath(args.input)
args.results = os.path.abspath(args.results)
args.output = os.path.abspath(args.output)

ld = make_log_dir()

subdirs = get_subdirs(args)
print("Found {} samples".format(len(subdirs)))
if args.comb_scenes:
    print("args.comb_scenes True. Combining samples by patient / stain protocol...")
    subdirs = [s.split("-Scene")[0] for s in subdirs]
    subdirs = list(set(subdirs))

fdir = os.path.dirname(os.path.realpath(__file__))
bash_path = os.path.join(fdir, "tools", "plot.sh")

datafiles = {"tsne": None, "umap": None}
if args.umap:
    f = os.path.join(args.results, "umap.csv")
    if os.path.isfile(f):
        datafiles["umap"] = f
        print("Found UMAP embedding data: {}".format(f))
     

if args.tsne:
    f = os.path.join(args.results, "tsne.csv")
    if os.path.isfile(f):
        datafiles["tsne"] = f
        print("Found t-SNE embedding data: {}".format(f))

try:
    procs = []
    samps = []
    log_files = []
    print("Starting processes to plot t-SNE / UMAP embeddings for {} samples...".format(len(subdirs)))

    for s in tqdm(subdirs):

        for plot, f in datafiles.items():
            if f is None:
                continue

            python_args = "--data_file {} --write_dir {} --sample {} --grid_plot_dim {} --channels {}".format(
                f, os.path.join(args.output, s), s, args.grid_plot_dim, " ".join([str(c) for c in args.channels])
            )
            if args.comb_scenes:
                python_args += " --comb_scenes"
            bash_string = "srun -c {} --time {} --mem {}gb {} {}".format(
                args.cpus, args.time, str(args.mem), bash_path, python_args)

            if not os.path.exists(os.path.join(args.output, "logs")):
                os.makedirs(os.path.join(args.output, "logs"))
            elif args.clear_logs:
                for f in os.listdir(os.path.join(args.output, "logs")):
                    if f.startswith(s):
                        os.remove(os.path.join(args.output, "logs", f))

            out_file = open(
                os.path.join(args.output, "logs",
                "{}-make-{}-plot-{}.log".format(s, plot, dt.datetime.now().strftime("%m_%d_%Y_%H_%M"))),
                "w"
            )
            out_file.write(bash_string + "\n")
            out_file.flush()
            log_files.append(out_file)

            if not os.path.isdir(os.path.join(args.output, s)):
                os.makedirs(os.path.join(args.output, s))

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