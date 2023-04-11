import os
import argparse
from tqdm import tqdm
from subprocess import Popen
import datetime as dt
from utils import make_log_dir, get_subdirs
import time
import pandas as pd

parser = argparse.ArgumentParser(description='Run MCMICRO')
parser.add_argument('-i', '--input', help='Date stamped directory containing subdirectories of inputs', required=True)
parser.add_argument('-c', '--cpus', help='Number of CPUs to use', default=1, type=int)
parser.add_argument('-m', '--mem', help='Memory to request from the scheduler (in GB)', default=64, type=int)
parser.add_argument('-t', '--time', help='Time string to request from the scheduler', default='4:00:00', type=str)
parser.add_argument('--jvm-xmx', help='JVM max heap size (in GB)', default=256, type=int)
parser.add_argument('--nf-mem', help='Memory limit for the nextflow process (in GB)', default=256, type=int)
parser.add_argument('--subdirs', help='Files containing subdirectories to process if not all', type=str, nargs="+")
parser.add_argument('--clear-logs', help='Clear previous log files', action='store_true')
parser.add_argument('--exacloud', help='specifies running a scheduled job on exacloud', action='store_true')
args = parser.parse_args()

args.input = os.path.abspath(args.input)

ld = make_log_dir()

subdirs = get_subdirs(args)

fdir = os.path.dirname(os.path.realpath(__file__))
bash_path = os.path.join(fdir, "tools", "mcmicro_noslurm.sh")
if args.exacloud: bash_path = os.path.join(fdir, "tools", "mcmicro.sh")
    
print("Found {} samples".format(len(subdirs)))

try:

    procs = []
    log_files = []
    print("Starting {} MCMICRO processes...".format(len(subdirs)))
    for s in tqdm(subdirs):

        time.sleep(60)

        config_path = os.path.join(args.input, s, 'x.config')
        with open(config_path, 'w') as f:
            f.write("process.memory = '{}GB'\n".format(args.nf_mem))

        markers_file = os.path.join(args.input, s, 'markers.csv')
        markers = pd.read_csv(markers_file)
        align_channel = markers.loc[markers["seg_type"] == "nuclear"]["channel"].item()

        ashlar_opts = f"ashlar-opts: --flip-y -m 30 --filter-sigma 1.0 --pyramid --align-channel {align_channel}"
        with open(os.path.join(args.input, s, 'ashlar-opts.yml'), 'w') as f:
            f.write(ashlar_opts + '\n')

        if args.exacloud: 
            bash_string = "srun -c {} --mem {}gb --time {} {} {} {} {} {}".format(
            str(args.cpus), 
            str(args.mem), 
            args.time, 
            bash_path, 
            os.path.join(args.input, s), 
            config_path, 
            str(args.jvm_xmx),
            str(align_channel)
        )
        else:
            bash_string = f'bash {bash_path} {os.path.join(args.input, s)} {config_path} {str(args.jvm_xmx)} {str(align_channel)}'

        if not os.path.exists(os.path.join(args.input, s, "logs")):
            os.makedirs(os.path.join(args.input, s, "logs"))
        elif args.clear_logs:
            for f in os.listdir(os.path.join(args.input, s, "logs")):
                if f.startswith("mcmicro") or "nextflow" in f:
                    os.remove(os.path.join(args.input, s, "logs", f))

        out_file = open(
            os.path.join(args.input, s, "logs",
            "mcmicro-{}.out".format(dt.datetime.now().strftime("%m_%d_%Y_%H_%M"))),
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
    err_file = open("{}/run-mcmicro_err_{}.log".format(ld, dt.datetime.now().strftime("%m_%d_%Y_%H_%M")), "w")
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
