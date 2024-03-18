from argparse import ArgumentParser
import os
from glob import glob
import datetime as dt
from tifffile import imread, imwrite
import numpy as np
from tqdm import tqdm
from skimage.segmentation import find_boundaries
from skimage import img_as_uint
from skimage.measure import regionprops_table
import shutil
import pandas as pd
import sys
import urllib3
import json
import traceback

def make_log_dir():
    log_dir = ".logs"
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    return log_dir

def get_subdirs(args):
    if args.subdirs is None:
        subdirs = os.listdir(args.input)
    else:
        subdirs = []
        for s in args.subdirs:
            if s == "QC":
                continue
            #else:
                #with open(s, 'r') as f:
                #subdirs += [x.strip() for x in f.readlines()]
            else:
                subdirs.append(s)
    return sorted(subdirs)

def write_viz_paths(args):
    subdirs = get_subdirs(args)
    all_paths = []
    print("Writing viz paths for {} samples".format(len(subdirs)))
    for s in tqdm(subdirs):
        viz_paths = glob(os.path.join(args.input, s, "viz", "*.ome.tiff"))
        all_paths += sorted([os.path.abspath(p).split("/", 2)[-1] for p in viz_paths])
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    viz_paths_file = "{}_{}.txt".format(args.viz_paths_tag, dt.datetime.now().strftime("%m_%d_%Y_%H_%M"))
    with open(os.path.join(args.output, viz_paths_file), "w") as f:
        for p in all_paths:
            f.write(p.replace(os.path.expanduser("~") + "/strgar", "") + "\n")

def join_channels(impath, segpath_cell, segpath_nuc, savepath):
    os.makedirs(os.path.join(savepath, "join"))
    im = imread(impath)
    cell = imread(segpath_cell).astype(np.uint32)
    nuc = imread(segpath_nuc).astype(np.uint32)
    outfile = "joined.tif"
    cell_bounds = find_boundaries(cell, connectivity=1, mode='outer')
    cell_bounds = img_as_uint(cell_bounds)
    cell_bounds = np.expand_dims(cell_bounds, 0)
    nuc_bounds = find_boundaries(nuc, connectivity=1, mode='outer')
    nuc_bounds = img_as_uint(nuc_bounds)
    nuc_bounds = np.expand_dims(nuc_bounds, 0)
    im = np.append(im, cell_bounds, axis=0)
    im = np.append(im, nuc_bounds, axis=0)
    im = np.expand_dims(im,0)
    imwrite(os.path.join(savepath, "join", outfile), im)
    

def split_channels(args):
    subdirs = get_subdirs(args)
    print("Splitting channels for {} samples".format(len(subdirs)))
    for s in tqdm(subdirs):
        fpath = os.path.join(args.input, s, "registration", s + ".ome.tif")
        if not os.path.exists(os.path.join(args.input, s, "viz", "split")):
            os.makedirs(os.path.join(args.input, s, "viz", "split"))
        im = imread(fpath)
        im = np.squeeze(im)
        if len(im.shape) == 2:
            im = np.expand_dims(im, 0)
        for c in range(im.shape[0]):
            im_c = im[c, :, :]
            imwrite(os.path.join(args.input, s, "viz", "split", "{}_c{}.tif".format(s, c)), im_c)
            del im_c
        del im

def write_seg_bounds(args):
    subdirs = get_subdirs(args)
    print("Writing segmentation boundaries for {} samples".format(len(subdirs)))
    if args.cell_size_threshold is not None:
        print("Using user provided cell size threshold of {}".format(round(args.cell_size_threshold)))
        if args.output is None:
            raise ValueError("Please provide an output directory when using a cell size threshold")
        logfile = os.path.join(args.output, "excluded-count-{}-{}.txt".format(args.cell_size_threshold, dt.datetime.now().strftime("%m_%d_%Y_%H_%M")))
    for s in tqdm(subdirs):
        seg = imread(os.path.join(args.input, s, "segmentation", s + ".ome.tif_CELL__MESMER.tif"))
        seg = np.squeeze(seg)
        if args.cell_size_threshold is not None:
            rps = pd.DataFrame(regionprops_table(seg, properties=["major_axis_length", "label"]))
            rps = rps[rps["major_axis_length"] > args.cell_size_threshold]
            with open(logfile, "a+") as f:
                f.write("{} {}\n".format(s, len(rps.index)))
            tmp_seg = np.zeros(seg.shape)
            for _, rp in rps.iterrows():
                tmp_seg += (seg == rp["label"])
            seg = (tmp_seg > 0)
            outfile = "segmentation_bounds_gt{}um.tif".format(round(args.cell_size_threshold))
        else:
            outfile = "segmentation_bounds.tif"
        bounds = find_boundaries(seg, connectivity=2, mode='thick')
        bounds = img_as_uint(bounds)
        imwrite(os.path.join(args.input, s, "viz", "split", outfile), bounds)
        del seg
        del bounds

def remove_seg_bounds(args):
    subdirs = get_subdirs(args)
    print("Removing segmentation bounds for {} samples".format(len(subdirs)))
    for s in tqdm(subdirs):
        files = glob(os.path.join(args.input, s, "viz", "split", "segmentation_bounds*"))
        for f in files:
            os.remove(f)

def remove_directories(args):
    subdirs = get_subdirs(args)
    print("Removing {} directories for {} samples".format(args.dir, len(subdirs)))
    for s in tqdm(subdirs):
        if os.path.isdir(os.path.join(args.input, s, args.dir)):
            shutil.rmtree(os.path.join(args.input, s, args.dir))

def reorder_channels(args):
    subdirs = get_subdirs(args)
    print("Reordering image channels for {} samples".format(len(subdirs)))
    for s in tqdm(subdirs):
        markerspath = os.path.join(args.input, s, "markers.csv")
        markers = pd.read_csv(markerspath)
        channels = markers["channel"].tolist()
        channels = [int(x) for x in channels]
        if channels != sorted(channels):
            impath = os.path.join(args.input, s, "registration", s + ".ome.tif")
            im = imread(impath)
            im = np.squeeze(im)
            tmp = [im[c, :, :] for c in channels]
            im = np.stack(tmp, axis=0)
            imwrite(impath, im)
            
def slack_notification(args):
    try:
        slack_message = {'text': args}
        webhook_url = 'https://hooks.slack.com/services/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'
        http = urllib3.PoolManager()
        response = http.request('POST',
                                webhook_url,
                                body = json.dumps(slack_message),
                                headers = {'Content-Type': 'application/json'},
                                retries = False)
    except:
        traceback.print_exc()

    return True
            
def count_false_positives(args):
    subdirs = get_subdirs(args)
    print("Counting false positives for {} samples".format(len(subdirs)))
    msg = ""
    for s in tqdm(subdirs):
        segpath = os.path.join(args.input, s, "segmentation", s + ".ome.tif__MESMER.tif")
        impath = os.path.join(args.input, s, "registration", s + ".ome.tif") 
        seg = imread(segpath).squeeze()
        mti = imread(impath).squeeze()
        mti = np.transpose(mti, (1, 2, 0))
        nchannels = mti.shape[2]
        rps = pd.DataFrame(
            regionprops_table(seg, mti, properties=["mean_intensity"], separator="_")
        )
        n_fp = (rps[[f"mean_intensity_{i}" for i in range(nchannels)]].sum(1) == 0).sum()
        msg += f"{s}: {n_fp}/{len(rps.index)} false positives\n"
        del mti
        del seg
        del rps
    
    print(msg)

    if args.output is not None:
        outfile = os.path.join(
            args.output, 
            "false_positives_{}.txt".format(dt.datetime.now().strftime("%m_%d_%Y_%H_%M"))
        )
        with open(outfile, "a") as f:
            f.write(msg)
        print(f"Results written to {outfile}")
    print("Done\n")


def compute_major_axis_dist(args):
    subdirs = get_subdirs(args)
    print("Computing major axis length distribution over {} samples".format(len(subdirs)))
    majax = []
    bysamp = {}
    for s in tqdm(subdirs):
        segpath = os.path.join(args.input, s, "segmentation", s + ".ome.tif__MESMER.tif")
        seg = imread(segpath).squeeze()
        rps = pd.DataFrame(regionprops_table(seg, properties=["major_axis_length"]))
        majax += rps["major_axis_length"].values.tolist()
        bysamp[s] = rps["major_axis_length"].values.tolist()
        del seg
        del rps

    majax_px = np.array(majax)
    majax_mc = majax_px * args.mpp
    ncell = len(majax)

    percentiles = [99.0, 99.9, 99.95, 99.99, 99.995, 99.999, 99.9999]
    msg = f"Found {ncell} cells\n" + f"Computing statistics assuming {args.mpp} microns per pixel\n" + "Results:\n"
    for p in percentiles:
        pxval = np.percentile(majax_px, p)
        mcval = np.percentile(majax_mc, p)
        msg += f"\t{p}% Threshold: {np.round(pxval, 2)} pixels | {np.round(mcval, 2)} microns\n"
        msg += "\t# cells excluded:\n"
        for s in subdirs:
            msg += f"\t\t{s}: {(np.array(bysamp[s]) > pxval).sum()} / {len(bysamp[s])}\n"
        msg += "-"*80 + "\n"

    print(msg)

    if args.output is not None:
        outfile = os.path.join(args.output, "majax_dist_{}.txt".format(dt.datetime.now().strftime('%m_%d_%Y_%H_%M')))
        with open(outfile, "w") as f:
            f.write(msg)
        datafile = os.path.join(args.output, "majax_data_{}.npy".format(dt.datetime.now().strftime('%m_%d_%Y_%H_%M')))
        np.save(
            datafile, 
            {
                "all": majax_px, 
                "bysamp": bysamp, 
                "mpp": args.mpp, 
                "percentiles": percentiles, 
            }
        )
        print(f"Summary results written to {outfile}")
        print(f"Data written to {datafile}")
        print("Done\n")


if __name__ == "__main__":

    parser = ArgumentParser(description='Processing Utils')
    parser.add_argument('-i', '--input', help='Date stamped directory containing subdirectories of inputs', required=True)
    parser.add_argument('-o', '--output', help='File or directory to write outptut to')
    parser.add_argument('--subdirs', help='Files containing subdirectories to process', type=str, nargs="+")
    parser.add_argument('--write-viz-paths', help='Make a file containing paths to all viz files', action='store_true')
    parser.add_argument('--viz-paths-tag', help='Tag to add to viz paths file', type=str, default="viz_paths")
    parser.add_argument('--split-channels', help='Split image channels into separate files', action='store_true')
    parser.add_argument('--write-seg-bounds', help='Extract and write segmentation boundaries', action='store_true')
    parser.add_argument('--remove-seg-bounds', help='Remove segmentation boundaries', action='store_true')
    parser.add_argument('--remove-directories', help='Remove directory specified with the --dir flag', action='store_true')
    parser.add_argument('--slack', help='Send progress updates via slack notification', action='store_true')
    parser.add_argument('--count-false-positives', help='Count false positives', action='store_true')
    parser.add_argument('--compute-major-axis-dist', help='Compute major axis length distribution', action='store_true')
    parser.add_argument('--mpp', help='Microns per pixel', type=float, default=0.325)
    parser.add_argument('--cell-size-threshold', help='If provided only write segmentation bounds for cells larger than value (pixels)', type=float)
    parser.add_argument('--dir', help='Directory to remove', type=str, choices=['viz', 'viz/split'], required='--remove-directories' in sys.argv)
    parser.add_argument('--reorder-channels', help='Reorder channels', action='store_true')
    args = parser.parse_args()

    args.input = os.path.abspath(args.input)
    if args.output is not None:
        args.output = os.path.abspath(args.output)

    ## TODO: Improve logging / monitoring of progress
    if args.write_viz_paths:
        write_viz_paths(args)
    if args.remove_directories:
        remove_directories(args)
    if args.split_channels:
        split_channels(args)
    if args.write_seg_bounds:
        write_seg_bounds(args)
    if args.slack:
        slack_notification(args)
    if args.compute_major_axis_dist:
        compute_major_axis_dist(args)
    if args.count_false_positives:
        count_false_positives(args)
    if args.remove_seg_bounds:
        remove_seg_bounds(args)
    if args.reorder_channels:
        reorder_channels(args)






    
