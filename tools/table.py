from tifffile import imread, imwrite
import argparse
import numpy as np
import os
import pandas as pd
from skimage.io import imread, imshow
from skimage.measure import regionprops_table
from skimage.morphology import dilation, disk
from skimage.segmentation import expand_labels
import pickle

parser = argparse.ArgumentParser(description='Make feature table of a single sample')
parser.add_argument('-i', '--image', help='Input image path', required=True)
parser.add_argument('-s', '--segmentation', help='Input segmentation folder', required=True)
parser.add_argument('-o', '--output', help='Output directory', required=True)
parser.add_argument('-m', '--markers', help='List of channel names (markers) starting from 0',nargs='+',required=True)
parser.add_argument('-n', '--name', help='File name', required=True, type=str)
parser.add_argument('-c', '--compartment',help='Compartment to measure', choices=["whole-cell", "nuclear", "both"], default="both", type=str)
parser.add_argument('-p', '--properties',help='Regionprops',nargs='+', default=["label","centroid","area","mean_intensity","equivalent_diameter","major_axis_length","eccentricity"], type=str)
args = parser.parse_args()

#load masks and images
masks = os.listdir(args.segmentation)
im = imread(args.image)

if args.compartment == "both":
    nuc_mask = imread(os.path.join(args.segmentation, [i for i in masks if "NUC" in i][0]))
    cell_mask = imread(os.path.join(args.segmentation, [i for i in masks if "CELL" in i][0]))
elif args.compartment == "nuclear":
    nuc_mask = imread(os.path.join(args.segmentation, [i for i in masks if "NUC" in i][0]))
else:
    cell_mask = imread(os.path.join(args.segmentation, [i for i in masks if "CELL" in i][0]))

print("Loaded imasks from {}".format(args.segmentation))

print("Making feature tables")

#make feature tables, rename columns to specify compartment/marker
if args.compartment == "both":
    nuc_table = pd.DataFrame(regionprops_table(nuc_mask.astype(int), im, properties=args.properties))
    p_names = nuc_table.columns.values.tolist()
    for p in p_names:
        nuc_table = nuc_table.rename(columns={p:'nucleus_'+p})
    
    cell_table = pd.DataFrame(regionprops_table(cell_mask.astype(int), im, properties=args.properties))
    p_names = cell_table.columns.values.tolist()
    for p in p_names:
        cell_table = cell_table.rename(columns={p:'cell_'+p})
    
    for i,l in enumerate(args.markers):
        nuc_table = nuc_table.rename(columns={'nucleus_mean_intensity-'+str(i):'nucleus_'+l+'_mean_intensity'})
        cell_table = cell_table.rename(columns={'cell_mean_intensity-'+str(i):'cell_'+l+'_mean_intensity'})
    
    feat_table = pd.concat([nuc_table, cell_table], axis = 1)
  
elif args.compartment == "nuclear":
    nuc_table = pd.DataFrame(regionprops_table(nuc_mask.astype(int), im, properties=args.properties))
                                    
    for p in p_names:
        nuc_table = nuc_table.rename(columns={p:'nuclei_'+p})
    
    for i,l in enumerate(args.markers):
        nuc_table = nuc_table.rename(columns={'nucleus_mean_intensity-'+str(i):'nucleus_'+l+'_mean_intensity'})
    
    feat_table = nuc_table

else:
    cell_table = pd.DataFrame(regionprops_table(cell_mask.astype(int), im, properties=args.properties))
    p_names = cell_table.columns.values.tolist()
    for p in p_names:
        cell_table = cell_table.rename(columns={p:'cell_'+p})
    
    for i,l in enumerate(args.markers):
        cell_table = cell_table.rename(columns={'cell_mean_intensity-'+str(i):'cell_'+l+'_mean_intensity'})
    
    feat_table = cell_table

with open(os.path.join(args.output, args.name+'.pkl'),'wb') as handle:
    pickle.dump(feat_table,handle)
                                  
print("Done!")