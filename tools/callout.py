from tifffile import imread, imwrite
import argparse
import numpy as np
import os
import math
import pandas as pd
from skimage.io import imread, imshow
from random import sample
from skimage.measure import regionprops_table
from skimage.morphology import dilation, disk
from skimage.segmentation import expand_labels
import pickle

parser = argparse.ArgumentParser(description='Make feature table of a single sample')
parser.add_argument('-t', '--table', help='Thresholded table path', required=True)
parser.add_argument('-o', '--output', help='Output folder', required=True)
parser.add_argument('-m', '--markers', help='List of channel names (markers)',nargs='+',required=True)
parser.add_argument('-p', '--PPR_thresholds', help='List of positive pixel ratio cutoffs for each marker in order of above markers', nargs='+', required=True)
parser.add_argument("--save_prefix", type=str, help="Prefix string for cropped image directory",required=True)
parser.add_argument('-c', '--compartment',help='Compartment to measure', choices=["whole-cell", "nuclear", "both"], default="whole-cell", type=str)
args = parser.parse_args()

#load thresholded table
with open(args.table, 'rb') as handle:
    table = pickle.load(handle)

thresholds = args.PPR_thresholds
markers = args.markers
print(markers)

#remove cells without nuclei
nuclei_labels = table['nucleus_label'].dropna().tolist()
table = table[table['cell_label'].isin(nuclei_labels)]

#remove small and large cells
min_area = table['cell_area'].quantile(0.01)
max_area = table['cell_area'].quantile(0.99)
table = table[table['cell_area'] <= max_area]
table = table[table['cell_area'] >= min_area]
    
    
if 's1' in args.table:
    print('Scene 1 skipped...')
else: 
    #sort by CK PPR
    sorted_table = table.sort_values(by=['cell_'+'CK'+'_PPR'],ascending=False,ignore_index=True).copy()
    #remove cells below CK threshold
    CK_positives = sorted_table[sorted_table['cell_'+'CK'+'_PPR'] >= float(thresholds[markers.index('CK')])].copy()

    #remove cells below CD45 threshold
    CHCs = CK_positives[CK_positives['cell_CD45_PPR'] >= float(thresholds[markers.index('CD45')])].copy()

    #record number of CHCs, index of last CHC, get list of CHC cell labels
    n = len(CHCs)
    print(f'Found {n} CHCs')
    idx = CHCs.axes[0].tolist()[-1]
    CHC_labels = CHCs['cell_label'].tolist()

    #remove CHCs from original sorted table
    CHCs_removed = sorted_table.loc[idx+1:len(sorted_table)].copy()

    #filter out CD45 negatives
    CD45_positives = CHCs_removed[CHCs_removed['cell_CD45_PPR'] >= float(thresholds[markers.index('CD45')])].copy()

    #get sorted remaining cell labels
    other_labels = CD45_positives['cell_label'].tolist()

    #get near non-CHC cell labels and random cell labels 
    non_CHC_labels = other_labels[0:n]
    
    #get negative controls, sorted by CD45 mean intensity
    Negative_controls = table[table['cell_'+'CK'+'_PPR_raw'] == 0].copy()
    Negative_controls = Negative_controls[Negative_controls['cell_'+'CD45'+'_PPR'] < 0.7]
    Negative_controls = Negative_controls.sort_values(by=['cell_CD45_mean_intensity'], ascending=True)
    Negative_control_labels = Negative_controls['cell_label'].tolist()[0:n]
    print(f'Found {n} negative cells')
    
    #(P)-positive CHCs 
    #(G)-grey area CHCs
    #(N)-grey area CHCs
    #save; order matters in these lists!
    label_dict = {"P":CHC_labels,"G":non_CHC_labels,"N":Negative_control_labels}

with open(os.path.join(args.output,args.save_prefix+'_CALLOUTS.pkl'), 'wb') as handle:
    pickle.dump(label_dict, handle)
                    
                    
                    
                    
                    
