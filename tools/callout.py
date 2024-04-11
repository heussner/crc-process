import argparse
import numpy as np
import os
import pandas as pd
import pickle

parser = argparse.ArgumentParser(description='Call out hybrid cells for a single sample')
parser.add_argument('-t', '--table', help='Thresholded table path', required=True)
parser.add_argument('-o', '--output', help='Output folder', required=True)
parser.add_argument('-m', '--markers', help='List of channel names (markers)',nargs='+',required=True)
parser.add_argument('-p', '--PPR_thresholds', help='List of positive pixel ratio cutoffs for each marker in order of above markers', nargs='+', required=True)
parser.add_argument("--save_prefix", type=str, help="Prefix string for cropped image directory",required=True)
args = parser.parse_args()

with open(args.table, 'rb') as handle:
    table = pickle.load(handle) # load thresholded table

thresholds = args.PPR_thresholds
markers = args.markers

tCD45 = float(thresholds[markers.index('CD45')])
tPanCK = float(thresholds[markers.index('PanCK')])

min_area = table['cell_area'].quantile(0.01) # remove small and large cells
max_area = table['cell_area'].quantile(0.997)
table = table[table['cell_area'] <= max_area]
table = table[table['cell_area'] >= min_area]
    
sorted_table = table.sort_values(by=['cell_PanCK_PPR'],ascending=False,ignore_index=True).copy() # sort by CK PPR

CK_positives = sorted_table[sorted_table['cell_PanCK_PPR'] >= tPanCK].copy() # remove cells below CK threshold

CHCs = CK_positives[CK_positives['cell_CD45_PPR'] >= tCD45].copy() # remove cells below CD45 threshold

n = len(CHCs) # record number of CHCs, index of last CHC, get list of CHC cell labels
print(f'Found {n} CHCs')
idx = CK_positives.axes[0].tolist()[-1]
CHC_labels = CHCs['cell_label'].tolist()

CHCs_removed = sorted_table.loc[idx+1:len(sorted_table)].copy() # remove CHCs from original sorted table

CD45_positives = CHCs_removed[CHCs_removed['cell_CD45_PPR'] >= tCD45].copy() # filter out CD45 negatives

near_CHC_labels = CD45_positives['cell_label'].tolist()[0:n] # get sorted remaining cell labels

negative_controls = table[table['cell_PanCK_PPR_raw'] == 0].copy() # get negative controls, sorted by CD45 mean intensity
negative_controls = negative_controls[negative_controls['cell_CD45_PPR'] < tCD45]
negative_controls = negative_controls.sort_values(by=['cell_CD45_mean_intensity'], ascending=True)
negative_labels = negative_controls['cell_label'].tolist()[0:n]

label_dict = {"P":CHC_labels,"G":near_CHC_labels,"N":negative_labels}
# (P)-positive CHCs 
# (G)-grey area CHCs
# (N)-negative controls

with open(os.path.join(args.output,args.save_prefix+'_CALLOUTS.pkl'), 'wb') as handle: # save
    pickle.dump(label_dict, handle)
                    
                    
                    
                    
                    
