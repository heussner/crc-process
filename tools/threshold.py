from tifffile import imread, imwrite
import argparse
import numpy as np
import os
import numpy as np
import pandas as pd
import os
import pickle
from skimage.measure import regionprops_table, label
from skimage.io import imread
from tqdm import tqdm

def pixel_values(regionmask, intensity_image):
    """
    Custom regionprop
    """
    return len(np.unique(intensity_image[regionmask]))

parser = argparse.ArgumentParser(description='Threshold a single sample')
parser.add_argument('-i', '--image', help='Input image path', required=True)
parser.add_argument('-s', '--segmentation', help='Input segmentation folder', required=True)
parser.add_argument('-t', '--table', help='Input feature table path', required=True)
parser.add_argument('-m', '--markers', help='List of channel names (markers) starting from 0',nargs='+', required=True, type=str)
parser.add_argument('-o', '--output', help='Output directory', required=True)
parser.add_argument('-n', '--name', help='File name', required=True, type=str)
parser.add_argument('-c', '--compartment',help='Compartment to threshold', choices=["whole-cell", "nuclear", "both"], default="whole-cell", type=str)
args = parser.parse_args()

image = imread(args.image) # read image
masks = os.listdir(args.segmentation) # read segmentation mask

mask_dict = {} # create mask dict based on compartments to threshold
if args.compartment == "whole-cell":
    mask_dict['cell'] = imread(os.path.join(args.segmentation, [m for m in masks if "CELL" in m][0]))
elif args.compartment == "nuclear":
    mask_dict['nucleus'] = imread(os.path.join(args.segmentation, [m for m in masks if "NUC" in m][0]))
else:
    mask_dict['cell'] = imread(os.path.join(args.segmentation, [m for m in masks if "CELL" in m][0]))
    mask_dict['nucleus'] = imread(os.path.join(args.segmentation, [m for m in masks if "NUC" in m][0]))

with open(args.table,'rb') as handle:
    table = pickle.load(handle) # read feature table

threshold_data = {}

print('Starting thresholding...')
for msk in mask_dict:
    mask = mask_dict[msk].copy()
    indices = [2,3,4] # hard-coded
    markers = ['CD45','EpCAM','VGT309'] # hard-coded
    for i,m in zip(indices, markers): # for i,m in enumerate(args.markers):
        print('Thresholding the ' + m + ' channel with the ' + msk + ' mask')

        img = image[i,:,:].copy() # get image channel

        img_fgd = np.where(mask.astype(int).copy()==0, np.nan, img) # clip image
        mu = np.mean(img_fgd[~np.isnan(img_fgd)])
        std = np.std(img_fgd[~np.isnan(img_fgd)])
        max_int = 1.5*(mu + 3*std)
        img[img > max_int] = max_int

        img_fgd = np.where(mask.astype(int).copy()==0, np.nan, img) # get foreground/background
        img_bgd = np.where(mask.astype(int).copy() != 0, np.nan, img)

        mu_b = np.mean(img_bgd[~np.isnan(img_bgd)]) # get statistics
        std_b = np.std(img_bgd[~np.isnan(img_bgd)])
        mu_f = np.mean(img_fgd[~np.isnan(img_fgd)])
        std_f = np.std(img_fgd[~np.isnan(img_fgd)])
        
        if (m == 'PanCK') | (m=='EpCAM'):         # set thresholds
            threshold = 1.5*(mu_b + 3 * std_b)
        else:
            threshold = mu_b + 3 * std_b
        
        threshold_data[msk+'_'+m+'_fgd_mu'] = mu_f
        threshold_data[msk+'_'+m+'_fgd_std'] = std_f
        threshold_data[msk+'_'+m+'_positive_ceiling'] = max_int
        threshold_data[msk+'_'+m+'_bgd_mu'] = mu_b
        threshold_data[msk+'_'+m+'_bgd_std'] = std_b
        threshold_data[msk+'_'+m+'_positive_threshold'] = threshold

        b_img = img.copy() # binarize image
        b_img = (b_img > threshold).astype(np.int_)

        max_area = table[msk+'_area'].quantile(0.997) # set min/max blob cell area for filtering artifacts
        min_area = table[msk+'_area'].quantile(0.01)
        
        labeled = label(b_img.astype(np.int_)) # remove small and large artifacts
        a = np.bincount(labeled.flatten())
        too_small = (a < min_area)
        mask1 = too_small[labeled]
        labeled[mask1] = 0
        stats1 = pd.DataFrame(regionprops_table(labeled,mask,properties=['label','area','intensity_image'],extra_properties=(pixel_values,))) # temp
        stats1['area_per_cell'] = stats1['area']/stats1['pixel_values']
        too_large = stats1[stats1['area_per_cell']>max_area]
        too_large = too_large['label'].tolist()
        for l in too_large:
            labeled[labeled==l] = 0
        b_img[labeled == 0] = 0
        
        stats = pd.DataFrame(regionprops_table(mask.astype(int), b_img, properties = ['mean_intensity'])) # measure positive pixel ratio (PPR)
        stats = stats.rename(columns={'mean_intensity':msk+'_'+m+'_PPR'})
        table = pd.concat([table, stats], axis=1)

    with open(os.path.join(args.output, args.name+'_THRESHOLDED.pkl'), 'wb') as handle:
        pickle.dump(table, handle) # save thresholded data

    with open(os.path.join(args.output, args.name+'_METADATA.pkl'), 'wb') as handle:
        pickle.dump(threshold_data, handle) # save metadata
print('Done!')