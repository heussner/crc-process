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

parser = argparse.ArgumentParser(description='Threshold a single sample')
parser.add_argument('-i', '--image', help='Input image path', required=True)
parser.add_argument('-s', '--segmentation', help='Input segmentation folder', required=True)
parser.add_argument('-t', '--table', help='Input feature table path', required=True)
parser.add_argument('-m', '--markers', help='List of channel names (markers) starting from 0',nargs='+', required=True, type=str)
parser.add_argument('-o', '--output', help='Output directory', required=True)
parser.add_argument('-n', '--name', help='File name', required=True, type=str)
parser.add_argument('-c', '--compartment',help='Compartment to threshold', choices=["whole-cell", "nuclear", "both"], default="whole-cell", type=str)
args = parser.parse_args()

#read image
image = imread(args.image)
masks = os.listdir(args.segmentation)

#mask dictionary
mask_dict = {}
if args.compartment == "whole-cell":
    mask_dict['cell'] = imread(os.path.join(args.segmentation, [m for m in masks if "CELL" in m][0]))
elif args.compartment == "nuclear":
    mask_dict['nucleus'] = imread(os.path.join(args.segmentation, [m for m in masks if "NUC" in m][0]))
else:
    mask_dict['cell'] = imread(os.path.join(args.segmentation, [m for m in masks if "CELL" in m][0]))
    mask_dict['nucleus'] = imread(os.path.join(args.segmentation, [m for m in masks if "NUC" in m][0]))

#read feature table
with open(args.table,'rb') as handle:
    table = pickle.load(handle)

threshold_data = {}

print('Starting thresholding...')
for msk in mask_dict:
    mask = mask_dict[msk].copy()
    for i,m in enumerate(args.markers):
        print('Thresholding the ' + m + ' channel with the ' + msk + ' mask')
        #isolate channel
        img = image[:,:,i].copy()

        #clip channel
        img_fgd = np.where(mask.astype(int).copy()==0, np.nan, img.copy())
        mu = np.mean(img_fgd[~np.isnan(img_fgd)])
        std = np.std(img_fgd[~np.isnan(img_fgd)])
        max_int = 1.5*(mu + 3*std)
        img[img > max_int] = max_int

        #separate background/foreground
        img_fgd = np.where(mask.astype(int).copy()==0, np.nan, img.copy())
        img_bgd = np.where(mask.astype(int).copy() != 0, np.nan, img.copy())

        #measure mean/std of background
        mu_b = np.mean(img_bgd[~np.isnan(img_bgd)])
        std_b = np.std(img_bgd[~np.isnan(img_bgd)])
        
        #measure mean/std of foreground
        mu_f = np.mean(img_fgd[~np.isnan(img_fgd)])
        std_f = np.std(img_fgd[~np.isnan(img_fgd)])
        
        #set positive threshold
        if m == 'PanCK':
            threshold = 1.5*(mu_b + 3 * std_b)
            max_area = table[msk+'_area'].quantile(0.99)
        else:
            threshold = mu_b + 3 * std_b
            max_area = table[msk+'_area'].quantile(0.999)
        
        #store maximum threshold data (MPS - maximum positive threshold
        threshold_data[msk+'_'+m+'_fgd_mu'] = mu_f
        threshold_data[msk+'_'+m+'_fgd_std'] = std_f
        threshold_data[msk+'_'+m+'_positive_ceiling'] = max_int

        #store minimum threshold data (PT - positive threshold)
        threshold_data[msk+'_'+m+'_bgd_mu'] = mu_b
        threshold_data[msk+'_'+m+'_bgd_std'] = std_b
        threshold_data[msk+'_'+m+'_positive_threshold'] = threshold

        #binarize image
        b_img = img.copy()
        b_img = (b_img > threshold).astype(np.int_)
        c_img = b_img.copy()

        #set min/max blob area based on compartment quantile
        min_area = table[msk+'_area'].quantile(0.01)
        
        #label blobs
        labeled = label(b_img.astype(np.int_))

        #remove small blobs
        a = np.bincount(labeled.flatten())
        too_small = (a < min_area)
        mask1 = too_small[labeled]
        labeled[mask1] = 0

        #define custom regionprop
        def pixel_values(regionmask, intensity_image):
            return len(np.unique(intensity_image[regionmask]))

        #measure regionprops of cell mask w/labeled blob mask as intensity image
        stats1 = pd.DataFrame(regionprops_table(labeled,mask,properties=['label','area','intensity_image'],extra_properties=(pixel_values,)))

        #add area/cells to table
        stats1['area_per_cell'] = stats1['area']/stats1['pixel_values']

        #find large blob ids
        too_large = stats1[stats1['area_per_cell']>max_area].copy()
        too_large = too_large['label'].tolist()

        #remove large blobs
        for l in too_large:
            labeled[labeled==l] = 0
        b_img[labeled == 0] = 0
        
        #measure positive pixel ratio and append to table (PPR - positive pixel ratio)
        stats = pd.DataFrame(regionprops_table(mask.astype(int), b_img, properties = ['mean_intensity']))
        stats = stats.rename(columns={'mean_intensity':msk+'_'+m+'_PPR'})
        table = pd.concat([table, stats], axis=1)
        
        #measure positive pixel ratio and append to table (PPR - positive pixel ratio)
        stats = pd.DataFrame(regionprops_table(mask.astype(int), c_img, properties = ['mean_intensity']))
        stats = stats.rename(columns={'mean_intensity':msk+'_'+m+'_PPR_raw'})
        table = pd.concat([table, stats], axis=1)

        #measure fraction of pixels above max threshold (CPR - clipped pixel ratio)
        b_img = image[:,:,i].copy()
        b_img = (b_img > max_int).astype(np.int_)
        stats = pd.DataFrame(regionprops_table(mask.astype(int), b_img, properties = ['mean_intensity']))
        stats = stats.rename(columns={'mean_intensity':msk+'_'+m+'_CPR'})
        table = pd.concat([table, stats], axis=1)

    with open(os.path.join(args.output, args.name+'_THRESHOLDED.pkl'), 'wb') as handle:
        pickle.dump(table, handle)

    with open(os.path.join(args.output, args.name+'_METADATA.pkl'), 'wb') as handle:
        pickle.dump(threshold_data, handle)
print('Done!')