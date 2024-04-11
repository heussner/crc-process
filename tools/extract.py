import os
import tifffile
from skimage.measure import regionprops, regionprops_table
import pandas as pd
from matplotlib import pyplot as plt
from skimage.morphology import label, erosion, binary_erosion
from skimage.segmentation import find_boundaries
import numpy as np
from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity
from matplotlib.colors import Normalize
import math
import pickle
import argparse

parser = argparse.ArgumentParser(description='Extract annotated single cells for a single sample')
parser.add_argument('-i', '--image', help='Image file',required=True)
parser.add_argument('-s','--segmentation', help='Cell segmentation file', required=True)
parser.add_argument('-a','--annotation', help='Cell annotation file', required=True)
parser.add_argument('-t','--table', help='Thresholded feature table', required=True)
parser.add_argument('-m','--metadata', help='Image statistics', required=True)
parser.add_argument('-o', '--output', help='Output folder', required=True)
parser.add_argument('-n', '--name', help='Sample name',required=True)
args = parser.parse_args()

def view_selection(ax, x, y, distance, intensity_image, arrow_image, mask_image, cellID, index):
    """
    View the extracted cell

    Parameters:
    - ax: matplotlib axis
    - x (float): x-coordinate of cell
    - y (float): y-coordinate of cell
    - distance (int): 1/2 of image size
    - intensity_image (int array): MTI
    - arrow_image (int array): annotation image
    - mask_image (int array): segmentation mask
    - cellID (int): Cell ID segmentation instance
    - index (int): position in list of selected cells

    Returns:
    - matplotlib ax image of selected cell
    """
    x1,x2 = int(max(0,(x-distance))), int(max(0,(x+distance)))
    y1,y2 = int(max(0,(y-distance))), int(max(0,(y+distance)))
    arrow = arrow_image[x1:x2,y1:y2].copy()
    try:
        image = img_as_ubyte(intensity_image[:,x1:x2,y1:y2])
        new_image = image.copy()
        new_image[0,:,:] = rescale_intensity(image[0,:,:],in_range = (0,np.percentile(image[0,:,:],99.9)))
        new_image[1,:,:] = rescale_intensity(image[1,:,:],in_range = (0,np.percentile(image[1,:,:],99.9)))
        new_image[2,:,:] = rescale_intensity(image[2,:,:],in_range = (0,np.percentile(image[2,:,:],99.9)))
        mask = mask_image[x1:x2,y1:y2].copy()
        mask[mask!=cellID] = 0
        mask = find_boundaries(mask.astype(np.int32), mode='outer')
        mask[mask>0] = 1
        arrow[arrow>0] = 1
        overlay = mask + arrow
        norm = Normalize(vmin=0, vmax=1)
        ax.imshow(np.moveaxis(new_image,0,2), norm=norm)
        ax.imshow(overlay,alpha=0.5, norm=norm)
        ax.axis('off')
    except Exception as e: 
        print(e)

def view_options(ax, x,y,distance,intensity_image,arrow_image,mask_image,cellID,index):
    """
    View options for selection as a saved image

    Parameters:
    - x (float): x-coordinate of cell
    - y (float): y-coordinate of cell
    - distance (int): 1/2 of image size
    - intensity_image (int array): MTI
    - arrow_image (int array): annotation image
    - mask_image (int array): segmentation mask
    - cellID (int): Cell ID segmentation instance
    - path (str): Path to save image

    Returns:
    - None
    """
    x1,x2 = int(max(0,(x-distance))), int(max(0,(x+distance)))
    y1,y2 = int(max(0,(y-distance))), int(max(0,(y+distance)))
    arrow = arrow_image[x1:x2,y1:y2].copy()
    try:
        image = img_as_ubyte(intensity_image[:,x1:x2,y1:y2])
        new_image = image.copy()
        new_image[0,:,:] = rescale_intensity(image[0,:,:],in_range = (0,np.percentile(image[0,:,:],99.9)))
        new_image[1,:,:] = rescale_intensity(image[1,:,:],in_range = (0,np.percentile(image[1,:,:],99.9)))
        new_image[2,:,:] = rescale_intensity(image[2,:,:],in_range = (0,np.percentile(image[2,:,:],99.9)))
        mask = mask_image[x1:x2,y1:y2].copy()
        rps = regionprops(mask.astype(int))
        surroundings = mask.copy()
        mask[mask!=cellID] = 0
        mask = find_boundaries(mask.astype(np.int32), mode='outer')
        mask[mask>0] = 1
        arrow[arrow>0] = 1
        overlay = mask + arrow
        #fig, ax = plt.subplots()
        norm = Normalize(vmin=0, vmax=1)
        ax.imshow(surroundings)
        for r in rps:
            ax.text(r.centroid[1],r.centroid[0],r.label,fontsize=3)
        ax.axis('off')
    except Exception as e: 
        print(e)
        
image = tifffile.imread(args.image)
mask = tifffile.imread(args.segmentation)
annotations = tifffile.imread(args.annotation)
with open(args.table, 'rb') as handle:
    table = pickle.load(handle)

with open(args.metadata, 'rb') as handle:
    metadata = pickle.load(handle)

# Process annotations
arrow_copy = annotations.copy()
labels = label(annotations, connectivity=2)
labels = erosion(labels).astype(np.uint8)
annotation_table = pd.DataFrame(regionprops_table(labels, labels, properties=['label','centroid']))
with open(os.path.join(args.output,args.name+'_ANN.pkl'), 'wb') as handle:
    pickle.dump(annotation_table, handle)
tifffile.imwrite(os.path.join(args.output,'labeled_annotations.tif'),labels)

# Match annotations to cells based on shortest distance
cellID = []
flagged_ids = []
flagged_idxs = []
for idx, row in annotation_table.iterrows():
    #print(idx)
    temp = table.copy().reindex()
    x, y = temp['cell_centroid-0'].to_numpy(), temp['cell_centroid-1'].to_numpy()
    temp['distance'] = ((x - row['centroid-0'])**2 + (y - row['centroid-1'])**2)**0.5
    temp = temp[temp['distance']<50]
    dist_rank = temp['distance'].rank()
    print(dist_rank)
    CK_rank = np.exp(temp['cell_PanCK_mean_intensity'].rank(ascending=False))
    #print(CK_rank)
    rank_sum = dist_rank + CK_rank
    if len(rank_sum) == 0:
        cellID.append(table['cell_label'][0].item())
    else:
        index = np.argmin(rank_sum)
        print(index)
        if hasattr(index, "__len__"):
            CK = [CK_rank[i] for i in index]
            index = index[np.argmin(CK)]
        cellID.append(temp['cell_label'].to_numpy()[index])

# Visualize the selected cells
x = math.floor(math.sqrt(len(cellID)))
y = x + 2
fig, ax = plt.subplots(x,y)
ax = ax.flatten()
for i, id_ in enumerate(cellID):
    print(table[table['cell_label']==id_])
    view_selection(ax[i],table[table['cell_label']==id_]['cell_centroid-0'].item(), table[table['cell_label']==id_]['cell_centroid-1'].item(), 40, image, arrow_copy, mask, id_, i)
fig.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig(os.path.join(args.output,'selected_cells.png'),dpi=300)

fig, ax = plt.subplots(x,y)
ax = ax.flatten()
for i, id_ in enumerate(cellID):
    view_options(ax[i],annotation_table['centroid-0'][i].item(), annotation_table['centroid-1'][i].item(), 40, image, arrow_copy, mask, id_, i)
fig.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig(os.path.join(args.output,'options.png'),dpi=300)

for i in flagged_idxs:
    view_options(annotation_table['centroid-0'][i].item(), annotation_table['centroid-1'][i].item(), 40, image, arrow_copy, mask, cellID[int(i)], args.output, i)
    
label_dict = {"P":cellID}
with open(os.path.join(args.output,args.name+'_CALLOUTS.pkl'), 'wb') as handle:
    pickle.dump(label_dict, handle)

with open(os.path.join(args.output,args.name+'_FLAGGED.pkl'), 'wb') as handle:
    pickle.dump(flagged_ids, handle)
