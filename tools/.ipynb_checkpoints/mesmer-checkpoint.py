from tifffile import imread, imwrite
import argparse
import numpy as np
import os
from skimage.measure import regionprops_table
from skimage.morphology import dilation, disk
from skimage.segmentation import expand_labels
from skimage.exposure import rescale_intensity

parser = argparse.ArgumentParser(description='Run Mesmer on single sample')
parser.add_argument('-i', '--input', help='Input image file', required=True)
parser.add_argument('-o', '--output', help='Output directory', required=True)
parser.add_argument('-n', '--nuclear-channel', help='Nuclear marker channel index (starting from 0)', 
                    default=0, type=int)
parser.add_argument('-m', '--membrane-channel',help='Membrane marker channel index (starting from 0)', 
                    default=[1], type=int, nargs='+')
parser.add_argument('-p','--membrane-projection', help='Projection type to combine multiple membrane markers', 
                    default='max', type=str, choices=['max', 'mean'])
parser.add_argument('-c', '--compartment',help='Compartment to segment', choices=["whole-cell", "nuclear", "both"], 
                    default="both", type=str)
parser.add_argument('--mpp', help="Microns per pixel", default=0.325, type=float)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3,4,5,6,7"
im = imread(args.input)
print("Loaded image from {}".format(args.input))
im = np.squeeze(im)
print("Image shape: {}, dtype: {}".format(im.shape, im.dtype))

if len(im.shape) == 2:
    im = np.expand_dims(im, 0)

nuclear = im[args.nuclear_channel, :, :]
print("Loaded nuclear channel index {}".format(args.nuclear_channel))
print("Nuclear channel shape: {}".format(nuclear.shape))
if nuclear.ndim != 2:
    raise ValueError("Nuclear channel must be 2D, found shape {}".format(nuclear.shape))

membrane = im[args.membrane_channel, :, :]

print("Loaded membrane channel indices {}".format(args.membrane_channel))
print("Membrane channel shape: {}".format(membrane.shape))
if membrane.ndim != 3:
    raise ValueError("Membrane channel must be 3D, found shap {}".format(membrane.shape))

if membrane.shape[0] == 1:
    print("Found single membrane channel")
    membrane = np.squeeze(membrane)
elif args.membrane_projection == 'max':
    print("Applying max projection to membrane channels")
    membrane = np.max(membrane, axis=0)
elif args.membrane_projection == 'mean':
    print("Applying mean projection to membrane channels")
    membrane = np.mean(membrane, axis=0)

im = np.stack((nuclear, membrane), axis=-1)
im = np.expand_dims(im, 0)
print("Final input image shape: {}".format(im.shape))

print("Running Mesmer...")
from deepcell.applications import Mesmer
app = Mesmer()
labeled_image = app.predict(im, compartment=args.compartment, image_mpp=args.mpp, batch_size=1, postprocess_kwargs_whole_cell={'interior_threshold': 0.2},preprocess_kwargs={'threshold':True,'percentile':99.9,'normalize':True,'kernel_size':128})
labeled_image = np.squeeze(labeled_image)
print("Mesmer label image output shape: {}, dtype: {}".format(labeled_image.shape, labeled_image.dtype))
if args.compartment == "both":
    print("Mesmer label image found {} cells".format(np.max(labeled_image[:,:,0])))
    print("Mesmer label image found {} nuclei".format(np.max(labeled_image[:,:,1])))
else:
    print("Mesmer label image found {} cells".format(np.max(labeled_image)))
    
def refine_masks(mask_cell, mask_nuc, dilation_radius=3):
    #generate rim mask
    mask_rim_align = mask_cell.copy()
    mask_rim_align[(mask_cell > 0) & (mask_nuc > 0)] = 0

    #match nuclei mask
    mask_nuc_align = mask_cell - mask_rim_align #simply subtract cell - rim

    #add additional nuclei which do not have cell
    #find corresponding nuclei (use the original mask_nuc)
    stats = regionprops_table(mask_nuc, mask_cell > 0, properties=['coords','max_intensity'])
    id_ = np.where(stats['max_intensity'] == 0)[0] # find nuclei ID

    #generate nuc mask for missing cells (note that there is no overlap between this nuclei and cells)
    mask_nuc_wo_cell = np.zeros(mask_cell.shape)
    max_ID = np.amax(mask_cell) #find max cell label ID
    for j,nuc_id in enumerate(id_):
        region_coords = stats['coords'][nuc_id]
        region_coords = (region_coords[:,0], region_coords[:,1]) # [[0,0],[1,1]] -> ([0,1], [0,1])
        mask_nuc_wo_cell[region_coords] = j + 1 + max_ID   

    #generate cell mask from nuc mask by using simple dilation
    #mask_cell_add = dilation(mask_nuc_wo_cell, footprint=disk(radius=dilation_radius))
    mask_cell_add = expand_labels(mask_nuc_wo_cell, distance=dilation_radius)
    mask_cell_add[(mask_cell_add > 0) & (mask_cell > 0)] = 0 #remove overlapped area from the original mesmer mask
    
    mask_cell_final = mask_cell + mask_cell_add
    mask_nuc_final = mask_nuc_align + mask_nuc_wo_cell
    
    #remove cells without nuclei
    max_ID = np.amax(mask_nuc_final)
    mask_cell_final[mask_cell_final>max_ID] = 0
        
    return mask_cell_final, mask_nuc_final

if args.compartment == "both":
    print("Matching nucleus and cell masks...")
    cell, nuc = refine_masks(labeled_image[:,:,0], labeled_image[:,:,1], dilation_radius=3)
    save_file = os.path.join(args.output, args.input.split('/')[-1] + "_NUC__MESMER.tif")
    print("Saving cell output to {}".format(save_file))
    imwrite(save_file, nuc)
    save_file = os.path.join(args.output, args.input.split('/')[-1] + "_CELL__MESMER.tif")
    print("Saving nucleus output to {}".format(save_file))
    imwrite(save_file, cell)
elif args.compartment == "nuclear":
    save_file = os.path.join(args.output, args.input.split('/')[-1] + "_NUC__MESMER.tif")
    print("Saving nuclei output to {}".format(save_file))
    imwrite(save_file, labeled_image)
else:
    save_file = os.path.join(args.output, args.input.split('/')[-1] + "_CELL__MESMER.tif")
    print("Saving cell output to {}".format(save_file))
    imwrite(save_file, labeled_image)
print("Done!")
