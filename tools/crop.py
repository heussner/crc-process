import os
from copy import deepcopy
from cv2 import resize
from tifffile import imread, imwrite
from skimage.measure import regionprops, label
from skimage.transform import rotate
from typing import Any, Tuple, Text, List, Union
import numpy as np
from glob import glob
import argparse
from random import shuffle, sample
import pickle


def segmentation_crops_to_disk(
    segmentation_path: Text,
    mti_path: Text,
    save_dir: Text,
    save_prefix: Text,
    fix_orientation: bool = False,
    crop_length: int = 64,
    dtype: Text = "uint16",
    labels=None,
    arrange=None,
    min_max=False,
    robust_saturation=False,
    saturation=False,
    max_cells=None
) -> None:

    """
    Extract cell segmentation instances, crop, and write crops to disk.

    Args:
        segmentation_path (Text): Path to whole cell segmentation. File should contain np.ndarray with shape [H, W]
        mti_path (Text): Path to multiplex tissue imaging data file or directory. If file, should contain np.ndarray with shape [H, W, C (optional)]. If directory, each file should contain np.ndarray with shape [H, W]. In both cases expect dtype uint16 or int16.
        save_dir (Text): Directory in which to save cropped single cell segmentations.
        save_prefix (Text): Prefix string for save single cell image filenames
        fix_orientation (bool, optional): If True, rotate crops to a fixed major-axis orientation of 0 radians. Defaults to True
        crop_length (int, optional): Length of each side of the cropped object regions. Defaults to 64.
        dtype (Text, optional): Desired dtype of the cropped regions when saved to disk. Defaults to uint16.
        contrast_limits (Union[Tuple[float, float], Tuple[List[float], List[float]]], optional): Lower and upper percentiles at which to saturate MTI data channel-wise. Defaults to (0.0, 99.9)

    Raises:
        FileNotFoundError: Provided file paths should point to existing files or directories.
    """
    segmentation = load_segmentation(segmentation_path)

    mti = load_mti(mti_path)
    mti = as_dtype(mti, dtype)
    
    if args.arrange != None:
        mti = order_channels(markers_df, mti)
    
    if saturation:
        mti = saturate(mti)
    elif robust_saturation:
        print("Applying robust saturation")
        mti = robust_saturate(mti, segmentation)
    else:   
        print("Skipping contrast limits adjustment...")
    
    if min_max:
        print("Min-max scaling")
        mti = min_max_scale(mti)
    else:
        print("Skipping min-max scaling..")

    check_shapes(mti.shape, segmentation.shape)
    make_save_dir(save_dir)
    
    if args.labels != None:
        with open(args.labels,'rb') as handle:
            label_dict = pickle.load(handle)
        #reduce number of negative control cells
        label_dict['N'] = sample(label_dict['N'],min(2000,len(label_dict['N'])))
        
        #crop_length = label_dict['crop_length']
        #print(f'Crop length set to {crop_length}')
        for name, labels in label_dict.items():
            seg = segmentation.copy()
            mask = np.isin(seg, labels)
            seg[~mask] = 0
            rps = regionprops(seg.astype(np.uint32), mti)
            shuffle(rps)
            num_saved = 0
            total_num = len(rps)
            n_too_large = 0
            n_at_edge = 0
            
            for i, r in enumerate(rps):
                if at_img_edge(r, seg.shape, crop_length) == True:
                    n_at_edge += 1
                    continue

                elif exceeds_crop_length(r, crop_length) == True:
                    n_too_large += 1
                    continue
                
                else:
                    crop = crop_mti_cell_instance(r, mti, seg, crop_length, fix_orientation)

                    rank = labels.index(r.label)

                    save_crop(crop, save_dir, save_prefix, name, rank, r.label, r.centroid)

                    num_saved += 1

                    if num_saved == max_cells:
                        break
            print(
            f"Finished cropping. Saved {num_saved} of {total_num} possible of {name}"
            f" segmentation instances in {save_dir}. Crops saved as {dtype} dtype."
            f" Found and ignored {n_too_large} instances too large for the crop size,"
            f" and {n_at_edge} instances at the image edge. Number of crops limited by"
            f" user provided 'max_cells'={max_cells}."
            )
            
    else:

        rps = regionprops(segmentation.astype(np.uint32), mti)
        shuffle(rps)
        print("Extracted image region properties. Found {} cells".format(len(rps)))

        num_saved = 0
        total_num = len(rps)
        n_too_large = 0
        n_at_edge = 0

        print("Beginning crops...")

        for i, r in enumerate(rps):
            if at_img_edge(r, segmentation.shape, crop_length):
                n_at_edge += 1
                continue

            if exceeds_crop_length(r, crop_length):
                n_too_large += 1
                continue

            crop = crop_mti_cell_instance(r, mti, segmentation, crop_length, fix_orientation)

            # if fix_orientation:
            #     crop = zero_orient(crop, radians=r.orientation)

            save_crop(crop, save_dir, save_prefix, r.label, r.centroid)

            num_saved += 1

            if num_saved == max_cells:
                break

        print(
            f"Finished cropping. Saved {num_saved} of {total_num} possible"
            f" segmentation instances in {save_dir}. Crops saved as {dtype} dtype."
            f" Found and ignored {n_too_large} instances too large for the crop size,"
            f" and {n_at_edge} instances at the image edge. Number of crops limited by"
            f" user provided 'max_cells'={max_cells}."
        )


def load_mti(mti_path: Text,) -> Tuple[np.ndarray, Any]:
    """Load MTI data from provided file or directory and return MTI path suffix

    Args:
        mti_path (Text): Path to file or directory containing MTI data

    Raises:
        FileNotFoundError: Provided MTI data file or folder must exist.

    Returns:
        np.ndarray: Loaded mti data as np.ndarray
    """
    if os.path.isfile(mti_path):
        mti: np.ndarray = imread(mti_path).squeeze()
        mti = np.transpose(mti, (1, 2, 0))
        if len(mti.shape) == 2:
            mti = mti[None, :, :]
        elif len(mti.shape) != 3:
            raise ValueError(
                f"Loaded MTI data from {mti_path} is inappropriate shape. Found"
                f" {mti.shape}. Should be [H, W] or [H, W, C]."
            )
    elif os.path.isdir(mti_path):
        mti_files = glob(os.path.join(mti_path, "*"))
        mti: np.ndarray = load_mti_stack(mti_files)
    else:
        raise FileNotFoundError(
            "Provided 'mti_path' does not exist as file or directory."
        )

    print(
        f"Loaded MTI data from {mti_path}, Shape: {mti.shape}, dtype: {mti.dtype}, channel minimums/maximums: {mti.min((0, 1))} / {mti.max((0, 1))}"
    )
    return mti


def load_segmentation(segmentation_path: Text) -> np.ndarray:
    """[summary]

    Args:
        segmentation_path (Text): Path to file containing whole cell segmentation.

    Raises:
        FileNotFoundError: Provided segmentation file must exist.
        ValueError: Segmentation data should contain two dimensions, [H, W]

    Returns:
        np.ndarray: Whole cell instance segmentation as 2D np.ndarray
    """
    if not os.path.isfile(segmentation_path):
        raise FileNotFoundError("Provided 'segmentation_path' file does not exist")
    else:
        segmentation: np.ndarray = imread(segmentation_path).squeeze()
        if len(segmentation.shape) != 2:
            raise ValueError(
                "Segmentation shape indicates > 2 dimensions. Should be [H, W]."
            )

    print(
        f"Loaded segmentation from {segmentation_path}, Shape:"
        f" {segmentation.shape}, dtype: {segmentation.dtype}"
    )
    return segmentation


def load_mti_stack(mti_files: List[Text]) -> np.ndarray:
    """Build np.ndarray stack of MTI channels from list of files

    Args:
        mti_files (List[Text]): List of MTI channel files

    Raises:
        ValueError: The np.ndarray found in each of the provided files should have matching dimensions

    Returns:
        np.ndarray: np.ndarray stack of MTI channels
    """
    channels = []
    dims = None
    for f in mti_files:
        c = imread(f).squeeze()
        if len(c.shape) != 2:
            raise ValueError(f"MTI channel should have shape [H, W]. Found: {c.shape}")
        if not dims:
            dims = c.shape
        elif dims != c.shape:
            raise ValueError(
                "MTI data channels do not share common dimensions and cannot"
                " be stacked."
            )
        channels.append(c)
        dims = c.shape
    stack = np.stack(channels, axis=-1)
    return stack


def crop_mti_cell_instance(
    region: Any, mti: np.ndarray, segmentation: np.ndarray, crop_length: int, fix_orientation: bool
) -> np.ndarray:
    """Crop in individual cell isntance from provided MTI np.ndarray

    Args:
        region (Any): skimage.measure.RegionProperties instance
        mti (np.ndarray): MTI data
        segmentation (np.ndarray): Segmentation data
        crop_length (int): Length of each side of the cropped object regions.

    Raises:
        ValueError: MTI data and segmentation dimensions must match.

    Returns:
        np.ndarray: Cropped cell instance
    """

    if crop_length % 2 != 0:
        raise ValueError("Parameter 'crop_length' must be divisible by 2")

    y0, x0, y1, x1 = region.bbox
    crop = deepcopy(mti[y0:y1, x0:x1, :])
    seg_crop = segmentation[y0:y1, x0:x1]
    crop *= (seg_crop == region.label)[:, :, None]
    if fix_orientation:
        crop = zero_orient(crop, region.orientation)
    h, w, c = crop.shape
    dif_h, dif_w = (crop_length - h), (crop_length - w)
    assert dif_h >= 0, f"Crop length: {crop_length}, img height: {h}"
    assert dif_w >= 0, f"Crop length: {crop_length}, img width: {w}"
    h_addone, w_addone = dif_h % 2 != 0, dif_w % 2 != 0
    h_delta, w_delta = dif_h // 2, dif_w // 2
    crop = np.pad(
        crop,
        pad_width=(
            (h_delta, h_delta + int(h_addone)),
            (w_delta, w_delta + int(w_addone)),
            (0, 0),
        ),
        mode="constant",
    )
    return crop




def at_img_edge(region: Any, img_h: int, img_w: int) -> bool:
    """Determine if instance region overlaps with image boundary

    Args:
        region (Any): skimage.measure.RegionProperties instance
        img_h (int): Data image height.
        img_w (int): Data image width

    Returns:
        bool: True if instance overlaps image boundary, else False.
    """
    min_r, min_c, max_r, max_c = region.bbox
    if (min_r == 0) or (min_c == 0) or (max_r == img_h) or (max_c == img_w):
        return True
    else:
        return False


def check_shapes(mti_shape: Tuple, segmentation_shape: Tuple) -> None:
    """Confirm MTI and segmentation arrays match shape

    Args:
        mti_shape (Tuple):
        segmentation_shape (Tuple):

    Raises:
        ValueError: Shapes should match except in the channel dimension
    """
    if mti_shape[:2] != segmentation_shape:
        raise ValueError(
            "MTI and segmentation data have different shapes. To resolve this"
            " error ensure the shape of MTI is [C, H, W] and segmentation is [H,"
            f" W] match in both arrays. Segmentation was {segmentation_shape},"
            f" MTI was {mti_shape}"
        )


def make_save_dir(save_dir: Text) -> None:
    """Create save directory if it does not exist

    Args:
        save_dir (Text): Directory to make
    """
    if not os.path.isdir(save_dir):
        print(f"Creating save directory at: {save_dir}")
        os.makedirs(save_dir)


def as_dtype(crop: np.ndarray, dtype: Text) -> np.ndarray:
    """Change crop to specified dtype

    Args:
        crop (np.ndarray): Crop to adjust
        dtype (Text): Target dtype. Supported types are [uint8, uint16]

    Raises:
        ValueError: dtype must be one of the support types

    Returns:
        np.ndarray: Crop array as 'dtype'
    """
    if dtype not in ["uint8", "uint16"]:
        raise ValueError(f"Found dtype of {dtype}. Expected one of [uint8, uint16].")
    if crop.dtype == dtype:
        return crop

    orig_dtype = crop.dtype
    info = np.iinfo(crop.dtype)
    crop = crop.astype(np.float64) / info.max
    scale = (2 ** 8) if dtype == "uint8" else (2 ** 16)
    scale -= 1
    crop = scale * crop
    crop = crop.astype(dtype)
    print(f"MTI data converted from {orig_dtype} to {dtype}")

def robust_saturate(
    mti: np.ndarray,
    seg: np.ndarray,
) -> np.ndarray:
    """Saturate provided data array channelwise according to

    Args:
        mti (np.ndarray): Data array to saturate channel-wise
        lower_percentile (Union[List[float], float]): Channel specific lower percentile(s) to saturate. If float apply value to all channels. Defaults to 0.0.
        upper_percentile (Union[List[float], float]): Channel specific upper percentile(s) to saturate. If float apply value to all channels. Defaults to 99.9.

    Returns:
        np.ndarray: lower and upper percentile saturated MTI data array
    """
    dtype = mti.dtype
    scale = (2 ** 8) if dtype == "uint8" else (2 ** 16)
    
    upper_vals, lower_vals = [], []
    for i in range(mti.shape[2]):
        
        fgd = np.where(seg.astype(int).copy()==0, np.nan, mti[:,:,i].copy())
        mu_fgd = np.mean(fgd[~np.isnan(fgd)])
        std_fgd = np.std(fgd[~np.isnan(fgd)])
        bgd = np.where(seg.astype(int).copy()!=0, np.nan, mti[:,:,i].copy())
        mu_bgd = np.mean(bgd[~np.isnan(bgd)])
        std_bgd = np.std(bgd[~np.isnan(bgd)])
        
        lower = int(max(0, mu_bgd - 3 * std_bgd))
        upper = int(min(scale, 1.5*(mu_fgd + 3 * std_fgd)))
        upper_vals.append(upper)
        lower_vals.append(lower)
        mti[:, :, i][mti[:, :, i] <= lower] = lower
        mti[:, :, i][mti[:, :, i] >= upper] = upper
    print(
        f"MTI data saturated channel-wise. Actual values were {upper_vals},"
        f" {lower_vals}"
    )
    print(
        f"After setting contrast limits, channel minimums/maximums: {mti.min((0, 1))} / {mti.max((0, 1))}"
    )
    return mti

def saturate(
    mti: np.ndarray,
) -> np.ndarray:
    """Saturate provided data array channelwise according to

    Args:
        mti (np.ndarray): Data array to saturate channel-wise
        lower_percentile (Union[List[float], float]): Channel specific lower percentile(s) to saturate. If float apply value to all channels. Defaults to 0.0.
        upper_percentile (Union[List[float], float]): Channel specific upper percentile(s) to saturate. If float apply value to all channels. Defaults to 99.9.

    Returns:
        np.ndarray: lower and upper percentile saturated MTI data array
    """
    lower_percentile=0.00
    upper_percentile=99.9
    if type(lower_percentile) == float:
        lower_percentile = [lower_percentile for i in range(mti.shape[2])]
    if type(upper_percentile) == float:
        upper_percentile = [upper_percentile for i in range(mti.shape[2])]

    if (len(lower_percentile) != mti.shape[2]) or (
        len(upper_percentile) != mti.shape[2]
    ):
        raise ValueError(
            "If supplying lower and upper percentile values as a List[float] type"
            " ensure the length matches number of MTI channels. Length"
            f" lower_percentile and upper_percentile: {len(lower_percentile)},"
            f" {len(upper_percentile)}. MTI shape was {mti.shape}"
        )

    upper_vals, lower_vals = [], []
    for i in range(mti.shape[2]):
        lower = int(np.percentile(mti[:, :, i], lower_percentile[i]))
        upper = int(np.percentile(mti[:, :, i], upper_percentile[i]))
        upper_vals.append(upper)
        lower_vals.append(lower)
        mti[:, :, i][mti[:, :, i] <= lower] = lower
        mti[:, :, i][mti[:, :, i] >= upper] = upper
    print(
        f"MTI data saturated channel-wise with lower {lower_percentile} and upper"
        f" {upper_percentile} percentiles. Actual values were {upper_vals},"
        f" {lower_vals}"
    )
    print(
        f"After setting contrast limits, channel minimums/maximums: {mti.min((0, 1))} / {mti.max((0, 1))}"
    )
    return mti


def min_max_scale(mti: np.ndarray) -> np.ndarray:
    """Min-max channel-wise scaling

    Args:
        mti (np.ndarray): Data array to scale

    Returns:
        np.ndarray: Channel-wise scaled data array.
    """
    dtype = mti.dtype
    scale = (2 ** 8) if dtype == "uint8" else (2 ** 16)
    scale -= 1
    mti = mti.astype(np.float64)
    max_vals = []
    min_vals = []
    for i in range(mti.shape[2]):
        channel_max = np.max(mti[:, :, i])
        channel_min = np.min(mti[:, :, i])
        min_vals.append(channel_min)
        max_vals.append(channel_max)
        mti[:, :, i] = (mti[:, :, i] - channel_min) / (channel_max - channel_min)
    print(
        "MTI data min-max scaled channel-wise with discovered minimums and"
        f" maximums: {min_vals}, {max_vals}"
    )
    mti *= scale
    mti = mti.astype(dtype)
    return mti


def save_crop(
    crop: np.ndarray,
    save_dir: Text,
    save_prefix: Text,
    class_label: Text,
    rank: int,
    region_lab: int,
    region_centroid: Tuple[float, float],
):
    """Save crop array to disk

    Args:
        crop (np.ndarray): Crop array to save
        save_dir (Text): Directory to save in
        save_prefix (Text): Save file name prefix
        class_label (Text): P - Positive CHC, G - Grey area CHC, N - Negative control
        region_lab (int): Label of region inside crop
        region_centroid (Tuple[float, float]): Centroid of region inside crop
    """
    file_name = (
        save_prefix
        + f"_CLASS_{class_label}_RANK_{rank}_REG_LABEL_{region_lab}_CENTROID_{int(region_centroid[0])}_{int(region_centroid[1])}.tiff"
    )
    save_path = os.path.join(save_dir, file_name)
    imwrite(save_path, crop)


def exceeds_crop_length(region: Any, crop_length: int) -> bool:
    """Determine if instance region is larger than specified crop length

    Args:
        region (Any): skimage.measure.RegionProperties instance
        crop_length (int): Length of desired region crops

    Returns:
        bool: True if larger else false
    """
    mal = region.major_axis_length
    min_r, min_c, max_r, max_c = region.bbox
    h = max_r - min_r
    w = max_c - min_c
    if max(h, w, mal) > crop_length:
         return True
    else:
         return False


def zero_orient(crop: np.ndarray, radians: float) -> np.ndarray:
    """Orient provided data crop array to 0 relative to array 0th (row) axis

    Args:
        crop (np.ndarray): MTI data crop to orient
        radians (float): Current crop orientation in radians. See https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops.

    Returns:
        np.ndarray: Oriented MTI data crop
    """
    degrees = radians * (180 / np.pi) + 90
    rotated = rotate(
        crop, angle=(-degrees), mode="constant", preserve_range=True, order=0, resize=True
    )
    rotated = rotated.astype(crop.dtype)
    return rotated

def arrange_channels(markers_df, mti):
    #hard-coded channel order
    order = {'CD45':0,'CK':1,'DAPI':2}
    ordered_mti = mti.deepcopy()
    markers = zip(markers_df["channel"].tolist(),markers_df["marker_name"].tolist())
    markers = sorted(markers, key=lambda x: x[1])
    
    for i, m in markers:
        if i != order[m]:
            ordered_mti[:,:,truth[m]] = mti[:,:,i].copy()

    return ordered_mti


def main(
    segmentation_path,
    mti_path,
    save_dir,
    save_prefix,
    fix_orientation,
    dtype,
    labels,
    min_max,
    robust_saturation,
    saturation,
    crop_length=None,
    max_cells=None,
):

    segmentation_crops_to_disk(
        segmentation_path,
        mti_path,
        save_dir,
        save_prefix=save_prefix,
        fix_orientation=fix_orientation,
        crop_length=crop_length,
        dtype=dtype,
        labels=labels,
        arrange=arrange,
        min_max=min_max,
        robust_saturation=robust_saturation,
        saturation=saturation,
        max_cells=max_cells
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--segmentation_path", type=str, required=True, help="Path to segmentation file"
    )
    parser.add_argument(
        "--mti_path",
        type=str,
        required=True,
        help="Path to directory or file containing MTI data for cropping",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Path to directory to save cropped data in",
    )
    parser.add_argument(
        "--save_prefix",
        type=str,
        required=True,
        help="Prefix string for cropped image file names",
    )
    parser.add_argument(
        "--fix_orientation",
        type=int,
        default=0,
        help="Fix orientation of crops wrt major axis",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["uint16", "uint8"],
        default="uint16",
        help="Data type to save cropped images with (uint8 or uint16)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Path to dict of specific labels to crop",
    )
    parser.add_argument(
        "--arrange",
        type=str,
        default=None,
        help="Path to markers file",
    )
    parser.add_argument(
        "--crop_length",
        default=64,
        type=int,
        help="Crop dimension. If not provided will be computed. See data.data_bbox_max",
    )
    parser.add_argument(
        "--min_max",
        action='store_true',
        help="Min-max scale the MTI image channel-wise before cropping",
    )

    def tuple_type(strings):
        strings = strings.replace("(", "").replace(")", "")
        mapped_float = map(float, strings.split(","))
        return tuple(mapped_float)

    parser.add_argument(
        "--saturation",
        action='store_true',
        help="Saturate upper and lower intensity percentiles channel-wise before cropping",
    )
    parser.add_argument(
        "--robust_saturation",
        action="store_true",
        help="Robustly saturate image channel-wise before cropping",
    )
    parser.add_argument(
        "--max_cells",
        type=int,
        default=None,
        help="Max number of cells to crop from image. If none, crop all."
    )

    args = parser.parse_args()

    main(
        args.segmentation_path,
        args.mti_path,
        args.save_dir,
        args.save_prefix,
        bool(args.fix_orientation),
        args.dtype,
        args.labels,
        args.arrange,
        args.min_max,
        args.robust_saturation,
        args.saturation,
        args.crop_length,
        args.max_cells
    )
