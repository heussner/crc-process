import os
import random
import numpy as np
from PIL import Image
from skimage.transform import resize
from tifffile import imread, imwrite
from skimage import img_as_ubyte
from typing import Any, Tuple, Text, List
import argparse
import pandas as pd


def cloud_2_grid(coords: np.ndarray, gridw: int, gridh: int):
    nx = coords[:, 0]
    ny = coords[:, 1]
    nx = nx - nx.min()
    ny = ny - ny.min()
    nx = gridw * nx // nx.max()
    ny = gridh * ny // ny.max()
    nc = np.column_stack((nx, ny))
    return nc


def sample(
    files: List[Text], coords: np.ndarray, random_samp: bool = True, n_plot: int = None
) -> Tuple[np.ndarray, List[Text]]:
    if n_plot == -1:
        n_plot = len(files)
    elif n_plot > len(files):
        n_plot = len(files)

    if random_samp:
        smpl = random.sample(range(len(files)), n_plot)
    else:
        smpl = [i for i in range(n_plot)]

    files = [files[s] for s in smpl]
    coords = coords[smpl, :]
    return coords, files


def rescale(coords: np.ndarray) -> np.ndarray:
    for i in range(2):
        coords[:, i] = coords[:, i] - coords[:, i].min()
        coords[:, i] = coords[:, i] / coords[:, i].max()
    return coords

# def load_and_size_coord_tile(f: Text, tile_size: int, channel: int, crop: int) -> Image:
#     img = np.asarray(imread(f)[:, :, :3])
#     img = img - np.min(img)
#     npi = (img * 255 / np.max(img)).astype("uint8")
#     npi = np.array(img, np.uint8)
#     if crop != 0:
#         npi = npi[crop:-crop, crop:-crop, :]
#     if channel != -1:
#         npi = npi[:, :, channel]
#     rsz = resize(npi, (tile_size, tile_size), mode="constant")
#     npi = (2 ** 8 - 1) * rsz / rsz.max()
#     npi = npi.astype(np.uint8)
#     img = Image.fromarray(npi)
#     return img

# def create_coord_plot(
#     files: list,
#     xs: np.ndarray,
#     ys: np.ndarray,
#     plot_h: int,
#     plot_w: int,
#     tile_size: int,
#     plot_dir: Text,
#     channel: int,
#     crop: int,
# ) -> None:
#     full_image = Image.new("RGB", (plot_w, plot_h))

#     for f, x, y in zip(files, xs, ys):
#         img = load_and_size_coord_tile(f, tile_size, channel, crop)
#         x_coord = int((plot_w - tile_size) * x)
#         y_coord = int((plot_h - tile_size) * y)
#         full_image.paste(img, (x_coord, y_coord))
#         img.close()

#     full_image.save(os.path.join(plot_dir, f"coordplot_c{channel}.png"))
#     full_image.close()


def create_grid_plot(
    files: List[Text],
    coords: np.ndarray,
    plot_w: int,
    plot_h: int,
    tile_size: int,
    plot_dir: Text,
    channel: int,
    crop: int,
    type: str,
) -> None:

    grid_assignment = cloud_2_grid(coords, plot_w / tile_size, plot_h / tile_size)
    grid_assignment = grid_assignment * tile_size
    #grid_image = Image.new("L", (plot_w, plot_h))
    grid_image = np.zeros((plot_h, plot_w), dtype=np.uint8)

    skipped = 0
    for f, grid_pos in zip(files, grid_assignment):
        x, y = grid_pos
        x = int(x)
        y = int(y)
        if x + tile_size >= plot_w or y + tile_size >= plot_h:
            skipped += 1
            continue
        tile = imread(f)
        if channel != -1 or crop != 0:
            tile = tile[crop:-crop, crop:-crop, channel]
        tile = resize(tile, (tile_size, tile_size), anti_aliasing=True)
        tile = img_as_ubyte(tile)
        #tile = Image.fromarray(tile).convert("L")
        #tile = tile.resize((tile_size, tile_size), Image.ANTIALIAS)
        grid_image[y:y + tile_size, x:x + tile_size] = tile
        #grid_image.paste(tile, (int(x), int(y)))
        #tile.close()
    print(f"Skipped {skipped} tiles")

    #grid_im_np = np.array(grid_image)
    imwrite(os.path.join(plot_dir, f"gridplot_{type}_c{channel}.tiff"), grid_image)
    #grid_image.save(os.path.join(plot_dir, f"gridplot_{type}_c{channel}_pil.tiff"))
    #grid_image.close()


def plot(
    files: List[Text],
    type: str,
    coords: np.ndarray,
    target_dir: Text,
    n_plot: int = -1,
    random_sample: bool = True,
    coord_plot: bool = False,
    grid_plot: bool = True,
    tile_size: int = 100,
    grid_plot_dim: int = 5000,
    coord_plot_dim: int = 5000,
    channels: List = [],
    crop: int = 60,
) -> None:

    coords, files = sample(files, coords, random_sample, n_plot)
    coords = rescale(coords)
    xs = coords[:, 0]
    ys = coords[:, 1]

    for c in channels:

        # if coord_plot:
        #     create_coord_plot(files, xs, ys, coord_plot_dim, coord_plot_dim, tile_size, target_dir, c, crop)

        if grid_plot:
            create_grid_plot(files, coords, grid_plot_dim, grid_plot_dim, tile_size, target_dir, c, crop, type)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to csv file containing embedding and corresponding filenames",
    )
    parser.add_argument(
        "--write_dir",
        type=str,
        required=True,
        help="Path to target directory for writing output plot files",
    )
    parser.add_argument(
        "--sample",
        type=str,
        required=False,
        help="Sample identifier to select relevant embeddings and name output files"
    )
    parser.add_argument(
        "--coord_plot", type=int, default=0, help="Create coordinate plot or not"
    )
    parser.add_argument(
        "--grid_plot", type=int, default=1, help="Create grid plot or not"
    )
    parser.add_argument(
        "--n_img_max",
        type=int,
        default=-1,
        help="Max number of image tiles to plot. Set to -1 for no maximum.",
    )
    parser.add_argument(
        "--random_sample",
        type=int,
        default=1,
        help="Randomly sample images to tile and plot",
    )
    parser.add_argument(
        "--grid_plot_dim",
        type=int,
        default=5000,
        help="Height and width of grid plot."
    )
    parser.add_argument(
        "--coord_plot_dim",
        type=int,
        default=5000,
        help="Height and width of coord plot."
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Image channels to plot."
    )
    parser.add_argument(
        '--comb_scenes', 
        help='combine different scenes from same patient into single plot', 
        action='store_true'
    )

    args = parser.parse_args()

    if not os.path.isfile(args.data_file):
        raise argparse.ArgumentError("Provided data file does not exist")

    if not os.path.isdir(args.write_dir):
        os.makedirs(args.write_dir)

    data = pd.read_csv(args.data_file)
    t = args.data_file.split("/")[-1].split(".")[0]
    data["samples"] = [f.split("/")[-2] for f in data["files"].tolist()]
    # data["samples"] = [f.split("/")[-1].split("_")[0] for f in data["files"].tolist()]
    if args.sample is not None:
        if args.comb_scenes:
            data["patient"] = [s.split("/")[-1].split("-Scene")[0] for s in data["samples"].tolist()]
            data = data[data["patient"] == args.sample]
        else:
            # args.sample = args.sample.replace("2021_10_07__", "").replace("R2_ECAD.CD45.panCK.Epcam_", "").replace("mix", "").split("-Scan")[0]
            data = data[data["samples"] == args.sample]
    print(f'Plotting {len(data.index)} embeddings')
    files = list(data["files"])
    xs = np.array(data["x"])
    ys = np.array(data["y"])
    coords = np.stack((xs, ys), axis=-1)


    plot(
        files=files,
        type=t,
        coords=coords,
        target_dir=args.write_dir,
        n_plot=args.n_img_max,
        random_sample=args.random_sample,
        coord_plot=args.coord_plot,
        grid_plot=args.grid_plot,
        grid_plot_dim=args.grid_plot_dim,
        channels=args.channels,
    )
