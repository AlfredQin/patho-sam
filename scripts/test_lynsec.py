
import os
from tqdm import tqdm
from glob import glob
from natsort import natsorted

import h5py
import numpy as np
import imageio.v3 as imageio
from typing import Literal
from torch_em.data.datasets.histopathology import pannuke

from micro_sam.util import get_sam_model
from micro_sam.evaluation import inference, evaluation
from torch_em.data.datasets.histopathology.lynsec import get_lynsec_paths

from scripts.utils import _pad_image


def run_interactive_segmentation(
        input_path, experiment_folder, model_type, start_with_box_prompt=True,
        choice: Literal['ihc', 'h&e'] = None,
):
    # Create clone of single images in input_path directory.
    data_dir = os.path.join(input_path, f"benchmark_2d/{choice}")

    if not (os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0):
        # First, we download the lizard datasets
        raw_paths, label_paths = get_lynsec_paths(path=input_path, download=True, choice=choice)

        # Then, extract image and gt mask from each h5 file
        # Store them one-by-one locally in an experiment folder.
        os.makedirs(data_dir, exist_ok=True)
        for idx, (image_path, label_path) in tqdm(enumerate(zip(raw_paths, label_paths))):
            image, label = imageio.imread(image_path), imageio.imread(label_path)
            # There has to be some foreground in the image to be considered for interactive segmentation.
            if len(np.unique(label)) == 1:
                continue
            image, label = _pad_image(image, (512, 512)), _pad_image(label, (512, 512))
            prefix = os.path.basename(image_path).split(".")[0]
            imageio.imwrite(os.path.join(data_dir, f"{idx:03d}_image.tif"), image, compression="zlib")
            imageio.imwrite(os.path.join(data_dir, f"{idx:03d}_mask.tif"), label, compression="zlib")

    # Well, now we have our image and label paths.
    image_paths = natsorted(glob(os.path.join(data_dir, "*_image.tif")))
    label_paths = natsorted(glob(os.path.join(data_dir, "*_mask.tif")))

    assert len(image_paths) == len(label_paths)

    # Now that we have the data ready, run interactive segmentation.

    # Get the Segment Anything model.
    predictor = get_sam_model(model_type=model_type)

    # Then run interactive segmentation by simulating prompts from labels.
    prediction_root = os.path.join(
        experiment_folder, ("start_with_box" if start_with_box_prompt else "start_with_point")
    )
    inference.run_inference_with_iterative_prompting(
        predictor=predictor,
        image_paths=image_paths,
        gt_paths=label_paths,
        embedding_dir=None,
        prediction_dir=prediction_root,
        start_with_box_prompt=start_with_box_prompt,
    )

    # And evaluate the results.
    results = evaluation.run_evaluation_for_iterative_prompting(
        gt_paths=label_paths,
        prediction_root=prediction_root,
        experiment_folder=experiment_folder,
        start_with_box_prompt=start_with_box_prompt,
    )

    print(results)


def main(args):
    run_interactive_segmentation(
        input_path=args.input_path,
        model_type=args.model_type,
        experiment_folder=args.experiment_folder,
        start_with_box_prompt=args.start_with_box_prompt,
        choice=args.choice,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", default="/home/qinc/Dataset/Microscopy/Histopathology/LyNSeC/", type=str)
    parser.add_argument("-e", "--experiment_folder", default="./experiments/lynsec/ihc", type=str)
    parser.add_argument("-m", "--model_type", default="vit_b_histopathology", type=str)
    parser.add_argument("--start_with_box_prompt", action="store_true", help="Whether to start with box prompt")
    parser.add_argument(
        "--choice", type=str, choices=['ihc', 'h&e'], default='ihc',
        help="Choice of dataset to use for interactive segmentation, either 'ihc' or 'h&e'."
    )
    args = parser.parse_args()
    main(args)
