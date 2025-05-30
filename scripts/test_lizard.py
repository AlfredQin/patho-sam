import os
from tqdm import tqdm
from glob import glob
from natsort import natsorted

import h5py
import numpy as np
import imageio.v3 as imageio

from torch_em.data.datasets.histopathology import pannuke

from micro_sam.util import get_sam_model
from micro_sam.evaluation import inference, evaluation
from torch_em.data.datasets.histopathology.lizard import get_lizard_data


def _pad_image(image, target_shape=(512, 512), pad_value=0):
    pad = [
        (max(t - s, 0) // 2, max(t - s, 0) - max(t - s, 0) // 2) for s, t in zip(image.shape[:2], target_shape)
    ]

    if image.ndim == 3:
        pad.append((0, 0))

    return np.pad(image, pad, mode="constant", constant_values=pad_value)


def run_interactive_segmentation(input_path, experiment_folder, model_type, split="test", start_with_box_prompt=True,
                                 batch_size: int = 32):

    # Create clone of single images in input_path directory.
    data_dir = os.path.join(input_path, f"benchmark_2d/{split}")

    if not (os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0):
        # First, we download the lizard datasets
        get_lizard_data(path=input_path, download=True, split="train")

        # Next, we get all h5 files under the folder
        folder = os.path.join(input_path, split)
        file_paths = glob(os.path.join(folder, "*.h5"))
        if len(file_paths) == 0:
            raise ValueError(f"No h5 files found in {folder}")

        # Then, extract image and gt mask from each h5 file
        # Store them one-by-one locally in an experiment folder.
        os.makedirs(data_dir, exist_ok=True)

        for file_path in tqdm(file_paths):
            with h5py.File(file_path, "r") as f:
                image, mask = np.array(f["image"]).transpose(1, 2, 0), np.array(f["labels/segmentation"])

            # There has to be some foreground in the image to be considered for interactive segmentation.
            if len(np.unique(mask)) == 1:
                continue
            image = _pad_image(image)
            mask = _pad_image(mask)
            prefix = os.path.basename(file_path).split(".")[0]
            imageio.imwrite(os.path.join(data_dir, f"{prefix}_image.tif"), image, compression="zlib")
            imageio.imwrite(os.path.join(data_dir, f"{prefix}_mask.tif"), mask, compression="zlib")

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
        batch_size=batch_size,
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
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", default="/home/qinc/Dataset/Microscopy/Histopathology/Lizard/", type=str)
    parser.add_argument("-e", "--experiment_folder", default="./experiments/lizard", type=str)
    parser.add_argument("-m", "--model_type", default="vit_b_histopathology", type=str)
    parser.add_argument("--start_with_box_prompt", action="store_true", help="Whether to start with box prompt")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size for inference")
    args = parser.parse_args()
    main(args)
