import os
from tqdm import tqdm
from glob import glob
from natsort import natsorted

import h5py
import numpy as np
import imageio.v3 as imageio
from skimage.segmentation import relabel_sequential

from torch_em.data.datasets import histopathology as hist

from micro_sam.util import get_sam_model
from micro_sam.evaluation import inference, evaluation
import shutil


def _pad_image(image, target_shape=(512, 512), pad_value=0):
    pad = [
        (max(t - s, 0) // 2, max(t - s, 0) - max(t - s, 0) // 2) for s, t in zip(image.shape[:2], target_shape)
    ]

    if image.ndim == 3:
        pad.append((0, 0))

    return np.pad(image, pad, mode="constant", constant_values=pad_value)


def get_data_paths(input_path, dataset_name):
    # Set specific data folders.
    input_path = os.path.join(input_path, dataset_name)
    if dataset_name == "lizard":

        # Create clone of cropped images in input_path directory.
        # data_dir = os.path.join(input_path, "benchmark_2d")
        data_dir = input_path
        os.makedirs(data_dir, exist_ok=True)

        data_paths = hist.lizard.get_lizard_paths(
            path=input_path, download=True, split="test",
        )
        label_key = 'labels/segmentation'
        image_key = 'image'

        cached_images = os.path.join(input_path, "loaded_images")
        cached_labels = os.path.join(input_path, "loaded_labels")
        os.makedirs(cached_images, exist_ok=True)
        os.makedirs(cached_labels, exist_ok=True)
        for h5_path in data_paths:
            with h5py.File(h5_path, 'r') as file:
                img = file[image_key]
                label = file[label_key]
                img = img[:]
                label = label[:]
                # There has to be some foreground in the image to be considered for interactive segmentation.
                if len(np.unique(label)) == 1:
                    continue
                image = img.transpose(1, 2, 0)
                # breakpoint()
                label = relabel_sequential(label)[0]
                # breakpoint()
                img_path = os.path.join(
                    cached_images, os.path.basename(h5_path).replace(".h5", ".tiff"))
                label_path = os.path.join(
                    cached_labels, os.path.basename(h5_path).replace(".h5", ".tiff"))
                assert image.shape[:2] == label.shape, f"{image.shape}, {label.shape}"

                imageio.imwrite(img_path, image)
                imageio.imwrite(label_path, label)

        image_paths = glob(os.path.join(cached_images, "*.tiff"))
        label_paths = glob(os.path.join(cached_labels, "*.tiff"))

        image_paths, label_paths = natsorted(image_paths), natsorted(label_paths)

        image_outpath = os.path.join(input_path, "loaded_images")
        label_outpath = os.path.join(input_path, "loaded_labels")

        os.makedirs(image_outpath, exist_ok=True)
        os.makedirs(label_outpath, exist_ok=True)

    else:
        raise ValueError

    return image_paths, label_paths


def run_interactive_segmentation(input_path, experiment_folder, model_type, start_with_box_prompt=True):
    # Setup 2: MoNuSeg images (since the images are larger than training patch shape, we crop them to shape (512, 512))
    image_paths, label_paths = get_data_paths(input_path, "lizard")  # NOTE: comment this before running other setups.

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
        start_with_box_prompt=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", default="/home/qinc/Dataset/Microscopy/Histopathology/", type=str)
    parser.add_argument("-e", "--experiment_folder", default="./experiments/lizard",
                        type=str)
    parser.add_argument("-m", "--model_type", default="vit_b_histopathology", type=str)
    args = parser.parse_args()
    main(args)