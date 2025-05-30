import os
import argparse
from tqdm import tqdm
from glob import glob
from natsort import natsorted

import numpy as np
import pandas as pd
import imageio.v3 as imageio
from skimage.measure import label

from tukra.io import read_image
from tukra.inference import segment_using_stardist

from elf.evaluation import mean_segmentation_accuracy


def _run_evaluation(gt_paths, prediction_paths, verbose=True):
    assert len(gt_paths) == len(prediction_paths), \
        f"label / prediction mismatch: {len(gt_paths)} / {len(prediction_paths)}"

    msas, sa50s, sa75s = [], [], []
    for gt_path, pred_path in tqdm(
        zip(gt_paths, prediction_paths, strict=False),
        desc="Evaluate predictions",
        total=len(gt_paths),
        disable=not verbose,
    ):
        assert os.path.exists(gt_path), gt_path
        assert os.path.exists(pred_path), pred_path

        gt = imageio.imread(gt_path)
        gt = label(gt)
        pred = imageio.imread(pred_path)

        msa, scores = mean_segmentation_accuracy(pred, gt, return_accuracies=True)
        sa50, sa75 = scores[0], scores[5]
        msas.append(msa), sa50s.append(sa50), sa75s.append(sa75)

    return msas, sa50s, sa75s


def evaluate_stardist(prediction_dir, label_dir, result_dir, dataset):
    gt_paths = natsorted(glob(os.path.join(label_dir, "eval_split", "test_labels", "*")))
    for checkpoint in ["stardist"]:
        save_path = os.path.join(result_dir, dataset, checkpoint, f'{dataset}_stardist_stardist_ais_result.csv')
        if os.path.exists(save_path):
            continue

        prediction_paths = natsorted(glob(os.path.join(prediction_dir, "*")))
        if len(prediction_paths) == 0:
            print(f"No predictions for {dataset} dataset on {checkpoint} checkpoint found")
            continue

        os.makedirs(os.path.join(result_dir, dataset, checkpoint), exist_ok=True)
        print(f"Evaluating {dataset} dataset on Stardist ...")
        msas, sa50s, sa75s = _run_evaluation(gt_paths=gt_paths, prediction_paths=prediction_paths)
        results = pd.DataFrame.from_dict(
            {"mSA": [np.mean(msas)], "SA50": [np.mean(sa50s)], "SA75": [np.mean(sa75s)]}
        )

        results.to_csv(save_path, index=False)
        print(results.head(2))


def infer_stardist(data_dir, output_path):
    image_paths = natsorted(glob(os.path.join(data_dir, "eval_split", "test_images", "*")))
    os.makedirs(output_path, exist_ok=True)
    for image_path in image_paths:
        image = read_image(image_path)
        segmentation = segment_using_stardist(image=image, model_name="2D_versatile_he")
        imageio.imwrite(os.path.join(output_path, os.path.basename(image_path)), segmentation)


def run_inference(input_dir, output_dir, dataset):
    output_path = os.path.join(output_dir, 'inference', dataset, "stardist")
    input_path = os.path.join(input_dir, dataset)
    if os.path.exists(
        os.path.join(output_dir, "results", dataset, "stardist", f'{dataset}_stardist_stardist_ais_result.csv')
    ):
        return

    os.makedirs(output_path, exist_ok=True)
    print(f"Running inference with StarDist model on {dataset} dataset... \n")
    infer_stardist(input_path, output_path)
    print(f"Inference on {dataset} dataset with the StarDist model successfully completed \n")

    evaluate_stardist(
        prediction_dir=output_path,
        label_dir=input_path,
        result_dir=os.path.join(output_dir, 'results'),
        dataset=dataset
        )


def get_stardist_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=None, help="The path to the input data")
    parser.add_argument("-d", "--dataset", type=str, default=None, help="The datasets to infer on")
    parser.add_argument("-o", "--output_dir", type=str, default=None, help="The path where the results are saved")
    args = parser.parse_args()
    return args


def main():
    args = get_stardist_args()
    run_inference(input_dir=args.input, output_dir=args.output_dir, dataset=args.dataset)


if __name__ == "__main__":
    main()
