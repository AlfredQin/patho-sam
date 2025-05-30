import argparse

from torch_em.data.datasets.histopathology.lizard import get_lizard_data

def download_lizard(path):
    get_lizard_data(path=path, download=True, split="train")


download_methods = {
    "lizard": download_lizard,
}



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download histopathology dataset")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--path", type=str, help="path to save dataset")
    args = parser.parse_args()
    dataset = args.dataset
    path = args.path
    if dataset not in download_methods:
        raise ValueError(f"Dataset {dataset} is not supported.")
    download_method = download_methods[dataset]
    download_method(path)