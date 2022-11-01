import os
import pandas as pd
import shutil
from tqdm import tqdm

from speechsep.dataset import get_storage_dir

if __name__ == "__main__":
    dataset_mini_size = 100
    dataset_mini_path = "data/datasets/mini"

    dataset_full = "dev"
    dataset_full_path = f"{get_storage_dir()}/Libri2Mix/wav8k/min/"

    # Load metadata for full dataset and sample examples for mini dataset
    metadata = pd.read_csv(dataset_full_path + f"metadata/mixture_{dataset_full}_mix_both.csv")
    metadata_mini = (
        metadata.sample(dataset_mini_size, random_state=42)
        .reset_index(drop=True)
        .rename(
            columns={
                "mixture_path": "mixture_path_src",
                "source_1_path": "source_1_path_src",
                "source_2_path": "source_2_path_src",
                "noise_path": "noise_path_src",
            }
        )
    )

    # Generate paths for mini dataset
    def _generate_paths(example):
        id = example.mixture_ID
        example["mixture_path"] = f"{dataset_mini_path}/mix_both/{id}.wav"
        example["source_1_path"] = f"{dataset_mini_path}/s1/{id}.wav"
        example["source_2_path"] = f"{dataset_mini_path}/s2/{id}.wav"
        example["noise_path"] = f"{dataset_mini_path}/noise/{id}.wav"
        return example

    metadata_mini = metadata_mini.apply(_generate_paths, axis=1)

    # Create dataset folder from scratch
    shutil.rmtree(dataset_mini_path)
    os.makedirs(f"{dataset_mini_path}/mix_both", exist_ok=True)
    os.makedirs(f"{dataset_mini_path}/s1", exist_ok=True)
    os.makedirs(f"{dataset_mini_path}/s2", exist_ok=True)
    os.makedirs(f"{dataset_mini_path}/noise", exist_ok=True)

    # Copy examples
    for example in tqdm(metadata_mini.itertuples(), total=dataset_mini_size):
        shutil.copy2(example.mixture_path_src, example.mixture_path)
        shutil.copy2(example.source_1_path_src, example.source_1_path)
        shutil.copy2(example.source_2_path_src, example.source_2_path)
        shutil.copy2(example.noise_path_src, example.noise_path)

    # Write metadata file
    metadata_mini.drop(
        columns=["mixture_path_src", "source_1_path_src", "source_2_path_src", "noise_path_src"]
    ).to_csv(f"{dataset_mini_path}/mixture_mini_mix_both.csv", index=False)
