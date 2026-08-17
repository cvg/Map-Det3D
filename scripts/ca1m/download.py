"""Download and extract tar files from URLs listed in a text file."""

import argparse
import os
import tarfile
from multiprocessing import Pool, cpu_count

import requests
from tqdm import tqdm


def download_and_extract(args):
    """Download the data and extract it."""
    url, output_dir = args

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, os.path.basename(url))

    dir_name = os.path.splitext(os.path.basename(filename))[0].split("-")[-1]

    if os.path.exists(os.path.join(output_dir, dir_name)):
        print(f"Directory {dir_name} already exists. Skipping download.")
        return

    # Download file
    response = requests.get(url)
    response.raise_for_status()
    with open(filename, "wb") as f:
        f.write(response.content)

    # Extract tar file
    with tarfile.open(filename, "r") as tar:
        tar.extractall(path=output_dir)

    # Delete tar file
    print(f"Deleting {filename}...")
    os.remove(filename)


def process_txt_file(txt_file: str, output_dir: str, workers: int = 1):
    """Read URLs from a text file and download & extract each."""
    with open(txt_file, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    os.makedirs(output_dir, exist_ok=True)

    if workers > 1:
        with Pool(processes=workers) as pool:
            for _ in tqdm(
                pool.imap_unordered(
                    download_and_extract, [(url, output_dir) for url in urls]
                ),
                total=len(urls),
            ):
                pass
    else:
        for url in urls:
            try:
                download_and_extract((url, output_dir))
            except Exception as e:
                print(f"Failed to process {url}: {e}")


if __name__ == "__main__":
    """Main function to process the text file."""
    parser = argparse.ArgumentParser(
        description="Download CA1M dataset and extract tar files."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        help="The path to save the dataset.",
        default="./data/CA1M",
    )
    parser.add_argument(
        "--split",
        type=str,
        help="The dataset split to download (e.g., train, val).",
        default="train",
        choices=["train", "val"],
    )
    args = parser.parse_args()

    process_txt_file(
        os.path.join(args.data_root, f"{args.split}.txt"),
        output_dir=os.path.join(args.data_root, args.split),
        workers=cpu_count(),
    )
