import os
import sys
import argparse
from dataset.utils import extract_archive

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, help="Directory that original data in", required=True)
parser.add_argument("-r", "--remove", default=False, type=bool, help="remove finished")
args = parser.parse_args()

file_name = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",]

extract_root = args.dir
remove_finished = args.remove

os.makedirs(extract_root, exist_ok=True)

for f in file_name: 
    file_path = os.path.join(extract_root, f)
    extract_archive(file_path, extract_root, remove_finished)
