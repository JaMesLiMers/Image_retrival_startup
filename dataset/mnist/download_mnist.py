import os
import sys
import argparse
from dataset.utils import download_url

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, help="raw data directory", required=True)
args = parser.parse_args()

raw_folder = args.dir

resources = [
        ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]

os.makedirs(raw_folder, exist_ok=True)

for url, md5 in resources:  
    filename = url.rpartition('/')[2]
    download_url(url, root=raw_folder, filename=filename, md5=md5)






