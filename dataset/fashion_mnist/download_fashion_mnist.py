import os
import sys
import argparse
from dataset.utils import download_url

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, help="raw data directory", required=True)
args = parser.parse_args()

raw_folder = args.dir

resources = [
        ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
         "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
         "25c81989df183df01b3e8a0aad5dffbe"),
        ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
         "bef4ecab320f06d8554ea6380940ec79"),
        ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
         "bb300cfdad3c16e7a12a480ee83cd310")
    ]

os.makedirs(raw_folder, exist_ok=True)

for url, md5 in resources:  
    filename = url.rpartition('/')[2]
    download_url(url, root=raw_folder, filename=filename, md5=md5)




