import os
import sys
import argparse
import torch
from dataset.mnist.utils_mnist import read_image_file, read_label_file

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", type=str, help="Directory that original data in", required=True)
parser.add_argument("-o", "--output_dir", type=str, help="Directory that target data to put", required=True)
args = parser.parse_args()

raw_folder = args.input_dir
processed_folder = args.output_dir

os.makedirs(processed_folder, exist_ok=True)

training_file = "mnist_train.pt"
test_file = "mnist_test.pt"

print('Processing...')

training_set = (
    read_image_file(os.path.join(raw_folder, 'train-images-idx3-ubyte')),
    read_label_file(os.path.join(raw_folder, 'train-labels-idx1-ubyte'))
)
test_set = (
    read_image_file(os.path.join(raw_folder, 't10k-images-idx3-ubyte')),
    read_label_file(os.path.join(raw_folder, 't10k-labels-idx1-ubyte'))
)

with open(os.path.join(processed_folder, training_file), 'wb') as f:
    torch.save(training_set, f)
with open(os.path.join(processed_folder, test_file), 'wb') as f:
    torch.save(test_set, f)

