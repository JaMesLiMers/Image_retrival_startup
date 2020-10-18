# MNIST dataset prepare

- Setup your PYTHONPATH first
- Download MNIST dataset(run in root dir):
```
python ./dataset/mnist/download_mnist.py -d ./dataset/mnist/raw_data/
```
- Extract dataset file (delete the original '.gz' file):
```
python ./dataset/mnist/extract_mnist.py -d ./dataset/mnist/raw_data/ -r True
```
