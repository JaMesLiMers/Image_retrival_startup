# MNIST dataset prepare

- Setup your PYTHONPATH first
- Download MNIST dataset(run in root dir):
```
python ./dataset/fashion_mnist/download_fashion_mnist.py -d ./dataset/fashion_mnist/raw_data/
```
- Extract dataset file (delete the original '.gz' file):
```
python ./dataset/fashion_mnist/extract_fashion_mnist.py -d ./dataset/fashion_mnist/raw_data/ --remove True
```
- Process the dataset file:
```
python ./dataset/fashion_mnist/process_fashion_mnist.py -i ./dataset/fashion_mnist/raw_data/ -o ./dataset/fashion_mnist/processed_data/
```

- After run above code, the fashion_mnist processed_data folder will contail following file:
```
.
├─fashion_mnist
└─ processed code
    ├── mnist_test.pt
    └── mnist_train.pt
```