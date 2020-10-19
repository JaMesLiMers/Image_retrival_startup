# MNIST dataset prepare

- Setup your PYTHONPATH first
- Download MNIST dataset(run in root dir):
```
python ./dataset/mnist/download_mnist.py -d ./dataset/mnist/raw_data/
```
- Extract dataset file (delete the original '.gz' file):
```
python ./dataset/mnist/extract_mnist.py -d ./dataset/mnist/raw_data/ --remove True
```
- Process the dataset file:
```
python ./dataset/mnist/process_mnist.py -i ./dataset/mnist/raw_data/ -o ./dataset/mnist/processed_data/
```

- After run above code, the mnist processed_data folder will contail following file:
```
.
├─mnist
└─ processed code
    ├── mnist_test.pt
    └── mnist_train.pt
```