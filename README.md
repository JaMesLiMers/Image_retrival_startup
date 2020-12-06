# Image_retrival_startup
This is a basic image retrival machine learning algorithm framework implementation.
This project is build by jameslimer an Aruix

## Environment setup

- Clone the repository 
```
git clone https://github.com/JaMesLiMers/Image_retrival_startup.git
```

- Setup python environment
```
conda create -n torch python=3.8
conda activate torch
pip install -r requirements.txt
```

- Setup your PYTHONPATH first (only in linux like system):
```
chmod +x ./set_path.sh
source ./set_path.sh
```

# Dataset prepare
- To prepare the dataset please follow the instruction in `./dataset` folder.
- current support datasets are:
1. MNIST
2. Fashion_MNIST
3. ... (comming soon!)

# Model training
- how to train your model:
To train your model, just use:
```
python ./experiment/train.py
```
You can also specifiy the training config file in command line
```
python ./experiment/train.py --config_name ...
```

- After training you can test your model:
To test your model, just use:
```
python ./experiment/test.py 
```
OR
```
python ./experiment/test.py --config_name ...
```