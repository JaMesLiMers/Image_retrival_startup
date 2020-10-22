import os
import os.path
import torch
import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset

"""
The MNIST is copied from torchvition's MNIST dataset calss
but modified to suit our requirement.
"""
class MNIST(VisionDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    processed_folder = './dataset/mnist/processed_data/'
    training_file = 'mnist_train.pt'  
    test_file = 'mnist_test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, root=None, train=True, transform=None, target_transform=None):
        """
        Init process:
            load data from file, since MNIST is a small dataset, the entire dataset are load 
            into ram.
        """
        if root is None:
            root = self.processed_folder
            
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        
        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' Please prepare data first. (dataset should be placed in {})'.format(processed_folder))

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))


    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

    def __getitem__(self, index):
        """
        Defaulet use as torch dataset

        Args:
            index (int): Index

        Returns:
            dictionary:
                {
                    "img": target image,
                    "cls": target class, 
                    "other": other information,
                }
        """
        rdic = {}
        other = {}

        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        rdic["img"] = img
        rdic["cls"] = target
        rdic["other"] = other

        return rdic

    def get_random_instance(self):
        """
        docstring
        """
        pass

    def get_index_instance(self, index):
        """
        docstring
        """
        pass

    def get_specific_instance(self, cls=None):
        """
        docstring
        """
        pass

    def __len__(self):
        return len(self.data)

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}


if __name__ == "__main__":
    test_dataset = MNIST()

    print(test_dataset)