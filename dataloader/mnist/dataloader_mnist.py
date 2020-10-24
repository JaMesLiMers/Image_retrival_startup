import os
import os.path
import torch
from torchvision import transforms
import random
import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader


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


    Atrribute:
        processed_folder: 处理好的数据文件夹
        training_file: 训练集的文件名
        test_file: 测试集的文件名
        classes: 保存按照index编码的cls信息注释
        train: 根据true和false进行不同的数据集加载
        class_index: dict, 存的是根据cls来分的index列表

        property:
            class_to_idx: 返回一个对应class的注释dict

        __init__: 将整个数据集load进内存
        __getitem__: 根据index获取数据样本
        __len__: 获取整个数据集的长度

        get_instance: 获取一个数据样本的基本方法, 分别包括随机, 指定index, 指定cls三种

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
                               ' Please prepare data first. (dataset should be placed in {})'.format(self.processed_folder))

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

        self.class_index = self._filt_class_idx()


    def _filt_class_idx(self):
        """
        filter class, seperate them into idx

        Return:
            class_dict: 
                {
                    "class_1": [index_1, ....],
                    "class_2": [index_1, ....],
                    "class_3": [index_1, ....],
                    "class_4": [index_1, ....]
                }
        """
        cls_dic = {}

        total_idx = list(range(len(self.targets)))

        for i in self.class_to_idx.values():
            cls_dic[i] = [x for x in total_idx if self.targets[x] == i]
        
        return cls_dic


    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

    def __getitem__(self, index):
        """
        Defaulet use as torch dataset

        Return according to index.

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
        return self.get_instance(index=index)
        

    def get_instance(self, index=None, cls=None):
        """
        Get one instance from dataset, acccording to the specification

        The default performance of function is return a random sample, 
        If the index is specified, then return index sample, else if cls
        is specified, then get target class sample.

        Args:
            index (int): Index num
            cls (int):   Class num

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

        if index is not None:
            index = index
        elif cls is not None:
            index = random.choice(self.class_index[cls])
        else:
            index = random.randint(0, len(self))
            
        img, target = self.data[index], int(self.targets[index])

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


    def __len__(self):
        return len(self.data)

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}


if __name__ == "__main__":
    test_dataset = MNIST()

    # get index 1
    print("get index 1 for 2 times:\n")
    for i in range(2):
        print(test_dataset.get_instance(index=1))

    # random get 10 
    print("random get 10 :\n")
    for i in range(10):
        print(test_dataset.get_instance())
    
    # get 10 class 0
    print("get 10 class 0:\n")
    for i in range(10):
        print(test_dataset.get_instance(cls=0))


    transformer = transforms.Compose([
                                    transforms.ToTensor(),
                                    ])
    transformed_dataset = MNIST(transform=transformer)

    # random get 10 on transformer
    print("random get 10 and transform:\n")
    for i in range(10):
        print(type(transformed_dataset.get_instance()["img"]))

    # warp with dataloader
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=0)

    # iterate all batch
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['img'].size(),
                sample_batched['cls'].size())


