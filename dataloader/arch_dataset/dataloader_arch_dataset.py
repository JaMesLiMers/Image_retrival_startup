import os
import os.path
import torch
from torchvision import transforms
import random
import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader
import re

class UnNormalize(object):
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensor = torch.clone(tensor)
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

"""
The ArchDatset is transformed from torchvition's MNIST dataset calss
but modified to suit our requirement.
"""
class ArchDatset(VisionDataset):

    processed_folder = './dataset/arch_dataset/processed_data/'
    training_file = 'train.pt'  
    test_file = 'test.pt'
    classes = []
    unnorm = UnNormalize()

    def __init__(self, root=None, train=True, transform=None, target_transform=None):
        """
        Init process:
            load data from file, the entire dataset are load into ram.
        """
        if root is None:
            root = self.processed_folder
            
        super(ArchDatset, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        
        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' Please prepare data first. (dataset should be placed in {})'.format(self.processed_folder))

        self.data, self.targets, self.classes = torch.load(os.path.join(self.processed_folder, data_file))

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
                        {
                            "index" : index,
                        }
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
                        {
                            "index" : index,
                        }
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
        img = self.unnorm(img)
        img = transforms.ToPILImage()(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        other["index"] = index

        rdic["img"] = img
        rdic["cls"] = target
        rdic["other"] = other

        return rdic

    def get_raw_image(self, index=None):
        """
            Get one instance from dataset, acccording to the specification index

            The default performance of function is return a random PIL.Image, 
            If the index is specified, then return index Image

            Args:
                index (int): Index num

            Returns:
                A PIL.Image.
        """
        if index is not None:
            index = index
        else:
            index = random.randint(0, len(self))

        img = self.data[index]

        # to return a PIL Image
        img = self.unnorm(img)
        img = transforms.ToPILImage()(img)
        return img


    def __len__(self):
        return len(self.data)

    @property
    def class_to_idx(self):
        
        result = {}
        for i, _class in enumerate(self.classes):
            index = int(re.findall(r"\d+",_class)[0])
            result[_class] = index

        return result




if __name__ == "__main__":
    test_dataset = ArchDatset()

    for i in range(10):
        a = test_dataset.get_instance(cls=12)
        b = a["img"]
        b.save("./test.png")


    # # get index 1
    # print("get index 1 for 2 times:\n")
    # for i in range(2):
    #     print(test_dataset.get_instance(index=10))

    

    # # random get 10 
    # print("random get 10 :\n")
    # for i in range(10):
    #     print(test_dataset.get_instance())
    
    # # get 10 class 0
    # print("get 10 class 0:\n")
    # for i in range(10):
    #     print(test_dataset.get_instance(cls=0))


    # transformer = transforms.Compose([
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                                 ])
    # transformed_dataset = ArchDatset(transform=transformer)

    # # random get 10 on transformer
    # print("random get 10 and transform:\n")
    # for i in range(10):
    #     print(type(transformed_dataset.get_instance()["img"]))

    # # warp with dataloader
    # dataloader = DataLoader(transformed_dataset, batch_size=4, num_workers=0)

    # # iterate all batch
    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(i_batch, sample_batched['img'].size(),
    #             sample_batched['cls'].size())




