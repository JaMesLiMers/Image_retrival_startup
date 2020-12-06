import os.path
from torchvision import transforms
from torch.utils.data import DataLoader

from dataloader.mnist.dataloader_mnist import MNIST

"""
The Fashion MNIST is build from torchvition's MNIST dataset calss
but modified to suit our requirement.
"""
class Fashion_MNIST(MNIST):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``FashionMNIST/processed/training.pt``
            and  ``FashionMNIST/processed/test.pt`` exist.
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
        get_raw_image: 根据指定的index, 获取一个未处理的图, 在test中会被用到.

    """
    processed_folder = './dataset/fashion_mnist/processed_data/'
    training_file = 'mnist_train.pt'  
    test_file = 'mnist_test.pt'
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    


if __name__ == "__main__":
    test_dataset = Fashion_MNIST()

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
    dataloader = DataLoader(transformed_dataset, batch_size=4, num_workers=0)

    # iterate all batch
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['img'].size(),
                sample_batched['cls'].size())