import logging
from torch._C import Size
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataloader.mnist.dataloader_mnist import MNIST
from dataloader.sampler.triplet_sampler import TripletSampler

from utils.log_helper import init_log, add_file_handler

logger = init_log("global")

def get_train_dataloader(dataset, batch_size, output_size, num_worker, use_cuda, pre_process_transform=[]):
    """select the dataset, warp them in triplet dataset.

    select the dataset according to the args, current support:

    ["MNIST_triplet",]

    If the dataset dont have test version, just return None for test_loader.

    Args:
        dataset: specific dataset
        batch_size: size of one batch that dataset generate
        output_size: output image size
        num_worker: multiprocess load dataset
    """

    if dataset == "MNIST_triplet":
        transform=transforms.Compose(pre_process_transform + 
        [
                transforms.Grayscale(3),
                transforms.Resize(output_size),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_dataset = MNIST(transform=transform)
        test_dataset = MNIST(train=False, transform=transform)

        train_triplet_dataset = TripletSampler(train_dataset)
        test_triplet_dataset = TripletSampler(test_dataset)

        if use_cuda:
            train_loader = DataLoader(train_triplet_dataset, batch_size=batch_size, num_workers=num_worker, pin_memory=True)
            test_loader = DataLoader(test_triplet_dataset, batch_size=batch_size, num_workers=num_worker, pin_memory=True)
        else:
            train_loader = DataLoader(train_triplet_dataset, batch_size=batch_size, num_workers=num_worker)
            test_loader = DataLoader(test_triplet_dataset, batch_size=batch_size, num_workers=num_worker)

    elif dataset == "MNIST":
        transform=transforms.Compose(pre_process_transform + 
        [
                transforms.Grayscale(3),
                transforms.Resize(output_size),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_dataset = MNIST(transform=transform)
        test_dataset = MNIST(train=False, transform=transform)

        if use_cuda:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_worker, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_worker, pin_memory=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_worker)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_worker)

    else:
        raise NotImplementedError("Please specific a valid dataset name")

    logger.info("\nUsing {} dataset.\n".format(dataset))

    return train_loader, test_loader


if __name__ == "__main__":
    """
    Test is workable
    """
    train_loader, test_loader = get_train_dataloader("MNIST", 64, 224, 0, True, [transforms.RandomHorizontalFlip()])

    for index, batch in enumerate(train_loader):
        anc_img = batch["anchor_img"]
        pos_img = batch["pos_img"]
        neg_img = batch["neg_img"]

        print(anc_img.size())