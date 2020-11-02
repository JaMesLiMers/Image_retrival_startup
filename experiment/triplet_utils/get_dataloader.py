from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader.mnist.dataloader_mnist import MNIST
from dataloader.sampler.triplet_sampler import TripletSampler

from utils.log_helper import init_log

logger = init_log("global")

def get_train_dataloader(cfg: dict, use_cuda, pre_process_transform=[]):
    """select the dataset, warp them in triplet dataset.

    select the dataset according to the config's, current support:

    ["MNIST_triplet", "MNIST"]

    If the dataset dont have test version, just return None for test_loader.

    Args:
        cfg: Dict class that must contains required parameter.

    Return:
        A dataloader that defined by following config entry:

            dataset_name  : (str) specific dataset_name
            batch_size    : (int) size of one batch that dataset generate
            image_size   : (int) output image size
            num_workers    : (int) multiprocess load dataset
    """

    # check cfg
    must_include = ["dataset_name", "batch_size", "image_size", "num_workers"]
    for i in must_include:
        assert i in cfg.keys(), "The config file must include {} part for loadding dataloader".format(i)

    # get patten from config file
    dataset_name = cfg["dataset_name"]
    batch_size = cfg["batch_size"]
    num_workers = cfg["num_workers"]
    image_size = cfg["image_size"]


    if dataset_name == "MNIST_triplet":
        transform=transforms.Compose(pre_process_transform + 
        [
                transforms.Grayscale(3),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_dataset = MNIST(transform=transform)
        test_dataset = MNIST(train=False, transform=transform)

        train_triplet_dataset = TripletSampler(train_dataset)
        test_triplet_dataset = TripletSampler(test_dataset)

        if use_cuda:
            train_loader = DataLoader(train_triplet_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
            test_loader = DataLoader(test_triplet_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        else:
            train_loader = DataLoader(train_triplet_dataset, batch_size=batch_size, num_workers=num_workers)
            test_loader = DataLoader(test_triplet_dataset, batch_size=batch_size, num_workers=num_workers)

    elif dataset_name == "MNIST":
        transform=transforms.Compose(pre_process_transform + 
        [
                transforms.Grayscale(3),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_dataset = MNIST(transform=transform)
        test_dataset = MNIST(train=False, transform=transform)

        if use_cuda:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    else:
        raise NotImplementedError("Please specific a valid dataset name")

    logger.info("\nUsing {} dataset.\n".format(dataset_name))

    return train_loader, test_loader


if __name__ == "__main__":
    """
    Test is workable
    """
    pass