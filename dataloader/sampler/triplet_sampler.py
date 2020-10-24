import os
import os.path
import torch
from torch.utils.data import Dataset
import random


class TripletSampler(Dataset):
    """Triplet network sampler dataset

    This dataset is a warper class, which input is another dataset class, 
    it will store that class as inner class. This data sampler will call the
    following function of inner calss to get class pair:

    return_dict = inner_class.get_instance(index=..., cls=...)
        return_dict:{
            "img": ...,
            "cls": ...,
            "other": ...
        }

    what's more, the dataset class may also need to inplement following 
    method/property:

    @property
    class_to_idx: return a dict with class comment, like:
        {
            0: "0 th class comment",
            1: "1 th class comment",
            2: "2 th class comment",
            3: "3 th class comment",
            ...
        }

    The default work is discrieb in self.__item__;

    Args:
        dataset: input dataset, which need to include a get_instance method, 
            else will raise error

    Atrribute:
    
    """
    def __init__(self, dataset):
        """
        Init process:
            load dataset, If dataset dose not have get_instance method, raise error.
        """
        self.inner_dataset = dataset

        # check format
        _test = getattr(self.inner_dataset, "class_to_idx", None)
        if _test is None:
            raise AttributeError("dataset must have attribute 'class_to_idx'!")

        _test = getattr(self.inner_dataset, "get_instance", None)
        if _test is None or not callable(_test):
            raise AttributeError("dataset must have attribute 'get_instance'!")

        _test = self.inner_dataset.get_instance()

        assert "img" in _test, "get_instance() should return a dic with {'img': ...}"
        assert "cls" in _test, "get_instance() should return a dic with {'cls': ...}"

    def __getitem__(self, index):
        """
        Get a triplet tuple, refer 'get_triplet_tuple' method for detail

        Args:
            index: dafault is None, specificed by user

        return:
            {
                "anchor_img": Tensor,
                "neg_img": Tensor,
                "pos_img": Tensor,
                "anchor_cls": Tensor,
                "pos_cls": Tensor,
                "neg_cls": Tensor,
                "other": {"some thing else..."}
            }
        """
        return self.get_triplet_tuple(index=index)

    def __len__(self):
        return len(self.inner_dataset)

    def get_triplet_tuple(self, index=None, cls=None):
        """Get a triplet tuple

        According to the args that user input, this method will perform diffrent
        behavior, default behavior is get a random triplet tuple; if 'index' is 
        specific, this method will chose the inner_class's index sample as anchor. 
        if 'cls' is specific, this method will chose the target class's sampel as 
        anchor.

        Args:
            index: dafault is None, specificed by user
            cls: dafault is None, specificed by user

        return:
            {
                "anchor_img": Tensor,
                "neg_img": Tensor,
                "pos_img": Tensor,
                "anchor_cls": Tensor,
                "pos_cls": Tensor,
                "neg_cls": Tensor,
                "other": {"some thing else..."}
            }
        """
        other = {}

        anchor_sample = self.inner_dataset.get_instance(index, cls)
        anchor_cls = anchor_sample["cls"]

        other_cls = list(self.class_to_idx.values())
        other_cls.remove(anchor_cls)

        pos_cls = anchor_cls
        neg_cls = random.choice(other_cls)

        pos_sample = self.inner_dataset.get_instance(cls=pos_cls)
        neg_sample = self.inner_dataset.get_instance(cls=neg_cls)

        return {
            "anchor_img": anchor_sample["img"],
            "neg_img": neg_sample["img"],
            "pos_img": pos_sample["img"],
            "anchor_cls": anchor_cls,
            "pos_cls": pos_cls,
            "neg_cls": neg_cls,
            "other": other,
        }


    @property
    def class_to_idx(self):
        return self.inner_dataset.class_to_idx

if __name__ == "__main__":
    from dataloader.mnist.dataloader_mnist import MNIST
    from torch.utils.data import DataLoader
    from torchvision import transforms

    transformer = transforms.Compose([
                                    transforms.ToTensor(),
                                    ])
    test_dataset = MNIST(transform=transformer)
    test_sample = TripletSampler(test_dataset)

    # warp with dataloader
    dataloader = DataLoader(test_sample, batch_size=4,
                        shuffle=True, num_workers=0)

    # iterate all batch
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched["anchor_img"].size(),
                sample_batched["anchor_cls"].size())