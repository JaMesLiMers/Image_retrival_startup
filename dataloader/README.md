# Dataloader of dataset
In this folder, we Implemented the Dataloader of the dataset.

## Currently supported Dataset:
- MNIST

## Folder architecture
The dataloader of each dataset is placed in their folder corresponding by their name.

The wrapper folder holds the different kinds of dataloader to get data, Current have:
- Triplet DataSampler.

## Specification
The dataloader class of each dataset used pytorch's `torchvision.datasets.vision.VisionDataset` as parent class, which provided the default transformer check *(We use torch's default transformer and PIL to load IMG and process them)*. 

But don't worry! The `VisionDataset` works exactly the same as `torch.utils.data.Dataloader`.

## Dataloader protocal:
- Default Dataloader for a dataset:
    ```
    overall return:
        __item__: 
            {
                "img" : Torch tensor,
                "cls" : Scalar,
                "other": {}
            }
    ```
- Triplet sampler dataset:
    ```
    overall return:
        __item__:
            {
                "anchor_img": Tensor,
                "neg_img": Tensor,
                "pos_img": Tensor,
                "anchor_cls": Tensor,
                "pos_cls": Tensor,
                "neg_cls": Tensor,
                "other": {"some thing else..."}
            }
    ```
