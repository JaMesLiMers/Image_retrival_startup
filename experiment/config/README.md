# YML文件
- 训练&验证文件格式:
```
# ------------------------ General Train Setting -------------------------
# 这里包括了general的设置

# 实验名称, 示例里是命名格式
experiment_name: "dataset_backbone_loss_train"

# 训练的epoch数
train_epochs: 10

# 是否resume, resume的.pt文件名称 (名字的前缀和实验名相同)
# 若留空, 则测试所有的模型
resume_name: "dataset_backbone_loss_train_epoch_n.pt"

# 训练的种子
experiment_seed: 1

# 是否不需要使用cuda
dont_use_cuda: False

# ------------------------ validation setting ------------------------
# 这里包括了关于validation的设置

# 是否进行validation
validate_model: True

# 每隔多少epoch进行一次validation
val_interval: 1

# ------------------------ log Setting ------------------------
# 关于log的设置

# 每隔多少batch显示一次进度.
log_interval: 100

# ------------------------ Dataloader Setting ------------------------
# 这里包括了Dataloader的设置, 使用哪一个数据集.

# 使用哪一个数据集, 现在支持的有: ["MNIST_triplet", "MNIST", "Fashion_MNIST_triplet", "Fashion_MNIST", "Arch_Dataset_triplet", "Arch_Dataset"]

dataset_name: "MNIST"

# 设置数据集的batch_size
batch_size: 64

# 设置数据集的平行读取
num_workers: 1

# 设置数据集产出的图像的大小
image_size: 224

# ------------------------ Backbone Setting ------------------------
# 这里包括了backbone网络的选取设置.

# 使用哪一个backbone, 现在支持的有: ["Alexnet", 
#        "VGG11", "VGG13", "VGG16", "VGG19",
#        "Resnet18", "Resnet34", "Resnet50", "Resnet101", "Resnet152", ]
backbone_name: "Alexnet"

# backbone最后输出的特征长度.
embedding_dim: 256

# 是否使用pretrain的模型, torch在ImageNet上的pretrain.
pretrained: False

# ------------------------ Optimizer Setting ------------------------
# 这里包括了optimizer的选取和超参数设置

# 使用哪一个optimizer, 现在支持的有: ["sgd", "adagrad", "rmsprop", "adam",]
optimizer_name: "adam"

# 对sgd的设置
sgd:
    lr: 1e-3
    
    momentum: 0.9
    
    dampening: 0
    
    weight_decay: 2e-4
    
    nesterov: False

# 对adagrad的设置
adagrad:
    lr: 1e-3

    lr_decay: 0

    weight_decay: 2e-4

    initial_accumulator_value: 0

    eps: 1e-10

# 对rmsprop的设置
rmsprop:
    lr: 1e-3

    alpha: 0.99

    eps: 1e-08

    weight_decay: 2e-4

    momentum: 0

    centered: False

# 对adam的设置
adam: 
    lr: 1e-3

    betas: [0.9, 0.999]

    eps: 1e-08

    weight_decay: 2e-4

    amsgrad: False


# ------------------------ Loss Setting ------------------------
# 这里包括了loss的选取和设置

# 使用哪一个loss, 现在支持的有: ["triplet"]
loss_name: "triplet"

# 对triplet loss的设置:
triplet:
   margin: 1.0

   norm_digree: 2

   reduction: "mean"


# ------------------------ End Setting ------------------------
```

- 测试文件格式:
```
# ------------------------ General Test Setting -------------------------
# 这里包括了general的设置

# 实验名称, 示例里是命名格式
experiment_name: "dataset_backbone_loss_train"

# 是否resume, resume的.pt文件名称 (名字的前缀和实验名相同)
resume_name: "dataset_backbone_loss_train_epoch_n.pt"

# 训练的种子
experiment_seed: 1

# 是否不需要使用cuda
dont_use_cuda: False

# ------------------------ log Setting ------------------------
# 关于log的设置

# 每隔多少batch显示一次进度.
log_interval: 100

# ------------------------ Dataloader Setting ------------------------
# 这里包括了Dataloader的设置, 使用哪一个数据集.

# 使用哪一个数据集, 现在支持的有:     ["MNIST_triplet", "MNIST", "Fashion_MNIST_triplet", "Fashion_MNIST", "Arch_Dataset_triplet", "Arch_Dataset"]

dataset_name: "MNIST"

# 设置数据集的batch_size
batch_size: 64

# 设置数据集的平行读取
num_workers: 1

# 设置数据集产出的图像的大小
image_size: 224

# ------------------------ Backbone Setting ------------------------
# 这里包括了backbone网络的选取设置.

# 使用哪一个backbone, 现在支持的有: ["Alexnet", 
#        "VGG11", "VGG13", "VGG16", "VGG19",
#        "Resnet18", "Resnet34", "Resnet50", "Resnet101", "Resnet152", ]
backbone_name: "Alexnet"

# backbone最后输出的特征长度.
embedding_dim: 256

# 是否使用pretrain的模型, torch在ImageNet上的pretrain.
pretrained: False

# ------------------------ End Setting ------------------------

```