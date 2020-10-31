import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataloader.sampler.triplet_sampler import TripletSampler

import time
import logging
from utils.average_meter_helper import AverageMeter
from utils.log_helper import init_log, add_file_handler, print_speed

from model.loss.triplet_loss import TripletLoss
from model.model.triplet_model import TripletNetModel
from experiment.triplet_utils.get_backbone import get_backbone
from experiment.triplet_utils.get_optimizer import get_optimizer
from experiment.triplet_utils.get_dataloader import get_train_dataloader


parser = argparse.ArgumentParser(description='Train triplet network')

# Dataset 
# TODO: Add more support
parser.add_argument('--dataset', type=str, default="MNIST",
                    choices=["MNIST"],
                    help='Which dataset to use for training.')

# Batch size
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training (default: 64)')

# num_workers
parser.add_argument('--num_workers', default=1, type=int,
                    help="Number of workers for data loaders (default: 1)"
                    )

# Model Backbone
# TODO: Add more support
parser.add_argument('--backbone', type=str, default="Alexnet",
                    choices=["Alexnet", 
                             "VGG11", "VGG13", "VGG16", "VGG19",
                             "Resnet18", "Resnet34", "Resnet50", "Resnet101", "Resnet152",],
                    help='Which backbone to use for training.')

# embedding dimention
parser.add_argument('--embedding_dim', default=256, type=int,
                    help="Dimension of the embedding vector (default: 256)"
                    )

# Image size
parser.add_argument('--image_size', default=224, type=int,
                    help='Input image size (default: 224 (224x224), depend on the backbone.)'
                    )

# Loss setting
# margin of triplet loss
parser.add_argument('--margin', type=float, default=0.2, 
                    help='margin for triplet loss (default: 0.2)')

# Training argument

# Epoch 
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train (default: 10)')

# use pretrained model
parser.add_argument('--pretrained', default=False, type=bool,
                    help="Download a model pretrained on the ImageNet dataset (Default: False)")

# Optimizer

# optimizer arch
parser.add_argument('--optimizer', type=str, default="adam", choices=["sgd", "adagrad", "rmsprop", "adam"],
                    help="Required optimizer for training the model: ('sgd','adagrad','rmsprop','adam'), (default: 'adam')")
                    
# learning rate
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')

# momentem 
parser.add_argument('--momentum', type=float, default=0.5,
                    help='SGD momentum (default: 0.5)')

# Other

# Resume pretrain or not
parser.add_argument('--resume_name', default='', type=str,
                    help='file name of latest checkpoint, placed in experiment folder (default: none)')


# Global setting

# random seed
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')

# Do not use cuda? if no cuda on pc, then don't use
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')

# log interval
parser.add_argument('--log_interval', type=int, default=100, 
                    help='how many samples batch to wait before logging training status')

# expriment folder name
parser.add_argument('--name', default='TripletNet', type=str,
                    help='name of experiment')

args = parser.parse_args()

dataset = args.dataset
batch_size = args.batch_size
num_workers = args.num_workers
backbone = args.backbone
embedding_dim = args.embedding_dim
image_size = args.image_size
margin = args.margin
epochs = args.epochs
pretrained = args.pretrained
optimizer = args.optimizer
lr = args.lr
momentum = args.momentum
resume_name = args.resume_name
seed = args.seed
no_cuda = args.no_cuda
log_interval = args.log_interval
name = args.name


def set_model_gpu_mode(model, cuda):
    """decide wether train on multi GPU

    If train on multi GPU, the model need to be warrped by dataparallel.
    
    Args:
        model: input model, is a nn.Moudle
        cuda: bool, wether use cuda.
    """
    flag_train_gpu = torch.cuda.is_available() & cuda
    flag_train_multi_gpu = False

    if flag_train_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.cuda()
        flag_train_multi_gpu = True
        logger.info('\nUsing multi-gpu training.\n')

    elif flag_train_gpu and torch.cuda.device_count() == 1:
        model.cuda()
        logger.info('\nUsing single-gpu training.\n')

    else:
        logger.info('\nUse cpu for training.\n')

    return model, flag_train_multi_gpu



"""
Decide wether to use cuda, set random seed and init the logger & average meter,
init experiment folder.
"""

# set cuda
cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# set seed 
torch.manual_seed(args.seed)
if cuda:
    torch.cuda.manual_seed(args.seed)

# Create experiment folder structure
curernt_file_path = os.path.dirname(os.path.abspath(__file__))
experiment_folder = os.path.join(curernt_file_path, name)
experiment_snap_folder = os.path.join(experiment_folder, "snap")
experiment_board_folder = os.path.join(experiment_folder, "board")
os.makedirs(experiment_folder, exist_ok=True)
os.makedirs(experiment_snap_folder, exist_ok=True)
os.makedirs(experiment_board_folder, exist_ok=True)

# init board writer
writer = SummaryWriter(experiment_board_folder)

# get log
logger = init_log("global")
add_file_handler("global", os.path.join(experiment_folder, 'test.log'), level=logging.INFO)

# init avg meter
# avg.update(time=1.1, accuracy=.99)
avg = AverageMeter()

# get dataset 
dataset_pre_processing = []
train_dataloader, test_dataloader = get_train_dataloader(dataset=dataset, 
                                                         batch_size=batch_size, 
                                                         output_size=image_size, 
                                                         num_worker=num_workers, 
                                                         use_cuda=cuda, 
                                                         pre_process_transform=dataset_pre_processing)

# Instantiate model
model = get_backbone(model_architecture=backbone, 
                     pretrained=pretrained, 
                     embedding_dimension=embedding_dim)

model = TripletNetModel(model)

# Load model to GPU or multiple GPUs if available
model, flag_train_multi_gpu = set_model_gpu_mode(model, cuda)

# Set optimizer
optimizer_model = get_optimizer(optimizer=optimizer,
                                model=model,
                                learning_rate=lr,
                                momentum=momentum)

# Set loss function
loss = TripletLoss()

# Resume from a model checkpoint
start_epoch = 0
resume_path = os.path.join(experiment_snap_folder, resume_name)
if resume_name:
    if os.path.isfile(resume_path):
        logger.info("\nLoading checkpoint {} in {} ...\n".format(resume_name, experiment_folder))

    checkpoint = torch.load(resume_path)
    start_epoch = checkpoint['epoch']

    optimizer_model.load_state_dict(checkpoint['optimizer_model_state_dict'])

    # In order to load state dict for optimizers correctly, model has to be loaded to gpu first
    if flag_train_multi_gpu:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    logger.info("\nCheckpoint loaded: start epoch from checkpoint epoch = {}\n".format(start_epoch))
else:
    logger.warning("\nWARNING: No checkpoint found at {}!\nTraining from scratch.\n".format(resume_path))

# Start Training loop
end_epoch = start_epoch + epochs
logger.info("\nTraining using triplet loss starting for {} epochs, current epoch: {}, target epoch: {};\n".format(epochs, start_epoch, end_epoch))

# for progress printing
total_batch = epochs * len(train_dataloader)
current_batch = 0
batch_time = 0

for epoch in range(start_epoch, end_epoch):
    for batch_idx, batch_sample in enumerate(train_dataloader):
        # Skip last iteration to avoid the problem of having different number of tensors while calculating
        # averages (sizes of tensors must be the same for pairwise distance calculation)
        if batch_idx + 1 == len(train_dataloader):
            continue

        batch_start_time = time.time()

        # Forward pass - compute embeddings
        anc_imgs = batch_sample['anchor_img']
        pos_imgs = batch_sample['pos_img']
        neg_imgs = batch_sample['neg_img']

        pos_cls = batch_sample['pos_cls']
        neg_cls = batch_sample['neg_cls']

        # move to gpu if use cuda
        if cuda:
            anc_imgs = anc_imgs.to(device)
            pos_imgs = pos_imgs.to(device)
            neg_imgs = neg_imgs.to(device)
            pos_cls = pos_cls.to(device)
            neg_cls = neg_cls.to(device)

        # forward
        output = model.forward_triplet(anc_imgs, pos_imgs, neg_imgs)
        
        # get output 
        anc_emb = output['anchor_map']
        pos_emb = output['pos_map']
        neg_emb = output['neg_map']

        pos_dists = torch.mean(output['dist_pos'])
        neg_dists = torch.mean(output['dist_neg'])

        # loss compute
        loss_value = loss(anc_emb, pos_emb, neg_emb)

        # Backward pass
        optimizer_model.zero_grad()
        loss_value.backward()
        optimizer_model.step()

        current_batch +=1
        batch_time = time.time() - batch_start_time

        avg.update(time=batch_time, triplet_loss=loss_value, pos_dists=pos_dists, neg_dists=neg_dists)
        writer.add_scalar("Loss/train", loss_value, global_step=current_batch)
        writer.add_scalar("Other/pos_dists", pos_dists, global_step=current_batch)
        writer.add_scalar("Other/neg_dists", neg_dists, global_step=current_batch)
        writer.add_scalar("Epoch/train_loss_epoch_{}".format(epoch), loss_value, global_step=current_batch)
        writer.add_scalar("Epoch/pos_dists_epoch_{}".format(epoch), pos_dists, global_step=current_batch)
        writer.add_scalar("Epoch/neg_dists_epoch_{}".format(epoch), neg_dists, global_step=current_batch)
        writer.add_scalar("Global_AVG/pos_dists", avg.pos_dists.avg, global_step=current_batch)
        writer.add_scalar("Global_AVG/neg_dists", avg.pos_dists.avg, global_step=current_batch)

        # log to logger
        if current_batch % log_interval == 0:
            print_speed(current_batch, batch_time, total_batch, "global")
            logger.info("\n current batch information:\n epoch: {0} | batch_time {1:5f} | triplet_loss: {2:.5f} | pos_dists: {3:.5f} | neg_dists: {4:.5f} \n".format(epoch + 1, avg.time.val, avg.triplet_loss.val, avg.pos_dists.val, avg.neg_dists.val))
            # logger.info("\n current global average information:\n epoch: {0} | batch_time {1:5f} | triplet_loss: {2:.5f} | pos_dists: {3:.5f} | neg_dists: {4:.5f} \n".format(epoch + 1, avg.time.avg, avg.triplet_loss.avg, avg.pos_dists.avg, avg.neg_dists.avg))
    else:
        # validate model on every epoch?
        pass

    # Save model checkpoint
    state = {
        'epoch': epoch + 1,
        'embedding_dimension': embedding_dim,
        'batch_size_training': batch_size,
        'model_state_dict': model.state_dict(),
        'model_architecture': backbone,
        'optimizer_model_state_dict': optimizer_model.state_dict()
    }

    # TODO: 写一下protocal
    # For storing data parallel model's state dictionary without 'module' parameter
    if flag_train_multi_gpu:
        state['model_state_dict'] = model.module.state_dict()

    # Save model checkpoint
    torch.save(state, os.path.join(experiment_snap_folder, 'model_{}_triplet_epoch_{}.pt').format(backbone, epoch + 1))


# TODO:
# 1. 增加模型表现的tensorboard展示
# 2. 增加validation部分
# 3. 写readme.




















    






