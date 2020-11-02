import os
import time
import torch
import logging
from torch.utils.tensorboard import SummaryWriter
from utils.average_meter_helper import AverageMeter
from utils.log_helper import init_log, add_file_handler, print_speed

# get method
from experiment.triplet_utils.get_loss import get_loss
from experiment.triplet_utils.get_backbone import get_backbone
from experiment.triplet_utils.get_optimizer import get_optimizer
from experiment.triplet_utils.get_dataloader import get_train_dataloader

# load model (more eazy way to get model.)
from experiment.triplet_utils.load_model import load_model_test

# init logger
logger = init_log("global")

"""
This file is implement for calculating the MAP@5/10/50/100 for the model.
in this 
"""