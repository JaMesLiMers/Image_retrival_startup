import os
import torch
import torch.nn as nn
from utils.log_helper import init_log
from torch.utils.tensorboard import SummaryWriter
from model.model.triplet_model import TripletNetModel
from experiment.triplet_utils.get_backbone import get_backbone
from experiment.triplet_utils.get_optimizer import get_optimizer
from experiment.triplet_utils.get_loss import get_loss

logger = init_log("global")


def set_model_gpu_mode_single(model, cuda):
    """decide wether use GPU of CPU to run model
    
    Args:
        model: input model, is a nn.Moudle
        cuda: bool, wether use cuda.
    """
    flag_train_gpu = torch.cuda.is_available() & cuda

    if flag_train_gpu and torch.cuda.device_count() == 1:
        model.cuda()
        logger.info('\nRunning model with single-gpu .\n')
    else:
        logger.info('\nRunning model with CPU.\n')

    return model

def set_model_gpu_mode_multi(model, cuda):
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


def load_all_train(backbone, pretrained, embedding_dim, 
                   loss_name, optimizer,
                   experiment_snap_folder, resume_name, cuda):
    """Load or initialize model, optimizer, loss, start_epoch for training.

    This function is aimed to get all model part for training, including:

        backbone model, loss, optimizer, epoch and cuda_flag

    The backbone, loss, optimizer is specified by name and used the "get"
    function to load into memory, what's more, according to the resume_name, 
    the model checkpoint can be loaded. in this case, the start epoch may not 
    is initialized value 0.

    If the model is trained on multi gpu, the model will be wrapped by dataparallel()
    and 'flag_train_multi_gpu' will become True.

    Args:
        backbone: 
            Model backbone name, can be:
                ["Alexnet", 
                "VGG11", "VGG13", "VGG16", "VGG19",
                "Resnet18", "Resnet34", "Resnet50", "Resnet101", "Resnet152",]
        pretrained: 
            Wether use the ImageNet pretrained weight for model parameter init.
        embedding_dim: 
            The output feature's dimention.
        loss_name: 
            Loss name, current support:
                ["triplet",]
        optimizer: 
            Optimizer name, current support:
                ["sgd", "adagrad", "rmsprop", "adam"]
        experiment_snap_folder: 
            where to load model snap.
        resume_name: 
            resume snap '.pt' file name.
        cuda: 
            use cuda or not.

    Return:
        model: 
            A model that resumed by snap file and loaded into GPU if use cuda.
        optimizer: 
            An optimizer that aimed to update model params and resumed by snap file.
        loss:
            A loss function that loaded according to it's name
        start_epoch:
            Resumed last epoch of the model, is not resume then is 0;
        flag_train_multi_gpu:
            Wether trained on multi gpu.
    """
    
    # Instantiate model
    model = get_backbone(model_architecture=backbone, 
                        pretrained=pretrained, 
                        embedding_dimension=embedding_dim)

    model = TripletNetModel(model)

    # Load model to GPU or multiple GPUs if available
    model, flag_train_multi_gpu = set_model_gpu_mode_multi(model, cuda)

    # Set optimizer
    optimizer_model = get_optimizer(optimizer=optimizer,
                                    model=model)

    # Set loss function
    loss = get_loss(loss_name=loss_name)

    # Resume from a model checkpoint
    start_epoch = 0
    resume_path = os.path.join(experiment_snap_folder, resume_name)
    if resume_name:
        if os.path.isfile(resume_path):
            logger.info("\nLoading checkpoint {} in {} ...\n".format(resume_name, experiment_snap_folder))

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

    return model, optimizer, loss, start_epoch, flag_train_multi_gpu



def load_model_test(backbone, pretrained, embedding_dim, 
                    experiment_snap_folder, resume_name, cuda):
    """Initialize and load model snap for testing.

    This function is aimed to get the model part for testing on single GPU;

    The backbone model is specified by name and used the "get"
    function to load into memory, what's more, according to the resume_name, 
    the model checkpoint can be loaded. in this case, the start epoch may not 
    is initialized value 0.(but is may not important in testing.)

    This function is mainly for testing, so multi GPU model is not in our 
    consideration.

    Args:
        backbone: 
            Model backbone name, can be:
                ["Alexnet", 
                "VGG11", "VGG13", "VGG16", "VGG19",
                "Resnet18", "Resnet34", "Resnet50", "Resnet101", "Resnet152",]
        pretrained: 
            Wether use the ImageNet pretrained weight for model parameter init.
        embedding_dim: 
            The output feature's dimention.
        experiment_snap_folder: 
            where to load model snap.
        resume_name: 
            resume snap '.pt' file name.
        cuda: 
            use cuda or not.
    Return:
        model: 
            A model that resumed by snap file and loaded into GPU if use cuda.
        start_epoch:
            Resumed last epoch of the model, is not resume then is 0;
    """
    # Instantiate model
    model = get_backbone(model_architecture=backbone, 
                        pretrained=pretrained, 
                        embedding_dimension=embedding_dim)

    model = TripletNetModel(model)

    # Load model to GPU or multiple GPUs if available
    model = set_model_gpu_mode_single(model, cuda)

    # Resume from a model checkpoint
    start_epoch = 0
    resume_path = os.path.join(experiment_snap_folder, resume_name)
    if resume_name:
        if os.path.isfile(resume_path):
            logger.info("\nLoading checkpoint {} in {} ...\n".format(resume_name, experiment_snap_folder))

        checkpoint = torch.load(resume_path)
        start_epoch = checkpoint['epoch']

        # In order to load state dict for optimizers correctly, model has to be loaded to gpu first
        model.load_state_dict(checkpoint['model_state_dict'])

        logger.info("\nCheckpoint loaded: From checkpoint epoch = {}\n".format(start_epoch))
    else:
        logger.warning("\nWARNING: No checkpoint found at {}!\nInitialize from scratch.\n".format(resume_path))

    return model, start_epoch