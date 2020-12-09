import os
import re
import torch
import torch.nn as nn
from utils.log_helper import init_log
from torch.utils.tensorboard import SummaryWriter
from model.model.triplet_model import TripletNetModel
from experiment.triplet_utils.get_backbone import get_backbone
from experiment.triplet_utils.get_optimizer import get_optimizer
from experiment.triplet_utils.get_loss import get_loss

logger = init_log("global")
re_digits = re.compile(r'(\d+)')

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


def get_pt_file(path):
    """
    get all .pt file name in the folder

    Args:
        path: folder to get all .pt name
    Return:
        A list of all .pt file
    """

    file_list = []

    f_list = os.listdir(path)
    # print f_list
    for i in f_list:
        # os.path.splitext():分离文件名与扩展名
        if os.path.splitext(i)[1] == '.pt':
            file_list.append(i)
    return file_list


def load_all_train(cfg, cuda):
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
        cfg: config file that used following part:

            backbone_name: 
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

    resume_name = cfg["resume_name"]
    experiment_snap_folder = os.path.join("experiment", "all_experiment", cfg["experiment_name"], "snap")
    
    # Instantiate model
    model = get_backbone(cfg=cfg)

    model = TripletNetModel(model)

    # Load model to GPU or multiple GPUs if available
    model, flag_train_multi_gpu = set_model_gpu_mode_multi(model, cuda)

    # Set optimizer
    optimizer_model = get_optimizer(cfg=cfg,
                                    model=model)

    # Set loss function
    loss = get_loss(cfg=cfg)

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

    return model, optimizer_model, loss, start_epoch, flag_train_multi_gpu



def load_model_test(cfg, cuda):
    """Initialize and load model snap for testing.

    This function is aimed to get the model part for testing on single GPU;

    The backbone model is specified by name and used the "get"
    function to load into memory, what's more, according to the resume_name, 
    the model checkpoint can be loaded. in this case, the start epoch may not 
    is initialized value 0.(but is may not important in testing.)

    This function is mainly for testing, so multi GPU model is not in our 
    consideration.

    Args:
        cfg: config file that used following part:
            backbone: 
                Model backbone name, can be:
                    ["Alexnet", 
                    "VGG11", "VGG13", "VGG16", "VGG19",
                    "Resnet18", "Resnet34", "Resnet50", "Resnet101", "Resnet152",]
            pretrained: 
                Wether use the ImageNet pretrained weight for model parameter init.
            embedding_dim: 
                The output feature's dimention.
            resume_name: 
                resume snap '.pt' file name.
        cuda: 
            use cuda or not.
    Return:
        models: (list of model)
            models that resumed by snap file and loaded into GPU if use cuda.
        start_epochs: (list of int)
            Resumed last epoch of the models, is not resume then is 0;
    """
    resume_name = cfg["resume_name"]
    experiment_snap_folder = os.path.join("experiment", "all_experiment", cfg["experiment_name"], "snap")

    models = []
    start_epochs = []
    model_resume_path = []

    # 若模型加载部分为空
    if not resume_name:
        model_resume_path = get_pt_file(experiment_snap_folder)
        # sort the path list by number
        def get_numbers(s):
            pieces = re_digits.split(s)             # 切成数字和非数字
            pieces[1::2] = map(int, pieces[1::2])       # 将数字部分转成整数
            return pieces

        model_resume_path.sort(key=get_numbers)
        if not model_resume_path:
            raise FileNotFoundError("Cannot find any file in the snap file.")
    else:
        model_resume_path.append(resume_name)
    
    for model_snap_name in model_resume_path:
        # Instantiate model
        model = get_backbone(cfg=cfg)

        model = TripletNetModel(model)

        # Load model to GPU or multiple GPUs if available
        model = set_model_gpu_mode_single(model, cuda)

        # Resume from a model checkpoint
        start_epoch = 0
        resume_path = os.path.join(experiment_snap_folder, model_snap_name)
        if model_snap_name:
            if os.path.isfile(resume_path):
                logger.info("\nLoading checkpoint {} in {} ...\n".format(model_snap_name, experiment_snap_folder))

            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']

            # In order to load state dict for optimizers correctly, model has to be loaded to gpu first
            model.load_state_dict(checkpoint['model_state_dict'])

            # add to all results
            models.append(model)
            start_epochs.append(start_epoch)

            logger.info("\nCheckpoint loaded: From checkpoint epoch = {}\n".format(start_epoch))
        else:
            logger.warning("\nWARNING: No checkpoint found at {}!\nInitialize from scratch.\n".format(resume_path))

    return models, start_epochs


def load_model_test_yeild(cfg, cuda):
    """Initialize and load model snap for testing.

    This function is aimed to get the model part for testing on single GPU;

    The backbone model is specified by name and used the "get"
    function to load into memory, what's more, according to the resume_name, 
    the model checkpoint can be loaded. in this case, the start epoch may not 
    is initialized value 0.(but is may not important in testing.)

    This function is mainly for testing, so multi GPU model is not in our 
    consideration.

    Args:
        cfg: config file that used following part:
            backbone: 
                Model backbone name, can be:
                    ["Alexnet", 
                    "VGG11", "VGG13", "VGG16", "VGG19",
                    "Resnet18", "Resnet34", "Resnet50", "Resnet101", "Resnet152",]
            pretrained: 
                Wether use the ImageNet pretrained weight for model parameter init.
            embedding_dim: 
                The output feature's dimention.
            resume_name: 
                resume snap '.pt' file name.
        cuda: 
            use cuda or not.
    Return:
        models: (list of model)
            models that resumed by snap file and loaded into GPU if use cuda.
        start_epochs: (list of int)
            Resumed last epoch of the models, is not resume then is 0;
    """
    resume_name = cfg["resume_name"]
    experiment_snap_folder = os.path.join("experiment", "all_experiment", cfg["experiment_name"], "snap")

    model_resume_path = []

    # 若模型加载部分为空
    if not resume_name:
        model_resume_path = get_pt_file(experiment_snap_folder)
        # sort the path list by number
        def get_numbers(s):
            pieces = re_digits.split(s)             # 切成数字和非数字
            pieces[1::2] = map(int, pieces[1::2])       # 将数字部分转成整数
            return pieces

        model_resume_path.sort(key=get_numbers)
        if not model_resume_path:
            raise FileNotFoundError("Cannot find any file in the snap file.")
    else:
        model_resume_path.append(resume_name)
    
    for model_snap_name in model_resume_path:
        # Instantiate model
        model = get_backbone(cfg=cfg)

        model = TripletNetModel(model)

        # Load model to GPU or multiple GPUs if available
        model = set_model_gpu_mode_single(model, cuda)

        # Resume from a model checkpoint
        start_epoch = 0
        resume_path = os.path.join(experiment_snap_folder, model_snap_name)
        if model_snap_name:
            if os.path.isfile(resume_path):
                logger.info("\nLoading checkpoint {} in {} ...\n".format(model_snap_name, experiment_snap_folder))

            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']

            # In order to load state dict for optimizers correctly, model has to be loaded to gpu first
            model.load_state_dict(checkpoint['model_state_dict'])

            logger.info("\nCheckpoint loaded: From checkpoint epoch = {}\n".format(start_epoch))

            yield model, start_epoch
        else:
            logger.warning("\nWARNING: No checkpoint found at {}!\nInitialize from scratch.\n".format(resume_path))