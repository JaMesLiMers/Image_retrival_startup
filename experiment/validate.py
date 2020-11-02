import os
import time
import torch
import logging
from torch.utils.tensorboard import SummaryWriter

# utility
from omegaconf import OmegaConf
from utils.loadConfig import load_cfg
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


def validation(epoch, log_interval, test_dataloader, model, loss, writer, device):
    """Validate on test dataset.

    Current validation is only for loss, pos|neg_distance.
    In future, we will add more validation like MAP5|10|50|100. 
    (maybe in another file.)

    Args:
        log_interval:
            How many time will the logger log once.
        test_dataloader:
            It should not be none! A Triplet dataloader to validate data.
        model:
            The model that used to test on dataset.
        loss: 
            Loss metric.
        writer:
            Tensorboard writer
        device: 
            Device that model compute on

    Return:
        epoch avrage value:
            triplet_loss, pos_dists, neg_dists
    
    """
    logger.info("\n------------------------- Start validation -------------------------\n")
    # epoch average meter
    avg_test = AverageMeter()

    # get test batch count
    current_test_batch = 0
    total_test_batch = len(test_dataloader)

    # check dataloader is not None
    assert test_dataloader is not None, "test_dataloader should not be None."

    for batch_idx, batch_sample in enumerate(test_dataloader):
        # Skip last iteration to avoid the problem of having different number of tensors while calculating
        # averages (sizes of tensors must be the same for pairwise distance calculation)
        if batch_idx + 1 == len(test_dataloader):
            continue

        # switch to evaluation mode.
        for param in model.parameters():
            param.requires_grad = False
        model.eval()


        # start time counting
        batch_start_time_test = time.time()

        # Forward pass - compute embeddings
        anc_imgs = batch_sample['anchor_img']
        pos_imgs = batch_sample['pos_img']
        neg_imgs = batch_sample['neg_img']

        pos_cls = batch_sample['pos_cls']
        neg_cls = batch_sample['neg_cls']

        # move to device
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

        # batch time & batch count
        current_test_batch += 1
        batch_time = time.time() - batch_start_time_test

        # update avg
        avg_test.update(time=batch_time, triplet_loss=loss_value, pos_dists=pos_dists, neg_dists=neg_dists)
        if current_test_batch % log_interval == 0:            
            print_speed(current_test_batch, batch_time, total_test_batch, "global")
            logger.info("\n current global average information:\n batch_time {0:5f} | triplet_loss: {1:.5f} | pos_dists: {2:.5f} | neg_dists: {3:.5f} \n".format(avg_test.time.avg, avg_test.triplet_loss.avg, avg_test.pos_dists.avg, avg_test.neg_dists.avg))
    else:
        writer.add_scalar("Validate/Loss/train", avg_test.triplet_loss.avg, global_step=epoch)
        writer.add_scalar("Validate/Other/pos_dists", avg_test.pos_dists.avg, global_step=epoch)
        writer.add_scalar("Validate/Other/neg_dists", avg_test.neg_dists.avg, global_step=epoch)

    return avg_test.triplet_loss.avg, avg_test.pos_dists.avg, avg_test.neg_dists.avg


if __name__ == "__main__":
    """
    单独Validate模型使用, 可以直接进行validate
    """
    import argparse

    parser = argparse.ArgumentParser(description='Validate triplet network')

    # config file name
    parser.add_argument('--config_name', default='MNIST_Alexnet_triplet_train.yml', type=str,
                        help='name of config file')

    args = parser.parse_args()

    # get config file name.
    config_name = args.config_name

    """
    Decide wether to use cuda, set random seed and init the logger & average meter,
    init experiment folder.
    """
    # get config folder
    curernt_file_path = os.path.dirname(os.path.abspath(__file__))
    experiment_config_folder = os.path.join(curernt_file_path, "config")

    # get cfg file
    cfg = load_cfg(experiment_config_folder, config_name)

    # 要从cfg里加载的东西
    # general的设置
    experiment_name = cfg["experiment_name"] 
    resume_name = cfg["resume_name"]         
    experiment_seed = cfg["experiment_seed"]
    dont_use_cuda = cfg["dont_use_cuda"]
    # log的设置
    log_interval = cfg["log_interval"]

    # set cuda
    cuda = not dont_use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # set seed 
    torch.manual_seed(experiment_seed)
    if cuda:
        torch.cuda.manual_seed(experiment_seed)

    # Create experiment folder structure
    curernt_file_path = os.path.dirname(os.path.abspath(__file__))
    experiment_folder = os.path.join(curernt_file_path, "all_experiment", experiment_name)
    experiment_snap_folder = os.path.join(experiment_folder, "snap")
    experiment_board_folder = os.path.join(experiment_folder, "board_validate")
    os.makedirs(experiment_folder, exist_ok=True)
    os.makedirs(experiment_snap_folder, exist_ok=True)
    os.makedirs(experiment_board_folder, exist_ok=True)

    # init board writer
    writer = SummaryWriter(experiment_board_folder)

    # get log
    add_file_handler("global", os.path.join(experiment_folder, 'validate.log'), level=logging.INFO)

    # get dataset 
    dataset_pre_processing = []
    train_dataloader, test_dataloader = get_train_dataloader(cfg=cfg,
                                                            use_cuda=cuda, 
                                                            pre_process_transform=dataset_pre_processing)

    # load model
    model, start_epoch = load_model_test(cfg, cuda)

    # get evaluation loss
    loss = get_loss(cfg)

    # start validte model
    validation(start_epoch, log_interval, test_dataloader, model, loss, writer, device)


    