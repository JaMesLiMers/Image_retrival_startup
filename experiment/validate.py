import os


import random
import torchvision
import torch
import torch.nn as nn
from utils.average_meter_helper import AverageMeter

from torch.utils.tensorboard import SummaryWriter


import time
import logging
from utils.average_meter_helper import AverageMeter
from utils.log_helper import init_log, add_file_handler, print_speed

from model.loss.triplet_loss import TripletLoss
from model.model.triplet_model import TripletNetModel
from experiment.triplet_utils.get_backbone import get_backbone
from experiment.triplet_utils.get_optimizer import get_optimizer
from experiment.triplet_utils.get_dataloader import get_train_dataloader


logger = init_log("global")


def set_model_gpu_mode(model, cuda):
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


def load_model(backbone, pretrained, embedding_dim, experiment_snap_folder, resume_name, cuda):
    # Instantiate model
    model = get_backbone(model_architecture=backbone, 
                        pretrained=pretrained, 
                        embedding_dimension=embedding_dim)

    model = TripletNetModel(model)

    # Load model to GPU or multiple GPUs if available
    model = set_model_gpu_mode(model, cuda)

    # Resume from a model checkpoint
    start_epoch = 0
    resume_path = os.path.join(experiment_snap_folder, resume_name)
    if resume_name:
        if os.path.isfile(resume_path):
            logger.info("\nLoading checkpoint {} in {} ...\n".format(resume_name, experiment_folder))

        checkpoint = torch.load(resume_path)
        start_epoch = checkpoint['epoch']

        # In order to load state dict for optimizers correctly, model has to be loaded to gpu first
        model.load_state_dict(checkpoint['model_state_dict'])

        logger.info("\nCheckpoint loaded: From checkpoint epoch = {}\n".format(start_epoch))
    else:
        logger.warning("\nWARNING: No checkpoint found at {}!\nInitialize from scratch.\n".format(resume_path))

    return model



def validation(epoch, val_interval, log_interval, test_dataloader, model, loss, writer, cuda, device):
    logger.info("\n------------------------- Start validation -------------------------\n")
    avg_test = AverageMeter()
    # validate model on every epoch if have test_dataloader.
    if test_dataloader is not None and epoch % val_interval == 0:
        # get test batch count
        current_test_batch = 0
        total_test_batch = len(test_dataloader)

        # best img pairs & worst img pairs
        # contains [anchor_imgs, pos_imgs, neg_imgs, loss_performance]
        best_img_triplets = []
        worst_img_triplets = []
        random_img_triplets = []
        random_sample_idx = random.randrange(0, len(test_dataloader)-1)

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

            # batch time & batch count
            current_test_batch += 1
            batch_time = time.time() - batch_start_time_test

            # find best and worst result
            if len(best_img_triplets) == 0:
                best_img_triplets = [anc_imgs, pos_imgs, neg_imgs, loss_value]
                worst_img_triplets = [anc_imgs, pos_imgs, neg_imgs, loss_value]
            else:
                if best_img_triplets[3] > loss_value:
                    best_img_triplets = [anc_imgs, pos_imgs, neg_imgs, loss_value]
                if worst_img_triplets[3] < loss_value:
                    worst_img_triplets = [anc_imgs, pos_imgs, neg_imgs, loss_value]
                if batch_idx == random_sample_idx:
                    random_img_triplets = [anc_imgs, pos_imgs, neg_imgs, loss_value]

            # update avg
            avg_test.update(time=batch_time, triplet_loss=loss_value, pos_dists=pos_dists, neg_dists=neg_dists)
            if current_test_batch % log_interval == 0:            
                print_speed(current_test_batch, batch_time, total_test_batch, "global")
                logger.info("\n current global average information:\n epoch: {0} | batch_time {1:5f} | triplet_loss: {2:.5f} | pos_dists: {3:.5f} | neg_dists: {4:.5f} \n".format(epoch + 1, avg_test.time.avg, avg_test.triplet_loss.avg, avg_test.pos_dists.avg, avg_test.neg_dists.avg))

        else:
            writer.add_scalar("Validate/Loss/train", avg_test.triplet_loss.avg, global_step=epoch)
            writer.add_scalar("Validate/Other/pos_dists", avg_test.pos_dists.avg, global_step=epoch)
            writer.add_scalar("Validate/Other/neg_dists", avg_test.neg_dists.avg, global_step=epoch)
            if len(best_img_triplets) != 0:
                best_grid_anc = torchvision.utils.make_grid(best_img_triplets[0])
                best_grid_pos = torchvision.utils.make_grid(best_img_triplets[1])
                best_grid_neg = torchvision.utils.make_grid(best_img_triplets[2])

                worst_grid_anc = torchvision.utils.make_grid(worst_img_triplets[0])
                worst_grid_pos = torchvision.utils.make_grid(worst_img_triplets[1])
                worst_grid_neg = torchvision.utils.make_grid(worst_img_triplets[2])

                random_grid_anc = torchvision.utils.make_grid(random_img_triplets[0])
                random_grid_pos = torchvision.utils.make_grid(random_img_triplets[1])
                random_grid_neg = torchvision.utils.make_grid(random_img_triplets[2])

                writer.add_image("Validate/Other/best_case/anchor_img", best_grid_anc, global_step=epoch)
                writer.add_image("Validate/Other/best_case/pos_img", best_grid_pos, global_step=epoch)
                writer.add_image("Validate/Other/best_case/neg_img", best_grid_neg, global_step=epoch)

                writer.add_image("Validate/Other/worst_case/anchor_img", worst_grid_anc, global_step=epoch)
                writer.add_image("Validate/Other/worst_case/pos_img", worst_grid_pos, global_step=epoch)
                writer.add_image("Validate/Other/worst_case/neg_img", worst_grid_neg, global_step=epoch)

                writer.add_image("Validate/Other/random_case/anchor_img", random_grid_anc, global_step=epoch)
                writer.add_image("Validate/Other/random_case/pos_img", random_grid_pos, global_step=epoch)
                writer.add_image("Validate/Other/random_case/neg_img", random_grid_neg, global_step=epoch)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Validate triplet network')

    # Dataset 
    # TODO: Add more support
    parser.add_argument('--dataset', type=str, default="MNIST",
                        choices=["MNIST"],
                        help='Which dataset to use for testing.')

    # Batch size
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for testing (default: 64)')

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
                        help='Which backbone to use for testing.')

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

    # Other

    # Resume pretrain
    parser.add_argument('--resume_name', default='model_Alexnet_triplet_epoch_10.pt', type=str,
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
    resume_name = args.resume_name
    seed = args.seed
    no_cuda = args.no_cuda
    log_interval = args.log_interval
    name = args.name


    """
    Decide wether to use cuda, set random seed and init the logger & average meter,
    init experiment folder.
    """

    # set cuda
    cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # set seed 
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    # Create experiment folder structure
    curernt_file_path = os.path.dirname(os.path.abspath(__file__))
    experiment_folder = os.path.join(curernt_file_path, name)
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
    train_dataloader, test_dataloader = get_train_dataloader(dataset=dataset, 
                                                            batch_size=batch_size, 
                                                            output_size=image_size, 
                                                            num_worker=num_workers, 
                                                            use_cuda=cuda, 
                                                            pre_process_transform=dataset_pre_processing)

    model = load_model(backbone, False, embedding_dim, experiment_snap_folder, resume_name, cuda)

    loss = TripletLoss()

    validation(1, 1, log_interval, test_dataloader, model, loss, writer, cuda, device)


    