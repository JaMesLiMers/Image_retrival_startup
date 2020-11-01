
import logging
import torch.optim as optim
from utils.log_helper import init_log, add_file_handler
from model.loss.triplet_loss import TripletLoss

logger = init_log("global")

def get_loss(loss_name):
    """Select the loss

    select the loss according to the args, current support:

    ["triplet", ]

    Args:
        loss_name: specific loss

    Return:
        A loss function.
    """
    if loss_name == "triplet":
        # with default args
        loss_model = TripletLoss(margin=1.0, 
                                 p=2, 
                                 reduction="mean")
        
    else:
        raise NotImplementedError("Please specific a valid loss name")

    logger.info("\nUsing {} loss.\n".format(loss_name))

    return loss_model
