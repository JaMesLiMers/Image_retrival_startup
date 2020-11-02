from utils.log_helper import init_log
from model.loss.triplet_loss import TripletLoss

logger = init_log("global")

def get_loss(cfg: dict):
    """Select the loss

    select the loss according to the config's, current support:

    ["triplet", ]

    Args:
        cfg: Dict class that must contains required parameter.

    Return:
        A loss function that defined by following config entry:

            loss_name: specific loss
    """
    # get patten from config file
    loss_name = cfg["loss_name"]

    # check cfg
    must_include = {
                    "triplet": ["margin", "norm_digree", "reduction"],
                    }
    for i in must_include.keys():
        if i == loss_name:
            for j in must_include[i]:
                assert j in cfg[i].keys(), "The config file must include {} part for loadding {} loss".format(j, i)


    if loss_name == "triplet":
        # with default args
        loss_model = TripletLoss(margin=cfg[loss_name]["margin"], 
                                 p=cfg[loss_name]["norm_digree"], 
                                 reduction=cfg[loss_name]["reduction"])
        
    else:
        raise NotImplementedError("Please specific a valid loss name")

    logger.info("\nUsing {} loss.\n".format(loss_name))

    return loss_model
