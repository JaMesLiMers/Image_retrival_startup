import torch.optim as optim
from utils.log_helper import init_log

logger = init_log("global")

def get_optimizer(cfg: dict, model):
    """Select the optimizer

    select the optimizer according to the args, current support:

    ["sgd", "adagrad", "rmsprop", "adam",]

    Args:
        cfg: Dict class that must contains required parameter.
        model: input model

    Return: 
        A optimizer init acrroding to the cfg

            optimizer_name: specific optimizer
    """

    optimizer_name = cfg["optimizer_name"]

    # check cfg
    must_include = {
                    "sgd": ["lr", "momentum", "dampening", "weight_decay", "nesterov"],
                    "adagrad": ["lr", "lr_decay", "weight_decay", "initial_accumulator_value", "eps"],
                    "rmsprop": ["lr", "alpha", "eps", "weight_decay", "momentum", "centered"],
                    "adam": ["lr", "betas", "eps", "weight_decay", "amsgrad"],
                    }
    for i in must_include.keys():
        if i == cfg["optimizer_name"]:
            for j in must_include[i]:
                assert j in cfg[i].keys(), "The config file must include {} part for loadding {} loss".format(j, i)



    if optimizer_name == "sgd":
        # with default args except lr/momentum
        optimizer_model = optim.SGD(
            params=model.parameters(),
            lr=1e-3,
            momentum=0.9,
            dampening=0,
            weight_decay=2e-4,
            nesterov=False
        )

    elif optimizer_name == "adagrad":
        # with default args except lr/momentum
        optimizer_model = optim.Adagrad(
            params=model.parameters(),
            lr=1e-3,
            lr_decay=0,
            weight_decay=2e-4,
            initial_accumulator_value=0,
            eps=1e-10
        )

    elif optimizer_name == "rmsprop":
        # with default args except lr/momentum
        optimizer_model = optim.RMSprop(
            params=model.parameters(),
            lr=1e-3,
            alpha=0.99,
            eps=1e-08,
            weight_decay=2e-4,
            momentum=0,
            centered=False
        )

    elif optimizer_name == "adam":
        # with default args except lr/momentum
        optimizer_model = optim.Adam(
            params=model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=2e-4,
            amsgrad=False
        )
    
    else:
        raise NotImplementedError("Please specific a valid optimizer name")

    logger.info("\nUsing {} optimizer.\n".format(optimizer_name))

    return optimizer_model

if __name__ == "__main__":
    """
    Test optimizer
    """
    pass