
import logging
import torch.optim as optim
from utils.log_helper import init_log, add_file_handler

logger = init_log("global")

def get_optimizer(optimizer, model, learning_rate=1e-3, momentum=0.5):
    """Select the optimizer

    select the optimizer according to the args, current support:

    ["sgd", "adagrad", "rmsprop", "adam",]

    Args:
        optimizer: specific optimizer
        model: input model
        learning_rate: change lr
    """
    if optimizer == "sgd":
        # with default args except lr/momentum
        optimizer_model = optim.SGD(
            params=model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            dampening=0,
            weight_decay=2e-4,
            nesterov=False
        )

    elif optimizer == "adagrad":
        # with default args except lr/momentum
        optimizer_model = optim.Adagrad(
            params=model.parameters(),
            lr=learning_rate,
            lr_decay=0,
            weight_decay=2e-4,
            initial_accumulator_value=0,
            eps=1e-10
        )

    elif optimizer == "rmsprop":
        # with default args except lr/momentum
        optimizer_model = optim.RMSprop(
            params=model.parameters(),
            lr=learning_rate,
            alpha=0.99,
            eps=1e-08,
            weight_decay=2e-4,
            momentum=momentum,
            centered=False
        )

    elif optimizer == "adam":
        # with default args except lr/momentum
        optimizer_model = optim.Adam(
            params=model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=2e-4,
            amsgrad=False
        )
    
    else:
        raise NotImplementedError("Please specific a valid optimizer name")

    logger.info("\nUsing {} optimizer.\n".format(optimizer))

    return optimizer_model

if __name__ == "__main__":
    """
    Test optimizer
    """
    pass