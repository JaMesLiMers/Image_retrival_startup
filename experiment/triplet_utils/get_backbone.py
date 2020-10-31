
import logging

from utils.log_helper import init_log, add_file_handler

from model.backbone.alexnet import AlexNetTriplet
from model.backbone.vggnet import VGG11Triplet, VGG13Triplet, VGG16Triplet, VGG19Triplet
from model.backbone.resnet import Resnet18Triplet, Resnet34Triplet, Resnet50Triplet, Resnet101Triplet, Resnet152Triplet

logger = init_log("global")

def get_backbone(model_architecture, pretrained, embedding_dimension):
    """Select the back bone model
    
    select the back bone model according to the args, current support:
    [
        "Alexnet", 
        "VGG11", "VGG13", "VGG16", "VGG19",
        "Resnet18", "Resnet34", "Resnet50", "Resnet101", "Resnet152", 
    ]

    Args:
        model_architecture: string of architecture name, is one of above architecture.
        pretrained: load pretrained model or not. (pretrained model is torch's imagenet version.)
        embedding_dimention: embedding feature dimension.
    """
    if model_architecture == "Resnet18":
        model = Resnet18Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "Resnet34":
        model = Resnet34Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "Resnet50":
        model = Resnet50Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "Resnet101":
        model = Resnet101Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "Resnet152":
        model = Resnet152Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "Alexnet":
        model = AlexNetTriplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "VGG11":
        model = VGG11Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "VGG13":
        model = VGG13Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "VGG16":
        model = VGG16Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "VGG19":
        model = VGG19Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    else:
        raise NotImplementedError("Please specific a valid backbone name")

    # log info
    logger.info("\nUsing {} model architecture.\n".format(model_architecture))
    return model

if __name__ == "__main__":
    """
    Test backbone model
    """
    pass