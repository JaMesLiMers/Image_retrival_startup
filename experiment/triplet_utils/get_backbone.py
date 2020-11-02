# logger
from utils.log_helper import init_log

# Backbones
from model.backbone.alexnet import AlexNetTriplet
from model.backbone.vggnet import VGG11Triplet, VGG13Triplet, VGG16Triplet, VGG19Triplet
from model.backbone.resnet import Resnet18Triplet, Resnet34Triplet, Resnet50Triplet, Resnet101Triplet, Resnet152Triplet

logger = init_log("global")

def get_backbone(cfg: dict):
    """Select the back bone model
    
    select the back bone model according to the config's, current support:
    [
        "Alexnet", 
        "VGG11", "VGG13", "VGG16", "VGG19",
        "Resnet18", "Resnet34", "Resnet50", "Resnet101", "Resnet152", 
    ]

    Args:
        cfg: Dict class that must contains required parameter.

    Return:
        nn.Model that defined by following config entry:

            "backbone_name"  : (str) string of architecture name, is one of above architecture.
            "pretrained"          : (bool) load pretrained model or not. (pretrained model is torch's imagenet version.)
            "embedding_dim" : (int) embedding feature dimension.
    """
    # check cfg
    must_include = ["backbone_name", "pretrained", "embedding_dim"]
    for i in must_include:
        assert i in cfg.keys(), "The config file must include {} part for loadding backbone".format(i)

    # get patten from config file
    backbone_name = cfg["backbone_name"]
    pretrained = cfg["pretrained"]
    embedding_dim = cfg["embedding_dim"]

    if backbone_name == "Resnet18":
        model = Resnet18Triplet(
            embedding_dimension=embedding_dim,
            pretrained=pretrained
        )
    elif backbone_name == "Resnet34":
        model = Resnet34Triplet(
            embedding_dimension=embedding_dim,
            pretrained=pretrained
        )
    elif backbone_name == "Resnet50":
        model = Resnet50Triplet(
            embedding_dimension=embedding_dim,
            pretrained=pretrained
        )
    elif backbone_name == "Resnet101":
        model = Resnet101Triplet(
            embedding_dimension=embedding_dim,
            pretrained=pretrained
        )
    elif backbone_name == "Resnet152":
        model = Resnet152Triplet(
            embedding_dimension=embedding_dim,
            pretrained=pretrained
        )
    elif backbone_name == "Alexnet":
        model = AlexNetTriplet(
            embedding_dimension=embedding_dim,
            pretrained=pretrained
        )
    elif backbone_name == "VGG11":
        model = VGG11Triplet(
            embedding_dimension=embedding_dim,
            pretrained=pretrained
        )
    elif backbone_name == "VGG13":
        model = VGG13Triplet(
            embedding_dimension=embedding_dim,
            pretrained=pretrained
        )
    elif backbone_name == "VGG16":
        model = VGG16Triplet(
            embedding_dimension=embedding_dim,
            pretrained=pretrained
        )
    elif backbone_name == "VGG19":
        model = VGG19Triplet(
            embedding_dimension=embedding_dim,
            pretrained=pretrained
        )
    else:
        raise NotImplementedError("Please specific a valid backbone name")

    # log info
    logger.info("\nUsing {} model architecture.\n".format(backbone_name))
    return model

if __name__ == "__main__":
    """
    Test backbone model
    """
    pass