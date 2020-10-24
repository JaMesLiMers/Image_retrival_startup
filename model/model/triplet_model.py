import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletNetModel(nn.Module):
    """
    Triplet Network model protocal

    This model includes a backbone, the detail information and structure are
    implemented at backbone directory. 

    Args:
        backbone: backbone network.
    
    """
    def __init__(self, backbone):
        super(TripletNetModel, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        """
        Default inference of backbone

        The default behavior is go through the backbone, input one sample and 
        output.

        Args:
            x: input sample, it will go through the backbone.
        """
        return self.backbone(x)

    def forward_triplet(self, anchor, positive, negative):
        """
        Inference of triplet sampled image.

        This method will output a triplet information for training triplet 
        network.

        Args:
            anchor: anchor sample(Tensor).
            positive: positive sample(Tensor).
            negative: negative sample(Tensor).

        Return:
            the return format is as follow:
            {
                "anchor_map": anchor image feature map,
                "pos_map": positive image feature map,
                "neg_map": negative image feature map,
                "dist_pos": parewise distance of anchor and positive feature map,
                "dist_neg": parewise distance of anchor and negative feature map
            }

        """
        anchor_out = self.backbone(anchor)
        pos_out = self.backbone(positive)
        neg_out = self.backbone(negative)
        dist_pos = F.pairwise_distance(anchor_out, pos_out, 2)
        dist_neg = F.pairwise_distance(anchor_out, neg_out, 2)
        return {
            "anchor_map": anchor_out,
            "pos_map": pos_out,
            "neg_map": neg_out,
            "dist_pos": dist_pos,
            "dist_neg": dist_neg
        }


