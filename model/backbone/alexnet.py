import torch.nn as nn
from torch.nn import functional as F
from model.backbone.utils.alexnet_torch import alexnet


class AlexNetTriplet(nn.Module):
    """Constructs a AlexNet model for model training using triplet loss.

    This model is modified according to the FaceNet: 
    https://github.com/tamerthamoqa/facenet-pytorch-vggface2/blob/25250bdac03eaa0ab94230df4db951e9d152849d/models/resnet.py
    What's more, this model loaded the default resnet model, then modify the lastlayer with Linear & BatchNorm1d to make sure the output 
    demention can be modified. 

    Args:
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                   using triplet loss. Defaults to 256.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                           Defaults to False.
    """

    def __init__(self, embedding_dimension=256, hidden_node=4096, pretrained=False):
        super(AlexNetTriplet, self).__init__()
        self.model = alexnet(pretrained=pretrained)

        # Output embedding (将最后一层修改到目标的dimention, TODO: 先这样之后可能还会变)
        self.model.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, hidden_node),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(hidden_node, hidden_node),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_node, embedding_dimension),
            nn.BatchNorm1d(embedding_dimension, eps=0.001, momentum=0.1, affine=True)
        )

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding = self.model(images)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding


if __name__ == "__main__":
    """
    Testing
    """
    model = AlexNetTriplet()

    print(model)
