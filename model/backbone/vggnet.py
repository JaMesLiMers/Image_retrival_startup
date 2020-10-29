import torch.nn as nn
from torch.nn import functional as F
from model.backbone.utils.vgg_torch import vgg11, vgg13, vgg16, vgg19

class VGG11Triplet(nn.Module):
    """Constructs a VGG11Net model for model training using triplet loss.

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

    def __init__(self, embedding_dimension=256, pretrained=False):
        super(VGG11Triplet, self).__init__()
        self.model = vgg11(pretrained=pretrained)

        # Output embedding (将最后一层修改到目标的dimention, TODO: 先这样之后可能还会变)
        # Following the Bin's answer from: https://discuss.pytorch.org/t/how-to-perform-finetuning-in-pytorch/419/9
        mod = list(self.model.classifier.children())
        mod.pop()
        mod.append(nn.Linear(4096, embedding_dimension))
        mod.append(nn.BatchNorm1d(embedding_dimension, eps=0.001, momentum=0.1, affine=True))
        self.model.classifier = nn.Sequential(*mod)

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding = self.model(images)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

class VGG13Triplet(nn.Module):
    """Constructs a VGG13Net model for model training using triplet loss.

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

    def __init__(self, embedding_dimension=256, pretrained=False):
        super(VGG13Triplet, self).__init__()
        self.model = vgg13(pretrained=pretrained)

        # Output embedding (将最后一层修改到目标的dimention, TODO: 先这样之后可能还会变)
        # Following the Bin's answer from: https://discuss.pytorch.org/t/how-to-perform-finetuning-in-pytorch/419/9
        mod = list(self.model.classifier.children())
        mod.pop()
        mod.append(nn.Linear(4096, embedding_dimension))
        mod.append(nn.BatchNorm1d(embedding_dimension, eps=0.001, momentum=0.1, affine=True))
        self.model.classifier = nn.Sequential(*mod)

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding = self.model(images)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding


class VGG16Triplet(nn.Module):
    """Constructs a VGG16Net model for model training using triplet loss.

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

    def __init__(self, embedding_dimension=256, pretrained=False):
        super(VGG16Triplet, self).__init__()
        self.model = vgg16(pretrained=pretrained)

        # Output embedding (将最后一层修改到目标的dimention, TODO: 先这样之后可能还会变)
        # Following the Bin's answer from: https://discuss.pytorch.org/t/how-to-perform-finetuning-in-pytorch/419/9
        mod = list(self.model.classifier.children())
        mod.pop()
        mod.append(nn.Linear(4096, embedding_dimension))
        mod.append(nn.BatchNorm1d(embedding_dimension, eps=0.001, momentum=0.1, affine=True))
        self.model.classifier = nn.Sequential(*mod)

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding = self.model(images)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding


class VGG19Triplet(nn.Module):
    """Constructs a VGG19Net model for model training using triplet loss.

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

    def __init__(self, embedding_dimension=256, pretrained=False):
        super(VGG19Triplet, self).__init__()
        self.model = vgg19(pretrained=pretrained)

        # Output embedding (将最后一层修改到目标的dimention, TODO: 先这样之后可能还会变)
        # Following the Bin's answer from: https://discuss.pytorch.org/t/how-to-perform-finetuning-in-pytorch/419/9
        mod = list(self.model.classifier.children())
        mod.pop()
        mod.append(nn.Linear(4096, embedding_dimension))
        mod.append(nn.BatchNorm1d(embedding_dimension, eps=0.001, momentum=0.1, affine=True))
        self.model.classifier = nn.Sequential(*mod)

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding = self.model(images)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding