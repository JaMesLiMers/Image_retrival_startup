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

    def __init__(self, embedding_dimension=256, pretrained=False):
        super(AlexNetTriplet, self).__init__()
        self.model = alexnet(pretrained=pretrained)

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


if __name__ == "__main__":
    """
    Testing
    """
    import torch
    from model.model.triplet_model import TripletNetModel
    from model.loss.triplet_loss import TripletLoss


    model = AlexNetTriplet()
    model_pretrain = AlexNetTriplet(pretrained=True)

    # init pos, anc, neg image.
    pos_img = torch.randn(4, 3, 227, 227)
    pos_img.requires_grad = True
    anc_img = torch.randn(4, 3, 227, 227)
    anc_img.requires_grad = True
    neg_img = torch.randn(4, 3, 227, 227)
    neg_img.requires_grad = True

    def test_model(model, pos_img, neg_img, anc_img, gpu=False):


        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cuda' if gpu else 'cpu')

        # print model basic infomation 
        print(model)

        # list param
        print('Our list of parameters:', [ np[0] for np in model.named_parameters() ])

        # init triplet net model.
        triplet_model = TripletNetModel(model)
        # test train mode
        # triplet_model.train()
        triplet_model.eval()

        # init loss
        loss_func = TripletLoss()

        model.to(device)
        triplet_model.to(device)
        loss_func.to(device)
        pos_img = pos_img.to(device)
        neg_img = neg_img.to(device)
        anc_img = anc_img.to(device)
        

        pos = triplet_model(pos_img)
        neg = triplet_model(neg_img)
        anc = triplet_model(anc_img)

        # optimizer use adam
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        # check have result and make sense
        triplet_loss = loss_func.forward(
                    anchor=anc,
                    positive=pos,
                    negative=neg)

        # print triplet loss
        print(triplet_loss)

        # test back ward
        # optimizer.zero_grad()
        # triplet_loss.backward()
        # optimizer.step()


    """
    Test default model
    & 
    Test pretrained model
    """
    test_model(model, pos_img, neg_img, anc_img)
    test_model(model_pretrain, pos_img, neg_img, anc_img)

    """
    test on gpu
    """
    test_model(model, pos_img, neg_img, anc_img, gpu=True)
    test_model(model_pretrain, pos_img, neg_img, anc_img, gpu=True)



    
