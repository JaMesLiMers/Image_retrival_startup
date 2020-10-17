import torch 
import torch.nn as nn

class TripletLoss(nn.Module):
    """
    包装后的Torch的Triplet loss类
    
    在初始化的时候标明了超参数, 使用方法:
        loss = TripletLoss()
        loss(anchor, positive, negative)

    更多的文档请参考: 
    https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html

    超参数:
        margin: Triplet loss margin, 默认是 1.0
        p: The norm degree for pairwise distance, 默认是 2
        reduction: loss 求出来后的降维方式, two options: "mean" & "sum"
    """
    def __init__(self):
        super(TripletLoss, self).__init__()
        """
        初始化超参数, 注册loss函数
        """
        self.margin=1.0
        self.p=2
        self.reduction="mean"
        self.loss=nn.TripletMarginLoss(margin=self.margin, p=self.p, reduction=self.reduction)

    def forward(self, anchor, positive, negative):
        """
        Triplet loss 的前向传播.
        
        输入的向量维度为: [B, D]

        Args:
            anchor: 基本点的特征
            positive: 正样本点的特征(要拉近)
            negative: 副样本点的特征(要拉远)
        """
        return self.loss(anchor, positive, negative)


if __name__ == "__main__":
    """
    测试, 由于是包装类, 测试和原来的类的功能是否一致即可
    """
    # our class
    loss = TripletLoss()
    anchor = torch.randn(100, 128, requires_grad=True)
    positive = torch.randn(100, 128, requires_grad=True)
    negative = torch.randn(100, 128, requires_grad=True)
    output = loss(anchor, positive, negative)
    print(output)
    print(output.backward())

    # default class in torch
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    output = triplet_loss(anchor, positive, negative)
    print(output)
    print(output.backward())

