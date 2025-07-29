import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets):
        # sigmoid ）
        # inputs = F.softmax(inputs, dim=0)[1].squeeze()
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()

        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()

        dice = (2. * intersection + self.eps) / (union + self.eps)
        return 1 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_w=0.5):
        super().__init__()
        pos_weight = torch.tensor([5.0]).to('cuda')
        # self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()
        self.bce_weight = bce_w

    def forward(self, inputs, labels):
        # inputs = inputs[-1]
        # labels = labels.squeeze()
        bce = self.bce(inputs,labels.float())
        # bce = F.binary_cross_entropy_with_logits(inputs, labels)
        dice = self.dice(inputs, labels.float())
        # print(bce,' ',dice)
        return self.bce_weight*bce+(1-self.bce_weight)*dice
        # return bce+dice


if __name__ == '__main__':
    in_x = torch.randn([2,388,388])
    label = torch.randn([388,388])

    BCEdice = BCEDiceLoss()
    loss = BCEdice(in_x,label)
    print(loss.item())