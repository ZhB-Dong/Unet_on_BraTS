import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch,out_ch)
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch,out_ch)

    # def forward(self, x1, x2):
    #     x1 = self.up(x1) # [512,56,56]
    #     # print(x1.size())
    #     # Corp X2 to X1 size and cat X1 and X2
    #     _, th, tw = x1.size()
    #     _, h, w = x2.size() # [512, 64, 64]
    #     top = (h - th) // 2
    #     left = (w - tw) // 2
    #     x2 = TF.crop(x2, top, left, th, tw)
    #
    #     x = torch.cat([x2,x1], dim=0)
    #     return self.conv(x)

    def forward(self, x1, x2):
        x1 = self.up(x1)  # 上采样
        # print(x1.size())
        # print(x2.size())
        _, _, th, tw = x1.size()
        _, _, h, w = x2.size()
        top = (h - th) // 2
        left = (w - tw) // 2
        x2 = TF.crop(x2, top, left, th, tw)
        x = torch.cat([x2, x1], dim=1)  # dim=1 为通道维
        return self.conv(x)

# class Up2(nn.Module):
#     def __init__(self, in_ch, out_ch, bilinear=True):
#         super().__init__()
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         else:
#             self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
#         self.conv = DoubleConv(in_ch, out_ch)
#
#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         diffY = x2.size(1) - x1.size(1)
#         diffX = x2.size(2) - x1.size(2)
#         print(x1.size())
#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         print(x1.size())
#         x = torch.cat([x2, x1], dim=0)
#         return self.conv(x)


class OutConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch,out_ch,kernel_size=1)

    def forward(self,x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.inc = DoubleConv(n_channels,64)
        self.down1 = Down(64,128)
        self.down2 = Down(128,256)
        self.down3 = Down(256,512)
        self.down4 = Down(512,1024)
        self.up1 = Up(1024,512)
        self.up2 = Up(512,256)
        self.up3 = Up(256,128)
        self.up4 = Up(128,64)
        self.outc = OutConv(64,n_classes)

    # def forward(self,x):
    #     x = x.squeeze(0)
    #     # print(x.size())
    #     x1 = self.inc(x)
    #     x2 = self.down1(x1)
    #     x3 = self.down2(x2)
    #     x4 = self.down3(x3)
    #     x5 = self.down4(x4)
    #     # print(x5.size())
    #     # print(x4.size())
    #     x = self.up1(x5, x4)
    #     x = self.up2(x, x3)
    #     x = self.up3(x, x2)
    #     x = self.up4(x, x1)
    #     return self.outc(x)

    def forward(self, x):
        # x = x.squeeze(0)  # 不要挤掉 batch 维！
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)



if __name__ == '__main__':
    x1 = torch.randn([1,1 , 512, 512])
    unet = Unet(1,2)
    x = unet(x1)
    print(x.size()) # [2, 388, 388]





