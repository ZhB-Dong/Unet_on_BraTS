import torch
from torch.utils.data import DataLoader
from Dataset.dataload2 import NiiSliceUNetDataset
from models.unet2 import Unet
from utils.loss import BCEDiceLoss
import torch.nn as nn
import torch.nn.init as init
from datetime import datetime
import os
from pycocotools.coco import COCO
from Dataset.data import COCOSegmentationDataset
import torchvision.transforms as transforms
import numpy as np

def init_weights_kaiming(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
def compute_metrics(preds, targets, eps=1e-6):
    """
    preds: [B, 1, H, W] 0/1 tensor
    targets: [B, 1, H, W] 0/1 tensor
    """
    preds = preds.float()
    targets = targets.float()

    TP = (preds * targets).sum(dim=(1,2,3))
    FP = (preds * (1 - targets)).sum(dim=(1,2,3))
    FN = ((1 - preds) * targets).sum(dim=(1,2,3))

    dice = (2 * TP) / (2 * TP + FP + FN + eps)
    sensitivity = TP / (TP + FN + eps)

    return dice.mean().item(), sensitivity.mean().item()


# params
epochs = 50
batch_size = 1
lr = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DCE_weight = 0.5

# data

# image_path = 'Dataset/COVID/tr_im2.nii.gz'
# mask_path = 'Dataset/COVID/tr_mask.nii.gz'


# train_dataset = NiiSliceUNetDataset(
#     image_nii_path=image_path,
#     mask_nii_path=mask_path,
#     slice_range=(0, 85)
# )
#
# test_dataset = NiiSliceUNetDataset(
#     image_nii_path=image_path,
#     mask_nii_path=mask_path,
#     slice_range=(85, 100)
# )

# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


train_dir = '/home/sda2/dzb/MyUnet/BrainTumorImage/train'
val_dir = '/home/sda2/dzb/MyUnet/BrainTumorImage/valid'
test_dir = '/home/sda2/dzb/MyUnet/BrainTumorImage/test'

train_annotation_file = '/home/sda2/dzb/MyUnet/BrainTumorImage/train/_annotations.coco.json'
test_annotation_file = '/home/sda2/dzb/MyUnet/BrainTumorImage/test/_annotations.coco.json'
val_annotation_file = '/home/sda2/dzb/MyUnet/BrainTumorImage/valid/_annotations.coco.json'

train_coco = COCO(train_annotation_file)
val_coco = COCO(val_annotation_file)
test_coco = COCO(test_annotation_file)

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

train_dataset = COCOSegmentationDataset(train_coco, train_dir, transform=transform)
val_dataset = COCOSegmentationDataset(val_coco, val_dir, transform=transform)
test_dataset = COCOSegmentationDataset(test_coco, test_dir, transform=transform)

# 创建数据加载器
BATCH_SIZE = 1
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# model
model = Unet(n_channels=3, n_classes=1)
model.apply(init_weights_kaiming)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Loss
bce_dice_loss = BCEDiceLoss(bce_w=DCE_weight)

# epoch
# for epoch in range(epochs):
#     model.train()
#     epoch_loss = 0
#     cnt = 1
#     for images, masks in train_loader:
#         print(cnt)
#         images, masks = images.to(device), masks.to(device)
#         outputs = model(images)
#         loss = bce_dice_loss(outputs, masks)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#         cnt+=1
#
#     print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(train_loader):.4f}")
#     torch.save(model.state_dict(), f"Weights/unet_epoch{epoch+1}.pth")
now = datetime.now()
# dirname = f'{now.strftime("%Y-%m-%d_%H%M%S")}_unet_weight'
dirname = f'/home/sda2/dzb/MyUnet/Weights/{now.strftime("%Y-%m-%d_%H%M%S")}_BTIunet_weight'
# if dirname in os.listdir() is False:
os.makedirs(dirname)

best_dice = 0
best_sens = 0

dice_all = []
sens_all = []
avg_train_loss_all = []
avg_test_loss_all = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = bce_dice_loss(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)
    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")

    # -------- val --------
    model.eval()
    dice_list = []
    sen_list = []
    val_loss = 0.0

    with torch.no_grad():
        for val_images, val_masks in test_loader:
            val_images, val_masks = val_images.to(device), val_masks.to(device)
            val_outputs = model(val_images)
            probs = torch.sigmoid(val_outputs)
            preds = (probs>0.3).float()
            # print(preds.sum(), val_masks.sum())
            # print(val_outputs.size())
            loss = bce_dice_loss(val_outputs, val_masks)
            val_loss += loss.item()
            dice, sen = compute_metrics(preds, val_masks)
            dice_list.append(dice)
            sen_list.append(sen)
    avg_dice = sum(dice_list) / len(dice_list)
    avg_sen = sum(sen_list) / len(sen_list)

    print(f"Validation Dice: {avg_dice:.4f}, Sensitivity: {avg_sen:.4f}")

    if avg_dice > best_dice:
        print('New best dice')
        best_dice = avg_dice
        best_sens = avg_sen
        torch.save(model.state_dict(), dirname+f"/unet_epoch{epoch+1}.pth")

    avg_val_loss = val_loss / len(test_loader)
    print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f}")

    # -------- save model --------
    # torch.save(model.state_dict(), f"Weights/unet_epoch{epoch+1}.pth")
    avg_train_loss_all.append(avg_train_loss)
    avg_test_loss_all.append(avg_val_loss)
    dice_all.append(avg_dice)
    sens_all.append(avg_sen)

paras = {}
paras['dice_all'] = dice_all
paras['sens_all'] = sens_all
paras['avg_train_loss_all'] = avg_train_loss_all
paras['avg_test_loss_all'] = avg_test_loss_all
np.save(dirname + f"/train_paras.npy", paras)
# np.save()
    ##best epoch99