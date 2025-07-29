import torch
from torch.utils.data import DataLoader
from Dataset.dataload import NiiSliceUNetDataset
from models.unet import Unet
from utils.loss import BCEDiceLoss
import torch.nn as nn
import torch.nn.init as init
from datetime import datetime
import os

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


# 参数
epochs = 300
batch_size = 1
lr = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DCE_weight = 1

# 数据

image_path = 'Dataset/COVID/tr_im2.nii.gz'
mask_path = 'Dataset/COVID/tr_mask.nii.gz'

train_dataset = NiiSliceUNetDataset(
    image_nii_path=image_path,
    mask_nii_path=mask_path,
    slice_range=(0, 85)
)

test_dataset = NiiSliceUNetDataset(
    image_nii_path=image_path,
    mask_nii_path=mask_path,
    slice_range=(85, 100)
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 模型
model = Unet(n_channels=1, n_classes=1)
model.apply(init_weights_kaiming)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Loss
bce_dice_loss = BCEDiceLoss(bce_w=DCE_weight)

# 训练循环
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
dirname = f'/home/sda2/dzb/MyUnet/Weights/{now.strftime("%Y-%m-%d_%H%M%S")}_unet_weight'
# if dirname in os.listdir() is False:
os.makedirs(dirname)


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

    # -------- 验证部分 --------
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

    avg_val_loss = val_loss / len(test_loader)
    print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f}")

    # -------- 保存模型 --------
    # torch.save(model.state_dict(), f"Weights/unet_epoch{epoch+1}.pth")

    # torch.save(model.state_dict(), dirname+f"/unet_epoch{epoch+1}.pth")

    ##best epoch99