import torch
from torch.utils.data import DataLoader
from models.unet2 import Unet
from pycocotools.coco import COCO
from Dataset.data import COCOSegmentationDataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random

def overlay_mask(image, mask, color=(1, 0, 0), alpha=0.4):
    """
    image: [H, W], range can be anything (e.g. [-1, 1])
    mask: binary mask
    color: RGB tuple (0~1), e.g. red=(1,0,0)
    alpha: overlay transparency
    """
    # Normalize image to [0, 1] for visualization
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    image_rgb = np.stack([image]*3, axis=-1)  # [H, W, 3]
    overlay = image_rgb.copy()
    for i in range(3):
        overlay[:, :, i] = np.where(mask == 1, color[i], overlay[:, :, i])

    overlay = (1 - alpha) * image_rgb + alpha * overlay
    return np.clip(overlay, 0, 1)


def visualize_overlay(model, dataloader, device='cuda', threshold=0.5, num_samples=5):
    model.eval()
    model.to(device)
    # count = 0

    with torch.no_grad():

        # images, masks = next(iter(dataloader))  # images: [B, 1, H, W], masks: [B, 1, H, W]

        # 在这个 batch 内随机取一个样本
        i = random.randint(0, 100 - 1)
        image, mask = dataloader.dataset[i]  # image: [1, H, W], mask: [1, H, W]
        image = image.unsqueeze(0)  # [1, H, W]
        mask = mask.unsqueeze(0)  # [1, H, W]
        images = image.to(device)
        masks = mask.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs)
        preds = (probs > threshold).float()

        # for i in range(images.size(0)):
        # i = random.randint(0, images.size(0) - 1)
        print(i)
        img_np = images[0][0].cpu().numpy()
        pred_np = preds[0][0].cpu().numpy()
        mask_np = masks[0][0].cpu().numpy()

        overlay_pred = overlay_mask(img_np, pred_np, color=(1, 0, 0), alpha=0.5)
        overlay_gt   = overlay_mask(img_np, mask_np, color=(0, 1, 0), alpha=0.5)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_np, cmap='gray')
        axes[0].set_title('Original Image')
        axes[1].imshow(overlay_pred)
        axes[1].set_title('Prediction Overlay (Red)')
        axes[2].imshow(overlay_gt)
        axes[2].set_title('Ground Truth Overlay (Green)')

        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f'results_{i}')

        # count += 1
        # if count >= num_samples:
        return



def compute_metrics(preds, targets, eps=1e-6):
    """
    preds: [B, 1, H, W] 0/1 tensor
    targets: [B, 1, H, W] 0/1 tensor
    """
    preds = preds.float()
    targets = targets.float()

    TP = (preds * targets).sum(dim=(1, 2, 3))
    FP = (preds * (1 - targets)).sum(dim=(1, 2, 3))
    FN = ((1 - preds) * targets).sum(dim=(1, 2, 3))

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

dirname = f'/home/sda2/dzb/MyUnet/Weights/2025-07-24_142723_BTIunet_weight/unet_epoch27.pth'
# model
model = Unet(n_channels=3, n_classes=1)
# model.apply(init_weights_kaiming)
model.load_state_dict((torch.load(dirname, map_location=device)))
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

visualize_overlay(model,test_loader)
