import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import torchvision.transforms.functional as TF


class NiiSliceUNetDataset(Dataset):
    def __init__(self, image_nii_path, mask_nii_path, slice_range=None, transform=None):
        """
        image_nii_path: 图像 NIfTI 路径 (.nii 或 .nii.gz)
        mask_nii_path: 掩膜 NIfTI 路径
        slice_range: (start, end) 指定切片范围，比如 (0, 85)
        transform: 可选图像增强（用于 image）
        """
        self.image = sitk.GetArrayFromImage(sitk.ReadImage(image_nii_path))  # [D, H, W]
        self.mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_nii_path))   # [D, H, W]

        assert self.image.shape == self.mask.shape, "图像和标签尺寸不一致"

        if slice_range:
            self.image = self.image[slice_range[0]:slice_range[1]]
            self.mask = self.mask[slice_range[0]:slice_range[1]]

        self.transform = transform

    def __len__(self):
        return self.image.shape[0]

    def pad_to_572(self, tensor):
        # pad image and label to 572*572
        _, h, w = tensor.shape
        pad_h = (572 - h) // 2
        pad_w = (572 - w) // 2
        padding = (pad_w, pad_h, pad_w, pad_h)  # left, top, right, bottom
        return TF.pad(tensor, padding, fill=0)

    def center_crop(self, tensor, target_size=(388, 388)):
        # get label crop region
        _, h, w = tensor.shape
        th, tw = target_size
        top = (h - th) // 2
        left = (w - tw) // 2
        return tensor[:, top:top+th, left:left+tw]

    def __getitem__(self, idx):
        # 提取一张 2D 切片
        img_slice = self.image[idx].astype('float32')  # 灰度图
        # mask_slice = self.mask[idx].astype('uint8')  # 掩膜（0/1）
        mask_slice = (self.mask[idx] == 1).astype('uint8')  # 只保留 label==1 区域

        # 转为 tensor: [1, H, W]
        image = TF.to_tensor(img_slice)
        #image = torch.clamp(image, -1.6065e+03, 600)
        #image = 1024*(image + 1000) / 1400  # → [0, 1]
        #image = (image * 2 - 1)  # → [-1, 1]
        mask = TF.to_tensor(mask_slice)

        # pad 到 572×572
        image = self.pad_to_572(image)  # [1, 572, 572]
        mask = self.pad_to_572(mask)  # [1, 572, 572]

        # 中心 crop 出 388×388 标签
        mask_crop = self.center_crop(mask)  # [1, 388, 388]

        # 可选图像增强（仅对图像）
        if self.transform:
            image = self.transform(image)

        return image, mask_crop


# if __name__ == '__main__':
#     train_dataset = NiiSliceUNetDataset(
#         image_nii_path=image_path,
#         mask_nii_path=mask_path,
#         slice_range=(0, 85)
#     )
#
#     test_dataset = NiiSliceUNetDataset(
#         image_nii_path=image_path,
#         mask_nii_path=mask_path,
#         slice_range=(85, 100)
#     )
#
#     from torch.utils.data import DataLoader
#
#     train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#     for x, y in train_loader:
#         print(x.shape)  # [B, 1, 572, 572]
#         print(y.shape)  # [B, 1, 388, 388]
#         break