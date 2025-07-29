# UNet Brain Tumor Segmentation on BraTS Dataset

This project implements a full **PyTorch reproduction of the UNet** model for brain tumor segmentation using the **BraTS** dataset. The goal is to segment tumor regions from brain MRI scans with high accuracy and generalization.

## Highlights
- Full **independent reproduction** of the UNet architecture
- Trained on **BraTS** dataset (512×512 2D MRI slices)
- Uses **BCE + Dice Loss** to handle class imbalance and optimize both pixel and region-level accuracy
- Avoids overlap-tile padding (original UNet) to save memory and training time
- Includes **visual evaluation** of predictions vs ground truth

## Dataset
- [BraTS Dataset](https://www.med.upenn.edu/cbica/brats2020/data.html)
- 1500+ training images, 200+ test images
- MRI scans with pixel-wise labeled tumor masks

## Training Setup
| Parameter     | Value     |
|---------------|-----------|
| Model         | UNet      |
| Epochs        | 50        |
| Batch Size    | 1         |
| Image Size    | 512×512   |
| Loss          | BCE + Dice Loss |
| Activation    | Sigmoid   |

## Metrics
- **Dice Coefficient**  
- **Sensitivity**

Both metrics reached ~0.8 on test set.

## Project Structure
.
├── models/ # UNet architecture (unet2.py)
├── utils/ # Loss functions, metrics
├── Dataset/ # Data loading, preprocessing
├── figures/ # Result visualizations
├── train.py # Training script
└── README.md # Project summary

## Sample Result

![segmentation](./figures/seg_res.png)

## Future Work
- Integrate elastic data augmentation for better generalization
- Explore 3D-UNet with volumetric (3D) MRI data

---

**Author**: _Zhanbin, Dong_  
**Framework**: PyTorch  
**GPU**: NVIDIA RTX 4090  