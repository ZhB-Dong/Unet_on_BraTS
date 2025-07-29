# Brain Tumor Segmentation on BraTS Dataset using UNet

## ABSTRACT
This project independently reproduces the classic semantic segmentation model **UNet** using the PyTorch framework, and applies it to the **BraTS brain tumor medical imaging dataset** for training and evaluation. Considering that brain tumors are often located near the center of the image, the model is fed with the full image without applying the overlap-tile padding strategy from the original paper. This avoids unnecessary edge interference and memory usage. A **combined loss function of Dice Loss and BCE Loss** is used, which improves sensitivity to small foreground regions while maintaining pixel-level classification accuracy. During training, the loss steadily decreases, and the model achieves high Dice coefficient and Sensitivity on the test set, validating its strong segmentation performance and generalization ability. Additionally, the project implements visualizations of segmentation results to provide an intuitive evaluation of the model’s predictions.

## 1. Introduction
UNet is a popular model for image segmentation and can be considered a classification model. It does not use fully connected layers, so the segmentation map only contains pixels where full context from the input image is available. This allows the model to rely on local semantic cues while reducing interference from distant pixels [[1]](https://arxiv.org/abs/1505.04597).

The network structure is illustrated as follows:  
![Unet](./figures/fig1-unet.png "Unet")

In this work, we independently reproduce the UNet model using PyTorch and train/evaluate it on the **Brain Tumor Image Segmentation (BraTS)** dataset, using **Dice coefficient** and **Sensitivity** as evaluation metrics.

## 2. Dataset
We choose the BraTS dataset, which contains semantically rich medical images:

| Dataset   | Image/Label Count |
|-----------|-------------------|
| Train Set | 1502              |
| Test Set  | 215               |

The dataset loading and mask generation are implemented using [this blog post](https://zhuanlan.zhihu.com/p/1895864478723186793) (`./Dataset/data.py`). An example image and label are shown below:  
![Label](./figures/label.png "label")

## 3. Model
This project independently reimplements the classic semantic segmentation model **UNet**, originally proposed by Ronneberger et al. in 2015 [[1]](https://arxiv.org/abs/1505.04597).

The model performs 5 levels of downsampling via max pooling to capture multi-scale features, and then 5 levels of upsampling, where each upsampled feature map is concatenated with its corresponding encoder feature map to enhance local sensitivity. While the original paper uses **overlap-tile and valid convolutions** to preserve edge information, **we instead use zero-padding on all convolutions** to maintain spatial dimensions and because edge information in brain MRI is often irrelevant (mostly noise). The model implementation is in `./models/unet2.py`.

## 4. Training & Evaluation

### 4.1 Data Preparation
Unlike the original paper, which mirrors image borders via the overlap-tile strategy to ensure contextual continuity, we do not apply this since **brain tumors tend to be centrally located**, and padding introduces memory overhead with little gain. Instead, we focus on **resizing and normalizing** full images.

Batch size is set to 1, and full-size images are fed into the network instead of splitting into patches. This is because tumors are spatially sparse and patch-based training may reduce learning efficiency or overfit to noise.

While the original paper enhances boundary weights in the loss, we omit this because **brain tumor masks are usually single connected regions**, and boundary weights do not improve segmentation quality in this scenario.

### 4.2 Activation Function and Loss Design
In segmentation tasks, combining **Binary Cross-Entropy (BCE) Loss** with **Dice Loss** allows balancing pixel-wise accuracy and overall shape overlap. BCE emphasizes local correctness, helping refine edges, while Dice Loss mitigates foreground-background imbalance by focusing on regional overlap.

We use **Sigmoid** as the activation function because it maps each pixel's output to a probability between 0 and 1, ideal for binary classification. It integrates well with BCE and Dice Loss. Loss functions are implemented in `./utils/loss.py`.

### 4.3 Training and Evaluation Setup
We use an **RTX 4090 GPU** for both training and inference. The training setup is as follows:

| Parameter     | Value     |
|---------------|-----------|
| Learning Rate | 1e-4      |
| Batch Size    | 1         |
| Epochs        | 50        |
| Image Size    | 512×512   |

Training script is located in `./train.py`.

Evaluation metrics:

- **Dice Coefficient** measures the overlap between prediction and ground truth:
  $$
  Dice = \frac{2 \times TP}{2 \times TP + FP + FN}
  $$
- **Sensitivity** measures the proportion of true positives correctly predicted:
  $$
  Sensitivity = \frac{TP}{TP + FN}
  $$

### 4.4 Training Results and Accuracy
Loss curve:
![loss](./figures/loss.png "loss")

As shown, the training loss decreases and stabilizes after around `n` epochs. The validation loss starts to rise after epoch 12, suggesting potential overfitting.

Validation Dice and Sensitivity curves:
![sens](./figures/testDiceSens.png "sens")

Both metrics increase steadily during training but plateau after 12 epochs. Final values approach 0.8, indicating good overlap with ground truth and strong ability to detect tumor regions.

Random prediction results on test samples:
![Segmentation](./figures/seg_res.png "segmentation")

Red (prediction) and green (ground truth) regions show high overlap. The model accurately distinguishes tumor regions from surrounding brain structures, especially in Subjects A and B.

## 5. Discussion
This project presents a complete, independent reproduction of the UNet model. The model is built from scratch using PyTorch and trained on the BraTS dataset. Evaluation results demonstrate that the model effectively distinguishes between healthy and abnormal brain tissue, with high agreement between predicted and annotated masks.

## 6. Future Work
- To improve generalization, we plan to apply data augmentation via **random elastic deformation**, mimicking anatomical variability.
- Explore **3D MRI segmentation** by upgrading the current 2D model to a 3D UNet and sourcing appropriate volumetric datasets.
