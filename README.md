# Image-Sharpening-using-Knowledge-Distillation

This project explores the use of Knowledge Distillation (KD) to perform image sharpening—a critical task in low-level computer vision where the goal is to enhance the visual quality of blurry or low-definition images. The approach centers on training a lightweight student model to replicate the sharpening performance of a high-capacity teacher model.

## Project Overview

The primary objective is to reconstruct sharp images from blurred inputs using deep learning. A teacher-student paradigm is used, where the teacher model provides guidance during training in the form of feature representations and predictions. The student model, being more computationally efficient, learns to mimic the teacher’s behavior while producing similar quality results.

This methodology is particularly useful in resource-constrained environments, such as mobile devices or edge computing scenarios, where deploying large models is impractical.

## Theoretical Foundation

**Knowledge Distillation** is a training strategy where a compact model (student) is trained to reproduce the behavior of a more accurate but computationally intensive model (teacher). Traditionally used in classification tasks, this technique is extended here to a regression task—image-to-image translation—specifically for the purpose of sharpening.

The training objective combines multiple loss functions:

- **Pixel-wise Loss (MSE or L1)**: Measures the difference between the student’s and ground truth output.
- **Perceptual Loss (LPIPS)**: Uses pretrained deep features to compare perceptual similarity.
- **Structural Similarity Index (SSIM)** and **Peak Signal-to-Noise Ratio (PSNR)**: Quantitative image quality metrics.

By using both low-level (pixel-wise) and high-level (perceptual) cues, the student model effectively generalizes the sharpening capability learned from the teacher.

## Model Implementation

The project is implemented in PyTorch. The key components include:

1. **Teacher Model**
   - A deep convolutional neural network with multiple layers and skip connections.
   - Trained separately or preloaded for high-fidelity sharpening.

2. **Student Model**
   - A shallower convolutional model.
   - Trained using a combination of ground truth supervision and distillation from the teacher’s intermediate features and output predictions.

3. **Training Pipeline**
   - Input images are blurred versions of high-resolution samples from the DIV2K dataset.
   - The model is trained to minimize a combined loss function across epochs.
   - Evaluation is conducted using PSNR, SSIM, and LPIPS metrics.

4. **Dataset**
   - DIV2K High Resolution dataset, commonly used for image restoration tasks.
   - Automatically downloaded during notebook execution if not already present.

## Sample Results

A sample comparison of input, sharpened output, and original image demonstrates the effectiveness of the student model:

![Sample Comparison](data/0035.png)
In the above example, the sharpened image achieved a PSNR of 43.73 dB and SSIM of 0.9948, indicating high similarity to the original sharp image. These metrics confirm the student model's strong capability to recover details and structure from blurred inputs.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Image-Sharpening-KD.git
   cd Image-Sharpening-KD
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Launch the notebook:
   ```bash
   jupyter notebook Final_ISKD.ipynb
The notebook will download the DIV2K dataset if not already available and perform training and evaluation on the provided samples.

## Results and Analysis
The student model achieves image quality metrics that closely match those of the teacher model while using significantly fewer parameters and compute resources. This highlights the feasibility of deploying the student model in real-world applications that require fast, on-device processing.
Quantitative metrics like PSNR and SSIM indicate fidelity at the pixel and structural level, while perceptual metrics such as LPIPS confirm visual closeness to ground truth.

## Future Work and Improvements
- Several directions can be explored to extend and improve this project:

- Data Augmentation: Introduce motion blur, lens distortion, or noise to increase robustness.

- Model Compression: Apply quantization or pruning to further reduce student model size.

- Edge Deployment: Optimize the student model for deployment on mobile or embedded systems.

- Attention Mechanisms: Incorporate spatial or channel attention to enhance edge restoration.

- Multi-task Learning: Combine sharpening with other image enhancement tasks like denoising or super-resolution.


