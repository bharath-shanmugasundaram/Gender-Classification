# 👤 Gender Classification using AlexNet (PyTorch)

This project builds a **Gender Classification** system using a modified **AlexNet** architecture trained on the `myvision/gender-classification` dataset from HuggingFace. The goal is to classify images into **Male** or **Female** categories using deep convolutional neural networks.

---

## 🚀 Project Overview

This project demonstrates how classical CNN architectures like **AlexNet** can still perform strongly on modern small-to-medium vision tasks.  
Key steps included:

- Loading and preprocessing a real-world image dataset using **HuggingFace Datasets**
- Applying standard **ImageNet-style transforms**
- Building a clean, modular implementation of **AlexNet** in PyTorch
- Training the model on GPU/Metal (MPS for macOS)
- Evaluating both global accuracy and **per-class accuracy**
- Testing on custom images
- Saving the trained model for deployment or reuse

---

## 📂 Dataset

- **Source:** `myvision/gender-classification` (HuggingFace)
- **Type:** Image classification
- **Classes:**  
  - `0` → Female  
  - `1` → Male  

All images pass through a standard preprocessing pipeline:

- Resize / Tensor conversion  
- Mean–Std normalization identical to ImageNet

---

## 🧠 Model Architecture — AlexNet

AlexNet is a landmark deep CNN architecture that introduced:

- Large kernel convolution layers  
- Overlapping max pooling  
- ReLU activations  
- High-capacity fully connected layers (4096 neurons each)  
- Dropout-based regularization  

The model used here is adapted for **2-class classification**.

---

## 🏋️ Training

The model is trained for multiple epochs using:

- **Loss:** CrossEntropyLoss  
- **Optimizer:** Adam (`lr=0.0001`)  
- **Batch size:** 32  
- **Device:** MPS (Apple Silicon) or CPU  

Training loop tracks epoch-wise loss to ensure proper convergence.

---

## 📊 Evaluation

Two evaluation metrics are computed:

### ✔️ Overall Accuracy  
Calculated on the test set using predicted vs actual labels.

### ✔️ Per-Class Accuracy  
Breakdown of accuracy per category (Male / Female) for deeper insight into class-wise performance.

---

## 🖼️ Prediction on Custom Images

After training, the model can infer gender from external images:

- Load image with OpenCV  
- Resize to 224×224  
- Apply same preprocessing pipeline  
- Run forward pass through the model  
- Output predicted class label (Male / Female)

---

## 💾 Saving the Model

The trained weights are saved as:





---

## 🛠️ Technologies Used

- **PyTorch**
- **Torchvision**
- **HuggingFace Datasets**
- **OpenCV**
- **Python 3.x**

---

## ⭐ Future Improvements

- Add augmentation (RandomCrop, HorizontalFlip)
- Replace AlexNet with ResNet / EfficientNet
- Hyperparameter tuning for improved accuracy
- Convert to TorchScript / ONNX for deployment
- Build a Streamlit or Gradio UI

---

## 🙌 Acknowledgements

Special thanks to:

- **HuggingFace** for open datasets  
- **PyTorch** for enabling flexible deep learning workflows  

---

If you want, I can generate:

✅ A better-styled README  
✅ A project folder + file structure  
✅ A downloadable `.md` file  
