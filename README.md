# 🚦 Traffic Sign Recognition using Deep Learning

## 📌 Project Overview

This project implements a deep learning-based Traffic Sign Recognition system using both a **Custom Convolutional Neural Network (CNN)** and **Transfer Learning (MobileNetV2)**.

The goal of the project is to accurately classify traffic signs into 43 different categories using the GTSRB dataset.

The system includes:

* Custom CNN architecture
* Transfer Learning with MobileNetV2
* Model comparison analysis
* Performance visualization (accuracy & loss curves)
* Confusion matrix evaluation
* GUI application for real-time prediction

---

## 🧠 Models Implemented

### 1️⃣ Custom CNN

* Built from scratch
* Multiple convolutional and pooling layers
* Fully connected classification head
* Achieved near-perfect validation performance

### 2️⃣ Transfer Learning (MobileNetV2)

* Pretrained ImageNet backbone
* Two-stage training:

  * Frozen feature extraction
  * Fine-tuning stage
* Improved generalization and stability

---

## 📊 Dataset

* Dataset: **GTSRB (German Traffic Sign Recognition Benchmark)**
* 43 traffic sign classes
* Train and Test split
* Preprocessing includes:

  * Resizing
  * Normalization
  * Brightness analysis
  * Data augmentation

---

## 📈 Model Evaluation

The models were evaluated using:

* Accuracy and Loss curves
* Confusion Matrix (raw & normalized)
* Precision, Recall, and F1-score
* Class imbalance analysis
* Brightness distribution analysis

### Key Results

* Custom CNN achieved very high validation accuracy (~99%)
* Transfer Learning model achieved ~96% after fine-tuning
* Minimal class confusion
* Strong generalization performance

---

## 🖥 GUI Application

The project includes a graphical interface where users can:

* Upload an image
* Choose the model (Custom CNN or MobileNetV2)
* Get real-time prediction
* View predicted traffic sign label

---

## 🛠 Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* PyQt / Tkinter (for GUI)
* Git & Git LFS

---

## 📂 Project Structure

```
Traffic-Sign-Recognition/
│
├── Dataset/
│   ├── Train/
│   ├── Test/
│   ├── Train.csv
│   └── Test.csv
│
├── saved_models/
│
├── app.py
├── model.ipynb
├── requirements.txt
├── model_comparison.csv
│
└── README.md
```

## 🔍 Future Improvements

* Deploy as a web application
* Add real-time camera detection
* Optimize model size for edge devices
* Implement model quantization
* Add explainability (Grad-CAM)

---

## 🎓 Author

Ahmed Ayman Soliman
AI & Data Engineering Student

---

## 📜 License

This project is for educational and research purposes.

---
