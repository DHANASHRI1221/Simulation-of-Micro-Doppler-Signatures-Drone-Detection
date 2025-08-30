
# 🛰️ Simulation of Micro-Doppler Signatures & Drone Detection (ML)

## 📌 Project Overview

This repository contains the implementation of **Micro-Doppler signature simulation** and **drone detection using Machine Learning techniques**.
Micro-Doppler signatures arise from micro-motions (e.g., rotor blades of drones, swinging arms, etc.) in radar returns. They are powerful features for target recognition and classification.

The goal of this project is to:

* Simulate **radar micro-Doppler signatures** for different drone and non-drone targets.
* Extract time-frequency features (e.g., spectrograms, STFT, wavelets).
* Train ML models for **drone vs. non-drone detection**.
* Evaluate classification accuracy under noise and varying SNR conditions.

---

## ⚙️ Features

* ✅ Radar echo & micro-Doppler signal simulation.
* ✅ Spectrogram generation using **Short-Time Fourier Transform (STFT)**.
* ✅ Dataset creation for drone vs. non-drone.
* ✅ Machine Learning classification (SVM, CNN, etc.).
* ✅ Performance metrics (Accuracy, Precision, Recall, F1-score).

---

## 🛠️ Tech Stack

* **Languages**: Python (NumPy, SciPy, Matplotlib, scikit-learn, PyTorch/TensorFlow).
* **Signal Processing**: STFT, Spectrogram, Wavelet transforms.
* **Machine Learning**: SVM, Random Forest, CNN for spectrogram-based classification.
* **Visualization**: Matplotlib, Seaborn.

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/DHANASHRI1221/Simulation-of-Micro-Doppler-Signatures-Drone-Detection.git
cd Simulation-of-Micro-Doppler-Signatures-Drone-Detection
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Simulation

```bash
python simulation.py
```

### 4️⃣ Train Model

```bash
python train.py --model cnn --epochs 50
```

### 5️⃣ Evaluate

```bash
python evaluate.py --model cnn
```

---

## 📊 Example Output

* Simulated micro-Doppler spectrograms.
* Classification results (Drone vs Non-Drone).
* Confusion matrices & ROC curves.

---

## 📌 Future Work

* Real radar data integration.
* Deep learning models for improved accuracy.
* Multi-class detection (bird, drone, human, etc.).


