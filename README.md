# Audio Deepfake Detection Using Sonic Sleuth Model

An end-to-end AI-powered forensic application to detect synthetic speech and audio deepfakes utilizing the **Sonic Sleuth** Convolutional Neural Network (CNN) architecture and acoustic spectral artifact analysis.

---

## 📌 Overview

With the rapid advancement of Text-to-Speech (TTS) models and neural vocoders (such as HiFi-GAN, WaveGlow, MelGAN, and diffusion-based voice synthesis), synthetic voices have become increasingly indistinguishable from human speech to the human ear.

**Sonic Sleuth** is an acoustic forensic deep learning system engineered to detect subtle acoustic signatures and high-frequency synthesis anomalies introduced during neural audio generation:
- **Phase Inconsistencies & Temporal Discontinuities**: Subtle temporal flaws across synthesis boundaries.
- **High-Frequency Vocoder Discrepancies**: Abnormal energy roll-off and checkerboard spectral artifacts in frequencies exceeding 6 kHz.
- **Harmonic Modulation Flattening**: Detection of rigid pitch trajectories contrasting with organic human vocal cord vibrations and micro-tremors.
- **Mel-Spectrogram 2D Feature Modeling**: Multi-scale 2D CNN representations capturing spatial frequency-time dependencies.

---

## 🚀 Key Features

- **Sonic Sleuth CNN Architecture**: Deep 4-stage convolutional pipeline with batch normalization, spatial dropout, and adaptive average pooling.
- **Interactive Streamlit Web Dashboard**: User-friendly UI for uploading audio recordings or testing built-in synthetic/authentic voice samples.
- **Acoustic Visualizations**: Real-time rendering of time-domain audio waveforms and STFT log-spectrograms.
- **Forensic Breakdown**: Quantitative metrics including High-Frequency Ratio, Spectral Flatness, Spectral Centroid, and Zero-Crossing Rate.
- **Adjustable Sensitivity**: Fine-tune classification thresholds for high-security or high-recall environments.
- **Exportable Reports**: Download forensic findings directly in structured JSON format.

---

## 🛠️ Project Structure

```text
├── app.py              # Streamlit forensic dashboard & Sonic Sleuth inference pipeline
├── requirements.txt    # Project dependencies
├── .gitignore          # Git exclusion rules
└── README.md           # Documentation & model description
```

---

## ⚡ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Navz1407/ML-Project.git
cd ML-Project
```

### 2. Set Up Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🖥️ Running the Application

Launch the Streamlit interface:
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` to start analyzing audio samples.

---

## 🔬 Model Architecture

```text
Input (Mel-Spectrogram 2D Tensor)
   │
   ├── Conv2D(32, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) -> Dropout
   ├── Conv2D(64, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) -> Dropout
   ├── Conv2D(128, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) -> Dropout
   ├── Conv2D(256, 3x3) -> BatchNorm -> ReLU -> AdaptiveAvgPool2d((4,4))
   │
   ├── Flatten
   ├── Dense(128) -> ReLU -> Dropout(0.4)
   └── Dense(2) -> Softmax [Authentic vs. Deepfake]
```

---

## 📄 License
This project is open-source under the MIT License.
