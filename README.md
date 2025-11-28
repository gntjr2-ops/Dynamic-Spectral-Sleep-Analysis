# Wearable Multimodal Sleep Stage Classification via Dynamic Spectral–Temporal Feature Fusion

## Overview
Sleep quality is closely associated with overall health. Disorders such as sleep apnea, insomnia, and abnormal REM activity are linked to cardiovascular and stress-related diseases.  
Traditional sleep-stage evaluation relies on polysomnography (PSG), which is not suitable for long-term home monitoring due to its invasiveness and complexity.

This project proposes a wearable-based sleep stage classification algorithm using **ECG**, **PPG**, and **IMU** signals.  
The algorithm extracts **dynamic spectral–temporal features** to classify four sleep stages:

- **Deep Sleep (NREM3)**
- **Light Sleep (NREM1–2)**
- **REM Sleep**
- **Wake**

---

## Key Features

### 1. Multimodal Biosignal Fusion
**ECG**
- R-peak detection and HR estimation  
- HRV analysis: SDNN, RMSSD, LF/HF ratios  
- Autonomic nervous system activity indicators  

**PPG**
- Pulse morphology (rise time, reflection index)  
- Respiratory-synchronized amplitude modulation (RSA)  
- Low/High frequency spectral ratios  

**IMU**
- Sleep posture detection (left/right/supine)  
- Micro-motion analysis related to respiration  
- Activity Index within 30-second windows  

---

### 2. Dynamic Time–Frequency Feature Extraction
- STFT and Wavelet transforms applied to ECG/PPG  
- Power spectral density estimation  
- Frequency band energy ratio analysis  
- IMU-based separation of low-frequency breathing vs. high-frequency motions  
- Final feature vector combining HRV + PTT + RSA + motion metrics  

---

### 3. Rule-Based or Lightweight ML Classification
Sleep Stage Classification Rules:
- **Deep Sleep (NREM3)**: high HRV, minimal motion, stable PPG  
- **Light Sleep (NREM1–2)**: moderate HR, small movement  
- **REM**: high HR variability, irregular breathing, micro-movements  
- **Wake**: high HR, strong movement amplitude  

Supports:
- Heuristic rule-based logic  
- Lightweight classifiers (KNN, LDA, etc.)

---

### 4. Real-Time & Offline Analysis
- Real-time streaming evaluation every **30 seconds**  
- Offline processing of long-term data  
- Export to CSV or MAT  
- Hypnogram visualization  

---

## System Workflow

### 1. Wearable Device Setup
- ECG electrodes  
- Wrist PPG sensor  
- IMU-based motion sensor  
- 6–8 hours of overnight recording  

### 2. Data Acquisition & Preprocessing
- Sampling rates:  
  - ECG: **128 Hz**  
  - PPG: **64 Hz**  
  - IMU: **32 Hz**  
- Noise removal using moving average & band-pass filtering  

### 3. Dynamic Window Analysis
- 30–60 second windows  
- Extract features: HR, HRV, PTT, RSA, posture, movement  
- STFT / Wavelet for spectral analysis  

### 4. Sleep Stage Classification
- Feature fusion: HRV + PPG + IMU metrics  
- Rule-based or ML-based classification  
- Output labels: **Deep / Light / REM / Wake**

### 5. Visualization & Storage
- Hypnogram generation  
- Export abnormal breathing event logs  
- Store raw, filtered, and processed features  

---

## Applications
- **Clinical research**: PSG alternative/auxiliary indicators  
- **Consumer wearable devices**  
- **Sleep apnea / hypopnea detection**  
- **Mobile healthcare apps** (fatigue & stress index estimation)

---

## Expected Benefits
- Non-invasive and low-cost long-term sleep monitoring  
- Higher accuracy through multimodal ECG–PPG–IMU fusion  
- Real-time abnormal breathing detection  
- Suitable for continuous home monitoring  

---

## Example Repository Structure
