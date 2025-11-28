Wearable Multimodal Sleep Stage Classification via Dynamic Spectral–Temporal Feature Fusion
Overview

Sleep quality is closely related to overall health, and disorders such as sleep apnea, insomnia, and abnormal REM activity are strongly associated with cardiovascular and stress-related diseases.
Traditional sleep-stage assessment relies on polysomnography (PSG), which requires bulky equipment and is not suitable for continuous, long-term home monitoring.

This project proposes a wearable-device-based sleep stage classification algorithm using ECG (electrocardiogram), PPG (photoplethysmogram), and IMU (inertial measurement) signals.
The algorithm performs dynamic spectral–temporal feature extraction, enabling classification of four sleep stages:

Deep Sleep (NREM3)

Light Sleep (NREM1–2)

REM Sleep

Wake

Key Features
1. Multimodal Biosignal Fusion

ECG

R-peak detection and heart rate estimation

HRV (SDNN, RMSSD, LF/HF ratio)

PPG

Pulse morphology (rise time, reflection index)

Respiratory-synchronized amplitude modulation

Spectral ratio between low/high-frequency bands

IMU

Sleep posture estimation (left/right supine)

Micro-motion detection associated with respiration

Activity index within 30-second windows

2. Dynamic Time–Frequency Feature Extraction

STFT (Short-Time Fourier Transform) and Wavelet transform applied to ECG and PPG

Power spectral density estimation across LF/HF bands

IMU frequency separation for:

Low-frequency respiratory rhythm

High-frequency motion artifacts

Feature vector construction combining:

HRV + PPG-derived vascular indicators + IMU motion metrics

3. Rule-Based & Lightweight Machine Learning Classification

Sleep Stage Decision Rules

Deep Sleep (NREM3): High HRV, low movement, stable PPG

Light Sleep (NREM1–2): Moderate HR, small movements

REM: High HR variability, irregular breathing, micro-movements

Wake: Elevated HR, large movement amplitude

Can be used with:

Heuristic rules

Statistical classifiers (KNN, LDA)

Lightweight ML models suitable for wearables

4. Real-Time & Offline Analysis Support

Real-time classification using 30-second streaming windows

Long-term monitoring with offline processing

CSV/MAT export

Hypnogram generation

System Workflow
1. Wearable Device Setup

ECG electrode patches

Wrist-based PPG sensor

IMU-equipped wearable device

Continuous 6–8 hour recording during sleep

2. Data Acquisition & Preprocessing

Sampling Rates:

ECG: 128 Hz

PPG: 64 Hz

IMU: 32 Hz

Noise reduction via:

Moving average

Band-pass filtering

Baseline correction

3. Dynamic Window-Based Analysis

Sliding window: 30–60 seconds

Extract:

HR, HRV

PTT (Pulse Transit Time)

RSA (Respiratory Sinus Arrhythmia)

Sleep posture

Movement intensity

STFT/Wavelet for spectral indicators

4. Sleep Stage Classification

Combines ECG, PPG, IMU features

Rule-based logic or ML model outputs

Final labels: Deep / Light / REM / Wake

5. Visualization & Storage

Hypnogram for timeline visualization

Export abnormal breathing events

Store raw, filtered, and processed features

Applications

Clinical Research

Supplementary or alternative indicator to PSG

Long-term sleep quality tracking

Consumer Healthcare Devices

Smartwatches, fitness bands, research wearables

Apnea & Hypopnea Detection

Micro-motion and respiratory irregularity monitoring

Mobile Healthcare Apps

Generates fatigue and stress index based on sleep quality

Expected Benefits

Non-invasive, low-cost alternative to PSG

Improved accuracy through multimodal fusion (ECG + PPG + IMU)

Real-time detection of abnormal respiratory activity

Suitable for home-based long-term monitoring
