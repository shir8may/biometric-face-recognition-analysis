# Biometric Face Recognition Analysis

This project focuses on evaluating and comparing multiple face recognition models using a biometric dataset.

## 📌 Overview

The project analyzes the performance of three different face recognition approaches:

* FaceNet (Deep Learning)
* VGG-Face (Deep Learning)
* OpenCV (LBPH – traditional method)

Each model is evaluated based on its ability to distinguish between matching and non-matching face pairs.

## ⚙️ Methodology

* Generated a dataset of face pairs (matching and non-matching)
* Computed similarity scores using embedding-based and traditional methods
* Evaluated performance across multiple thresholds
* Calculated key metrics:

  * False Match Rate (FMR)
  * False Non-Match Rate (FNMR)
  * Accuracy & Error Rate
* Identified optimal thresholds using Equal Error Rate (EER)
* Compared models using ROC curves

## 📊 Results

The analysis shows that:

* VGG-Face achieved the best overall performance with the highest accuracy and lowest false match rate
* FaceNet provided stable and balanced results
* OpenCV (LBPH) performed reasonably well but was less accurate compared to deep learning models

## 📁 Project Structure

* `src/` – Python scripts for analysis and evaluation
* `results/` – Output files (CSV, ROC curves, thresholds)
* `requirements.txt` – Project dependencies
* `final_report.pdf` – Full project report (detailed explanation and results)

## 🚀 Technologies

* Python
* DeepFace / FaceNet / VGG-Face
* OpenCV
* NumPy, Pandas
* Matplotlib

## ▶️ How to Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the main scripts:

   ```bash
   python run_deepface.py
   python run_opencv.py
   ```

> Note: Update file paths in the scripts according to your local environment.

## 🧠 Key Concepts

* Face embeddings
* Cosine similarity
* ROC curves
* Threshold tuning
* Biometric system evaluation
