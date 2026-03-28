import cv2
import os
import pandas as pd
import csv
import numpy as np

############# CHANGE HERE the paths for pairs_csv and output_csv #############
# Configuration section: set path to input pairs and output results
pairs_csv = r"D:\BiometricsProject\318172467\Results\pairs.csv"
output_csv = r"D:\BiometricsProject\318172467\Results\scores_OpenCV.csv"

# Load image pairs for face comparison
df = pd.read_csv(pairs_csv)
results = []

# Helper function: read an image in grayscale format
def read_gray_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img

# Iterate through all image pairs and evaluate similarity using OpenCV LBPH
for idx, row in df.iterrows():
    img1_path = row["img1"]
    img2_path = row["img2"]
    label = row["label"]

    try:
        img1 = read_gray_image(img1_path)
        img2 = read_gray_image(img2_path)

        # Create and train an LBPH face recognizer on the first image
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train([img1], np.array([0]))
        
        # Predict the second image and get similarity score (lower = more similar)
        label_pred, score = recognizer.predict(img2)  # Predict img2

        # LBPH gives a distance: lower = more similar
    except Exception as e:
        print(f"[!] Error comparing:\n  {img1_path}\n  {img2_path}\n  {e}")
        score = -1 # Indicate comparison failed

    results.append([img1_path, img2_path, label, score])
    if idx % 50 == 0:
        print(f"Compared {idx} pairs...")

# Save scores to output CSV file
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["img1", "img2", "label", "score"])
    writer.writerows(results)

print(f"\n Done! Scores saved to {output_csv}")
