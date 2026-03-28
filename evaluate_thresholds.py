import pandas as pd
import numpy as np
import os
import csv

############# CHANGE HERE the paths for scores_csv and output_csv #############
# Configuration section: set model name and relevant input/output file paths
model_name = "OpenCV"
scores_csv = rf"D:\BiometricsProject\318172467\Results\scores_{model_name}.csv"
output_csv = rf"D:\BiometricsProject\318172467\Results\thresholds_{model_name}.csv"

# Load scores and remove failed comparisons
df = pd.read_csv(scores_csv)
df = df[df["score"] >= 0]  # remove failed comparisons

# Generate a range of thresholds to evaluate model performance
# thresholds = np.linspace(0.1, 1.0, 50) # for FaceNet and VGG-Face
thresholds = np.linspace(5, 30, 50)      # for OpenCV 

# Evaluate FMR and FNMR for each threshold value
results = []

for t in thresholds:
    # prediction: 1 if score < threshold, else 0 (lower = more similar)
    df["pred"] = (df["score"] < t).astype(int)

    # Calculate confusion matrix components
    TP = len(df[(df["label"] == 1) & (df["pred"] == 1)])
    FN = len(df[(df["label"] == 1) & (df["pred"] == 0)])
    FP = len(df[(df["label"] == 0) & (df["pred"] == 1)])
    TN = len(df[(df["label"] == 0) & (df["pred"] == 0)])

    FMR = FP / (FP + TN + 1e-6)  # +ε to avoid div by zero
    FNMR = FN / (TP + FN + 1e-6)

    results.append([round(t, 3), round(FMR, 4), round(FNMR, 4)])

# Save threshold evaluation results to CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["threshold", "FMR", "FNMR"])
    writer.writerows(results)

print(f"\n Threshold evaluation complete for {model_name}")
print(f" Saved to: {output_csv}")
